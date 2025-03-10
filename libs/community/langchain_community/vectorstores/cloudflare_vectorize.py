import asyncio
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, TypeVar, Union
import uuid
import requests
import json
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

# Global constants
MAX_INSERT_SIZE = 1000
DEFAULT_TOP_K = 20
DEFAULT_TOP_K_WITH_MD_VALUES = 20
DEFAULT_DIMENSIONS = 1024
DEFAULT_METRIC = "cosine"

# Type variable for class methods that return CloudflareVectorize
VST = TypeVar("VST", bound="CloudflareVectorize")

# Class-level API token that can be set once
_api_token = None


# MARK: - VectorizeRecord
class VectorizeRecord:
    """Helper class to enforce Cloudflare Vectorize vector format.
    
    Attributes:
        id: Unique identifier for the vector
        text: The original text content
        values: The vector embedding values
        namespace: Optional namespace for the vector
        metadata: Optional metadata associated with the vector
    """

    def __init__(
            self,
            id: str,
            text: str,
            values: List[float],
            namespace: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize a VectorizeRecord.
        
        Args:
            id: Unique identifier for the vector
            text: The original text content
            values: The vector embedding values
            namespace: Optional namespace for the vector
            metadata: Optional metadata associated with the vector
        """
        self.id = id
        self.text = text
        self.values = values
        self.namespace = namespace
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for API requests."""
        vector_dict = {
            "id": self.id,
            "values": self.values,
            "text": self.text,
        }

        if self.namespace:
            vector_dict["namespace"] = self.namespace

        if self.metadata:
            vector_dict["metadata"] = self.metadata

        return vector_dict


# MARK: - CloudflareVectorize
class CloudflareVectorize(VectorStore):
    """Cloudflare Vectorize vector store.

    To use this, you need:
    1. Cloudflare Account ID
    2. Cloudflare API Token with appropriate permissions (Workers AI, Vectorize, D1)
    3. Index name
    4. D1 Database ID
    Reference: https://developers.cloudflare.com/api/resources/vectorize/
    """

    def __init__(
            self,
            embedding: Embeddings,
            account_id: str,
            api_token: Optional[str] = None,
            base_url: str = "https://api.cloudflare.com/client/v4",
            d1_database_id: Optional[str] = None,
            **kwargs: Any,
    ) -> None:
        """Initialize with Cloudflare credentials."""
        self.embedding = embedding
        self.account_id = account_id
        self.base_url = base_url
        self.d1_base_url = base_url
        self.d1_database_id = d1_database_id

        # Use the provided API token or get from class level
        self.api_token = api_token
        self.ai_api_token = kwargs.get("ai_api_token") or None
        self.vectorize_api_token = kwargs.get("vectorize_api_token") or None
        self.d1_api_token = kwargs.get("d1_api_token") or None

        # Set headers for Vectorize and D1
        self._headers = {
            "Authorization": f"Bearer {self.vectorize_api_token or self.api_token}",
            "Content-Type": "application/json",
        }
        self.d1_headers = {
            "Authorization": f"Bearer {self.d1_api_token or self.api_token}",
            "Content-Type": "application/json",
        }

        if not self.api_token \
                and (not self.ai_api_token
                     or not self.vectorize_api_token):
            raise ValueError(
                "Not enough API token values provided.  Please provide a global `api_token` or all of `ai_api_token`,`vectorize_api_token`.")

        if self.d1_database_id \
                and not self.api_token \
                and not self.d1_api_token:
            raise ValueError(
                "`d1_database_id` provided, but no global `api_token` provided and no `d1_api_token` provided.")

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding

    def _get_url(self, endpoint: str, index_name: str) -> str:
        """Get full URL for an API endpoint."""
        return f"{self.base_url}/accounts/{self.account_id}/vectorize/v2/indexes/{index_name}/{endpoint}"

    def _get_base_url(self, endpoint: str) -> str:
        """Get base URL for index management endpoints."""
        return f"{self.base_url}/accounts/{self.account_id}/vectorize/v2/indexes{endpoint}"

    def _get_d1_url(self, endpoint: str) -> str:
        """Get full URL for a D1 API endpoint."""
        return f"{self.d1_base_url}/accounts/{self.account_id}/d1/{endpoint}"

    def _get_d1_base_url(self, endpoint: str) -> str:
        """Get base URL for D1 API endpoints."""
        return f"{self.d1_base_url}/accounts/{self.account_id}/d1/{endpoint}"

    # MARK: - _combine_vectorize_and_d1_data
    def _combine_vectorize_and_d1_data(
            self,
            vector_data: List[Dict[str, Any]],
            d1_response: List[Dict[str, Any]]
    ) -> List[Document]:
        """Combine vector data from Vectorize API with text data from D1 database.

        Args:
            vector_data: List of vector data dictionaries from Vectorize API
            d1_response: Response from D1 database containing text data

        Returns:
            List of Documents with combined data from both sources
        """
        # Create a lookup dictionary for D1 text data by ID
        id_to_text = {}
        for item in d1_response:
            if "id" in item and "text" in item:
                id_to_text[item["id"]] = item["text"]

        documents = []
        for vector in vector_data:
            # Create a Document with the complete vector data in metadata
            vector_id = vector.get("id")
            vector_data = {
                "id": vector_id,
                "score": vector.get("score", 0.0),
            }

            # Add metadata if returned
            if "metadata" in vector:
                vector_data["metadata"] = vector.get("metadata", {})

            # Add namespace if returned
            if "namespace" in vector:
                vector_data["namespace"] = vector.get("namespace")

            # Add values if returned
            if "values" in vector:
                vector_data["values"] = vector.get("values", [])

            # Get the text content from D1 results
            text_content = id_to_text.get(vector_id, "")

            # Create a Document with the text content and vector data as metadata
            documents.append(Document(id=vector_id, page_content=text_content, metadata=vector_data))

        return documents

    # MARK: - _poll_mutation_status
    def _poll_mutation_status(self, index_name: str, mutation_id: str):
        err_cnt = 5
        err_lim = 0
        while True:
            try:
                response_index = self.get_index_info(index_name)
                err_cnt = 0
            except Exception as e:
                if err_cnt >= err_lim:
                    raise Exception("Index Mutation Error:", str(e))
                err_cnt += 1
                time.sleep(1)
                continue

            index_mutation_id = response_index.get("processedUpToMutation")
            if index_mutation_id == mutation_id:
                break
            time.sleep(1)

    # MARK: - _apoll_mutation_status
    async def _apoll_mutation_status(self, index_name: str, mutation_id: str):
        err_cnt = 5
        err_lim = 0
        while True:
            try:
                response_index = await self.aget_index_info(index_name)
                err_cnt = 0
            except Exception as e:
                if err_cnt >= err_lim:
                    raise Exception("Index Mutation Error:", str(e))
                err_cnt += 1
                await asyncio.sleep(1)
                continue

            index_mutation_id = response_index.get("processedUpToMutation")
            if index_mutation_id == mutation_id:
                break
            await asyncio.sleep(1)

    # MARK: - d1_create_table
    def d1_create_table(self, table_name: str, **kwargs) -> Dict[str, Any]:
        """Create a table in a D1 database using SQL schema.

        Args:
            database_id: ID of the database to create table in
            table_name: Name of the table to create

        Returns:
            Response data with query results
        """

        table_schema = f"""
        CREATE TABLE IF NOT EXISTS '{table_name}' (
            id TEXT PRIMARY KEY, 
            text TEXT, 
            namespace TEXT, 
            metadata TEXT
        )"""

        response = requests.post(
            self._get_d1_url(f"database/{self.d1_database_id}/query"),
            headers=self.d1_headers,
            json={"sql": table_schema},
        )

        return response.json().get("result", {})

    # MARK: - ad1_create_table
    async def ad1_create_table(self, table_name: str, **kwargs) -> Dict[str, Any]:
        """Asynchronously create a table in a D1 database using SQL schema.

        Args:
            database_id: ID of the database to create table in
            table_name: Name of the table to create

        Returns:
            Response data with query results
        """

        table_schema = f"""
        CREATE TABLE IF NOT EXISTS '{table_name}' (
            id TEXT PRIMARY KEY, 
            text TEXT, 
            namespace TEXT, 
            metadata TEXT
        )"""

        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self._get_d1_url(f"database/{self.d1_database_id}/query"),
                headers=self.d1_headers,
                json={"sql": table_schema},
                **kwargs
            )
            response.raise_for_status()
            response_data = response.json()

        return response_data.get("result", {})

    # MARK: - d1_drop_table
    def d1_drop_table(self, table_name: str, **kwargs) -> Dict[str, Any]:
        """Asynchronously delete a table from a D1 database.

        Args:
            table_name: Name of the table to delete

        Returns:
            Response data with query results
        """
        drop_query = f"DROP TABLE IF EXISTS '{table_name}'"

        response = requests.post(
            self._get_d1_url(f"database/{self.d1_database_id}/query"),
            headers=self.d1_headers,
            json={"sql": drop_query},
            **kwargs
        )

        response.raise_for_status()

        return response.json().get("result", {})

    # MARK: - ad1_drop_table
    async def ad1_drop_table(self, table_name: str, **kwargs) -> Dict[str, Any]:
        """Asynchronously delete a table from a D1 database.

        Args:
            table_name: Name of the table to delete

        Returns:
            Response data with query results
        """
        import httpx

        drop_query = f"DROP TABLE IF EXISTS '{table_name}'"

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self._get_d1_url(f"database/{self.d1_database_id}/query"),
                headers=self.d1_headers,
                json={"sql": drop_query},
                **kwargs
            )
            response.raise_for_status()
            response_data = response.json()

        return response_data.get("result", {})

    # MARK: - d1_upsert_texts
    def d1_upsert_texts(self, table_name: str, data: List[VectorizeRecord], **kwargs) -> Dict[str, Any]:
        """Insert or update text data in a D1 database table.

        Args:
            database_id: ID of the database to insert into
            table_name: Name of the table to insert data into
            data: List of dictionaries containing data to insert

        Returns:
            Response data with query results
        """
        if not data:
            return {"success": True, "changes": 0}

        statements = []
        for record in data:
            record_dict = record.to_dict()

            if "namespace" not in record_dict.keys():
                record_dict["namespace"] = ""

            if "metadata" not in record_dict.keys():
                record_dict["metadata"] = {}
            else:
                for k, v in record_dict['metadata'].items():
                    record_dict['metadata'][k] = v.replace("'", "''") if v else None

            statements.append(
                f"INSERT INTO '{table_name}' (id, text, namespace, metadata) " +
                f"VALUES (" +
                ", ".join(
                    [
                        f"'{x}'"
                        if x else "NULL"
                        for x in
                        [
                            record_dict["id"].replace("'", "''"),
                            record_dict["text"].replace("'", "''"),
                            record_dict["namespace"].replace("'", "''"),
                            json.dumps(record_dict["metadata"])
                        ]
                    ]
                ) + ")" +
                f"""
                ON CONFLICT (id) DO UPDATE SET
                text = excluded.text,
                namespace = excluded.namespace,
                metadata = excluded.metadata
                """
            )

        response = requests.post(
            self._get_d1_url(f"database/{self.d1_database_id}/query"),
            headers=self.d1_headers,
            json={
                "sql": ";\n".join(statements),
            },
            **kwargs
        )

        return response.json().get("result", {})

    # MARK: - ad1_upsert_texts
    async def ad1_upsert_texts(self, table_name: str, data: List[VectorizeRecord], **kwargs) -> Dict[str, Any]:
        """Asynchronously insert or update text data in a D1 database table.

        Args:
            database_id: ID of the database to insert into
            table_name: Name of the table to insert data into
            data: List of dictionaries containing data to insert

        Returns:
            Response data with query results
        """
        if not data:
            return {"success": True, "changes": 0}

        statements = []
        for record in data:
            record_dict = record.to_dict()
            for k, v in record_dict['metadata'].items():
                record_dict['metadata'][k] = v.replace("'", "''") if v else None

            statements.append(
                f"INSERT INTO '{table_name}' (id, text, namespace, metadata) " +
                f"VALUES (" +
                ", ".join(
                    [
                        f"'{x}'" if x else "NULL"
                        for x in
                        [
                            record_dict["id"].replace("'", "''"),
                            record_dict["text"].replace("'", "''"),
                            record_dict["namespace"].replace("'", "''"),
                            json.dumps(record_dict["metadata"])
                        ]
                    ]
                ) + ")" +
                f"""
                ON CONFLICT (id) DO UPDATE SET
                text = excluded.text,
                namespace = excluded.namespace,
                metadata = excluded.metadata
                """
            )

        import httpx

        # Execute with parameters
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self._get_d1_url(f"database/{self.d1_database_id}/query"),
                headers=self.d1_headers,
                json={
                    "sql": ";\n".join(statements),
                },
                **kwargs
            )
            response.raise_for_status()
            response_data = response.json()

        return response_data.get("result", {})

    # MARK: - d1_get_by_ids
    def d1_get_by_ids(self, index_name: str, ids: List[str], **kwargs) -> List:
        """Retrieve text data from a D1 database table.

        Args:
            database_id: ID of the database to query
            table_name: Name of the table to query
            filter_params: Optional dictionary of filter parameters

        Returns:
            Response data with query results
        """
        # query D1 for raw results
        placeholders = ','.join(['?'] * len(ids))  # Creates "?,?,?..." for the right number of IDs

        sql = f"""
            SELECT * FROM '{index_name}'
            WHERE id IN ({placeholders})
        """

        response = requests.post(
            self._get_d1_url(f"database/{self.d1_database_id}/query"),
            headers=self.d1_headers,
            json={"sql": sql, "params": ids},
            **kwargs
        )

        response_data = response.json()
        d1_results = response_data.get("result", {})
        if len(d1_results) == 0:
            return []

        d1_results_records = d1_results[0].get("results", [])

        return d1_results_records

    # MARK: - ad1_get_texts
    async def ad1_get_by_ids(self, index_name: str, ids: List[str], **kwargs) -> \
            Dict[str, Any]:
        """Asynchronously retrieve text data from a D1 database table.

        Args:
            database_id: ID of the database to query
            table_name: Name of the table to query
            filter_params: Optional dictionary of filter parameters

        Returns:
            Response data with query results
        """

        # query D1 for raw results
        placeholders = ','.join(['?'] * len(ids))  # Creates "?,?,?..." for the right number of IDs

        sql = f"""
            SELECT * FROM '{index_name}'
            WHERE id IN ({placeholders})
        """

        import httpx

        # Execute the query
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self._get_d1_url(f"database/{self.d1_database_id}/query"),
                headers=self.d1_headers,
                json={
                    "sql": sql,
                    "params": ids,
                },
                **kwargs
            )
            response.raise_for_status()
            response_data = response.json()

        return response_data.get("result", {})

    # MARK: - d1_delete
    def d1_delete(self, index_name: str, ids: List[str], **kwargs) -> Dict[str, Any]:
        """Delete data from a D1 database table.

        Args:
            database_id: ID of the database containing the table
            table_name: Name of the table to delete from
            filter_params: Dictionary of parameters to filter rows to delete

        Returns:
            Response data with deletion results
        """
        # query D1 for raw results
        placeholders = ','.join(['?'] * len(ids))  # Creates "?,?,?..." for the right number of IDs

        sql = f"""
                    DELETE FROM '{index_name}'
                    WHERE id IN ({placeholders})
                """

        response = requests.post(
            self._get_d1_url(f"database/{self.d1_database_id}/query"),
            headers=self.d1_headers,
            json={"sql": sql, "params": ids},
            **kwargs
        )

        return response.json().get("result", {})

    # MARK: - ad1_delete
    async def ad1_delete(self, index_name: str, ids: List[str], **kwargs) -> Dict[str, Any]:
        """Asynchronously delete data from a D1 database table.

        Args:
            database_id: ID of the database containing the table
            table_name: Name of the table to delete from
            filter_params: Dictionary of parameters to filter rows to delete

        Returns:
            Response data with deletion results
        """
        # query D1 for raw results
        placeholders = ','.join(['?'] * len(ids))  # Creates "?,?,?..." for the right number of IDs

        sql = f"""
                    DELETE FROM '{index_name}'
                    WHERE id IN ({placeholders})
                """

        import httpx

        # Execute the deletion
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self._get_d1_url(f"database/{self.d1_database_id}/query"),
                headers=self.d1_headers,
                json={
                    "sql": sql,
                    "params": ids,
                },
                **kwargs
            )
            response.raise_for_status()
            response_data = response.json()

        return response_data.get("result", {})

    # MARK: - add_texts
    def add_texts(
            self,
            texts: Iterable[str],
            metadatas: Optional[List[dict]] = None,
            ids: Optional[List[str]] = None,
            namespaces: Optional[List[str]] = None,
            insert_only: bool = False,
            index_name: str = None,
            include_d1: bool = True,
            wait: bool = True,
            **kwargs: Any,
    ) -> Dict:
        """Add texts to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of ids to associate with the texts.
            namespaces: Optional list of namespaces for each vector.
            insert_only: If True, uses the insert endpoint which will fail if vectors with
                        the same IDs already exist. If False (default), uses upsert which
                        will create or update vectors.
            index_name: Name of the Vectorize index.
            wait: If True (default), wait until vectors are ready.

        Returns:
            Operation response with ids of added texts.

        Raises:
            ValueError: If the number of texts exceeds MAX_INSERT_SIZE.
            :param include_d1:
        """
        if not index_name:
            raise ValueError("index_name must be provided")

        # Convert texts to list if it's not already
        texts_list = list(texts)

        # Check if the number of texts exceeds the maximum allowed
        if len(texts_list) > MAX_INSERT_SIZE:
            raise ValueError(
                f"Number of texts ({len(texts_list)}) exceeds maximum allowed ({MAX_INSERT_SIZE})"
            )

        # Generate embeddings for the texts
        embeddings = self.embedding.embed_documents(texts_list)

        # Generate IDs if not provided
        if ids is None or list(set(ids)) == [None]:
            ids = [str(uuid.uuid4()) for _ in texts_list]

        # Prepare vectors with metadata if provided
        vectors = []
        for i, (embedding, id, text) in enumerate(zip(embeddings, ids, texts_list)):
            # Get metadata if provided
            metadata = {}
            if metadatas is not None and i < len(metadatas):
                metadata = metadatas[i].copy()

            # Get namespace if provided
            namespace = None
            if namespaces is not None and i < len(namespaces):
                namespace = namespaces[i]

            # Create VectorizeRecord
            vector = VectorizeRecord(
                id=id,
                text=text,
                values=embedding,
                namespace=namespace or None,
                metadata=metadata
            )

            vectors.append(vector)

        # Choose endpoint based on insert_only parameter
        endpoint = "insert" if insert_only else "upsert"

        # Convert vectors to newline-delimited JSON
        ndjson_data = "\n".join(json.dumps(vector.to_dict()) for vector in vectors)

        # Copy headers and set correct content type for NDJSON
        headers = self._headers.copy()
        headers["Content-Type"] = "application/x-ndjson"

        # Make API call to insert/upsert vectors
        response = requests.post(
            self._get_url(endpoint, index_name),
            headers=headers,  # Use the NDJSON-specific headers
            data=ndjson_data.encode('utf-8'),
            **kwargs
        )
        response.raise_for_status()

        if include_d1 and self.d1_database_id:
            self.d1_create_table(
                table_name=index_name,
                **kwargs
            )

            # add values to D1Database
            self.d1_upsert_texts(
                table_name=index_name,
                data=vectors,
                **kwargs
            )

        mutation_response = response.json()
        mutation_id = mutation_response.get("result", {}).get("mutationId")

        if wait and mutation_id:
            self._poll_mutation_status(
                index_name=index_name,
                mutation_id=mutation_id,
            )

        return {
            **mutation_response,
            "ids": ids
        }

    # MARK: - aadd_texts
    async def aadd_texts(
            self,
            texts: Iterable[str],
            metadatas: Optional[List[dict]] = None,
            ids: Optional[List[str]] = None,
            namespaces: Optional[List[str]] = None,
            insert_only: bool = False,
            index_name: str = None,
            include_d1: bool = True,
            wait: bool = True,
            **kwargs: Any,
    ) -> Dict:
        """Asynchronously add texts to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of ids to associate with the texts.
            namespaces: Optional list of namespaces for each vector.
            insert_only: If True, uses the insert endpoint which will fail if vectors with
                        the same IDs already exist. If False (default), uses upsert which
                        will create or update vectors.
            index_name: Name of the Vectorize index.

        Returns:
            List of ids from adding the texts into the vectorstore.

        Raises:
            ValueError: If the number of texts exceeds MAX_INSERT_SIZE.
        """
        if not index_name:
            raise ValueError("index_name must be provided")

        # Convert texts to list if it's not already
        texts_list = list(texts)

        # Check if the number of texts exceeds the maximum allowed
        if len(texts_list) > MAX_INSERT_SIZE:
            raise ValueError(
                f"Number of texts ({len(texts_list)}) exceeds maximum allowed ({MAX_INSERT_SIZE})"
            )

        # Generate embeddings for the texts
        embeddings = await self.embedding.aembed_documents(texts_list)

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts_list]

        # Prepare vectors with metadata if provided
        vectors = []
        for i, (embedding, id, text) in enumerate(zip(embeddings, ids, texts_list)):
            # Get metadata if provided
            metadata = {}
            if metadatas is not None and i < len(metadatas):
                metadata = metadatas[i].copy()

            # Get namespace if provided
            namespace = None
            if namespaces is not None and i < len(namespaces):
                namespace = namespaces[i]

            # Create VectorizeRecord
            vector = VectorizeRecord(
                id=id,
                text=text,
                values=embedding,
                namespace=namespace,
                metadata=metadata
            )

            vectors.append(vector)

        # Choose endpoint based on insert_only parameter
        endpoint = "insert" if insert_only else "upsert"

        # Convert vectors to newline-delimited JSON
        ndjson_data = "\n".join(json.dumps(vector.to_dict()) for vector in vectors)

        # Import httpx here to avoid dependency issues
        import httpx

        # Make API call to insert/upsert vectors
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self._get_url(endpoint, index_name),
                headers=self._headers,
                content=ndjson_data.encode('utf-8'),
            )
            response.raise_for_status()

        mutation_response = response.json()
        mutation_id = mutation_response.get("result", {}).get("mutationId")

        if include_d1 and self.d1_database_id:
            # create D1 table if not exists
            await self.ad1_create_table(
                table_name=index_name
            )

            # add values to D1Database
            await self.ad1_upsert_texts(
                table_name=index_name,
                data=vectors,
            )

        if wait and mutation_id:
            await self._apoll_mutation_status(
                index_name=index_name,
                mutation_id=mutation_id,
            )

        return {
            **mutation_response,
            "ids": ids
        }

    # MARK: - similarity_search
    def similarity_search(
            self,
            query: str,
            index_name: str,
            k: int = DEFAULT_TOP_K,
            filter: Optional[Dict[str, Any]] = None,
            namespace: Optional[str] = None,
            return_metadata: str = "none",
            return_values: bool = False,
            include_d1: bool = True,
            **kwargs: Any
    ) -> List[Document]:
        """Search for similar documents to a query string."""
        if not index_name:
            raise ValueError("index_name must be provided")

        docs_and_scores = \
            self.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter,
                namespace=namespace,
                index_name=index_name,
                return_metadata=return_metadata,
                return_values=return_values,
                include_d1=include_d1,
                **kwargs
            )

        return docs_and_scores[0]

    # MARK: - similarity_search_with_score
    def similarity_search_with_score(
            self,
            query: str,
            index_name: str,
            k: int = DEFAULT_TOP_K,
            filter: Optional[Dict[str, Any]] = None,
            namespace: Optional[str] = None,
            return_metadata: str = "none",
            return_values: bool = False,
            include_d1: bool = True,
            **kwargs: Any
    ) -> Tuple[List[Document], List[float]]:
        """Search for similar vectors to a query string and return with scores."""
        if not index_name:
            raise ValueError("index_name must be provided")

        # Generate embedding for the query
        query_embedding = self.embedding.embed_query(query)

        # Prepare search request
        search_request = {
            "vector": query_embedding,
            "topK": k if not return_metadata and not return_values else min(k, DEFAULT_TOP_K_WITH_MD_VALUES),
        }

        if namespace:
            search_request["namespace"] = namespace

        # Add filter if provided
        if filter:
            search_request["filter"] = filter

        # Add metadata return preference
        if return_metadata:
            search_request["returnMetadata"] = return_metadata

        # Add vector values return preference
        if return_values:
            search_request["returnValues"] = return_values

        # Make API call to query vectors
        response = requests.post(
            self._get_url("query", index_name),
            headers=self._headers,
            json=search_request,
            **kwargs
        )
        response.raise_for_status()
        results = response.json().get("result", {}).get("matches", [])

        if include_d1 and self.d1_database_id:
            # query D1 for raw results
            ids = [x.get("id") for x in results]

            d1_results_records = \
                self.d1_get_by_ids(
                    index_name=index_name,
                    ids=ids,
                    **kwargs
                )
        else:
            d1_results_records = []

        # Use _combine_vectorize_and_d1_data to create documents
        documents = self._combine_vectorize_and_d1_data(results, d1_results_records)

        # Extract scores from results
        scores = [result.get("score", 0.0) for result in results]

        return documents, scores

    # MARK: - asimilarity_search
    async def asimilarity_search(
            self,
            query: str,
            k: int = DEFAULT_TOP_K,
            filter: Optional[Dict[str, Any]] = None,
            namespace: Optional[str] = None,
            index_name: str = None,
            include_d1: bool = True,
            **kwargs: Any,
    ) -> List[Document]:
        """Asynchronously search for similar documents to a query string.

        Args:
            query: Query string to search for.
            k: Number of results to return.
            filter: Optional metadata filter.
            namespace: Optional namespace to search in.
            index_name: Name of the Vectorize index.

        Returns:
            List of Documents most similar to the query.
        """
        if not index_name:
            raise ValueError("index_name must be provided")

        results = await self.asimilarity_search_with_score(
            query=query,
            k=k,
            filter=filter,
            namespace=namespace,
            index_name=index_name,
            include_d1=include_d1,
            **kwargs
        )
        return results[0]

    # MARK: - asimilarity_search_with_score
    async def asimilarity_search_with_score(
            self,
            query: str,
            k: int = DEFAULT_TOP_K,
            filter: Optional[Dict[str, Any]] = None,
            namespace: Optional[str] = None,
            index_name: str = None,
            return_metadata: Optional[str] = "none",
            return_values: bool = False,
            include_d1: bool = True,
            **kwargs: Any,
    ) -> Tuple[List[Document], List[float]]:
        """Asynchronously search for similar vectors to a query string and return with scores.

        Args:
            query: Query string to search for.
            k: Number of results to return.
            filter: Optional metadata filter expression to limit results.
            namespace: Optional namespace to search in.
            index_name: Name of the Vectorize index.
            return_metadata: Controls metadata return: "none" (default), "indexed", or "all".
            return_values: Whether to return vector values (default: False).

        Returns:
            Tuple of (List of Documents, List of similarity scores).

            Each Document has:
            - Empty page_content (as Vectorize doesn't store text)
            - metadata containing the complete vector data:
              - id: Identifier for the vector
              - metadata: Any metadata associated with the vector (if requested)
              - namespace: The namespace the vector belongs to
              - score: The similarity score
              - values: The vector values (if requested)
        """
        if not index_name:
            raise ValueError("index_name must be provided")

        # Generate embedding for the query
        query_embedding = self.embedding.embed_query(query)

        # Prepare search request
        search_request = {
            "vector": query_embedding,
            "top_k": k,
        }

        if namespace:
            search_request["namespace"] = namespace

        # Add filter if provided
        if filter:
            search_request["filter"] = filter

        # Add metadata return preference
        if return_metadata:
            search_request["return_metadata"] = return_metadata

        # Add vector values return preference
        if return_values:
            search_request["return_values"] = return_values

        # Import httpx here to avoid dependency issues
        import httpx

        # Make API call to query vectors
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self._get_url("query", index_name),
                headers=self._headers,
                json=search_request,
            )
            response.raise_for_status()
            response_data = response.json()

        results = response_data.get("result", {}).get("matches", [])

        if include_d1 and self.d1_database_id:
            # query D1 for raw results
            ids = [x.get("id") for x in results]

            d1_results_records = \
                await self.ad1_get_by_ids(
                    index_name=index_name,
                    ids=ids,
                    **kwargs
                )
        else:
            d1_results_records = []

        # Use _combine_vectorize_and_d1_data to create documents
        documents = self._combine_vectorize_and_d1_data(results, d1_results_records)

        # Extract scores from results
        scores = [result.get("score", 0.0) for result in results]

        return documents, scores

    # MARK: - delete
    def delete(
            self,
            ids: List[str],
            index_name: str = None,
            include_d1: bool = True,
            wait: bool = True,
            **kwargs: Any
    ) -> Dict:
        """Delete vectors by ID from the vectorstore.

        Args:
            ids: List of ids to delete.
            index_name: Name of the Vectorize index.
        """
        if not index_name:
            raise ValueError("index_name must be provided")

        delete_request = {"ids": ids}

        response = requests.post(
            self._get_url("delete_by_ids", index_name),
            headers=self._headers,
            json=delete_request,
            **kwargs
        )
        response.raise_for_status()

        if include_d1 and self.d1_database_id:
            self.d1_delete(
                index_name=index_name,
                ids=ids
            )

        mutation_response = response.json()
        mutation_id = mutation_response.get("result", {}).get("mutationId")

        if wait and mutation_id:
            self._poll_mutation_status(
                index_name=index_name,
                mutation_id=mutation_id,
            )

        return {
            **mutation_response,
            "ids": ids
        }

    # MARK: - adelete
    async def adelete(
            self,
            ids: List[str],
            index_name: str = None,
            include_d1: bool = True,
            wait: bool = True,
            **kwargs: Any
    ) -> Dict:
        """Asynchronously delete vectors by ID from the vectorstore.

        Args:
            ids: List of ids to delete.
            index_name: Name of the Vectorize index.
        """
        if not index_name:
            raise ValueError("index_name must be provided")

        delete_request = {"ids": ids}

        # Make API call to delete vectors
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self._get_url("delete_by_ids", index_name),
                headers=self._headers,
                json=delete_request,
            )
        response.raise_for_status()

        mutation_response = response.json()
        mutation_id = mutation_response.get("result", {}).get("mutationId")

        if include_d1 and self.d1_database_id:
            await self.ad1_delete(
                table_name=index_name,
                ids=ids,
                **kwargs
            )

        if wait and mutation_id:
            await self._apoll_mutation_status(
                index_name=index_name,
                mutation_id=mutation_id,
            )

        return {
            **mutation_response,
            "ids": ids
        }

    # MARK: - get_by_ids
    def get_by_ids(
            self,
            ids: List[str],
            index_name: str = None,
            include_d1: bool = True,
            **kwargs
    ) -> List[Document]:
        """Get vectors by their IDs.

        Args:
            ids: List of vector IDs to retrieve.
            index_name: Name of the Vectorize index.

        Returns:
            List of VectorizeRecord objects containing vector data.
        """
        if not index_name:
            raise ValueError("index_name must be provided")

        get_request = {"ids": ids}

        # Get vector data from Vectorize API
        response = requests.post(
            self._get_url("get_by_ids", index_name),
            headers=self._headers,
            json=get_request,
            **kwargs
        )
        response.raise_for_status()

        vector_data = response.json().get("result", {})

        if include_d1 and self.d1_database_id:
            # Get text data from D1 database
            d1_response = self.d1_get_by_ids(
                index_name=index_name,
                ids=ids,
                **kwargs
            )
        else:
            d1_response = []

        # Combine data into VectorizeRecord objects
        documents = \
            self._combine_vectorize_and_d1_data(
                vector_data,
                d1_response
            )

        return documents

    # MARK: - aget_by_ids
    async def aget_by_ids(
            self,
            ids: List[str],
            index_name: str = None,
            include_d1: bool = True,
            **kwargs
    ) -> List[Document]:
        """Asynchronously get vectors by their IDs.

        Args:
            ids: List of vector IDs to retrieve.
            index_name: Name of the Vectorize index.

        Returns:
            List of vector data.
        """
        if not index_name:
            raise ValueError("index_name must be provided")

        get_request = {"ids": ids}

        # Get vector data from Vectorize API
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self._get_url("get_by_ids", index_name),
                headers=self._headers,
                json=get_request,
                **kwargs
            )
        response.raise_for_status()

        if include_d1 and self.d1_database_id:
            d1_response = await self.ad1_get_by_ids(
                index_name=index_name,
                ids=ids,
                **kwargs
            )
        else:
            d1_response = []

        documents = \
            self._combine_vectorize_and_d1_data(
                response.json().get("result", {}).get("vectors", []),
                d1_response
            )

        return documents

    # MARK: - get_index_info
    def get_index_info(self, index_name: str, **kwargs) -> Dict[str, Any]:
        """Get information about the current index.

        Returns:
            Dictionary containing index information.
        """
        if not index_name:
            raise ValueError("index_name must be provided")

        response = requests.get(
            self._get_url("info", index_name),
            headers=self._headers,
            **kwargs
        )
        response.raise_for_status()

        return response.json().get("result", {})

    # MARK: - aget_index_info
    async def aget_index_info(self, index_name: str, **kwargs) -> Dict[str, Any]:
        """Asynchronously get information about the current index.

        Returns:
            Dictionary containing index information.
        """
        if not index_name:
            raise ValueError("index_name must be provided")

        # Import httpx here to avoid dependency issues
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.get(
                self._get_url("info", index_name),
                headers=self._headers,
                **kwargs
            )
            response.raise_for_status()

            response_data = response.json()

        return response_data.get("result", {})

    # MARK: - create_metadata_index
    def create_metadata_index(self, property_name: str, index_type: str = "string", index_name: str = None, **kwargs) -> \
            Dict[
                str, Any]:
        """Create a metadata index for a specific property.

        Args:
            property_name: The metadata property to index.
            index_type: The type of index to create (default: "string").
            index_name: Name of the Vectorize index.

        Returns:
            Response data with mutation ID.
        """
        if not index_name:
            raise ValueError("index_name must be provided")

        response = requests.post(
            f"{self._get_url('metadata_index/create', index_name)}",
            headers=self._headers,
            json={
                "propertyName": property_name,
                "indexType": index_type
            },
            **kwargs
        )
        response.raise_for_status()

        # todo: add wait

        return response.json().get("result", {})

    # MARK: - acreate_metadata_index
    async def acreate_metadata_index(self, property_name: str, index_type: str = "string", index_name: str = None,
                                     **kwargs) -> \
            Dict[str, Any]:
        """Asynchronously create a metadata index for a specific property.

        Args:
            property_name: The metadata property to index.
            index_type: The type of index to create (default: "string").
            index_name: Name of the Vectorize index.

        Returns:
            Response data with mutation ID.
        """
        if not index_name:
            raise ValueError("index_name must be provided")

        # Import httpx here to avoid dependency issues
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._get_url('metadata_index/create', index_name)}",
                headers=self._headers,
                json={
                    "propertyName": property_name,
                    "indexType": index_type
                },
                **kwargs
            )
            response.raise_for_status()

            response_data = response.json()

        # todo: add wait

        return response_data.get("result", {})

    # MARK: - list_metadata_indexes
    def list_metadata_indexes(self, index_name: str = None, **kwargs) -> List[Dict[str, str]]:
        """List all metadata indexes for the current index.

        Returns:
            List of metadata indexes with their property names and index types.
        """
        if not index_name:
            raise ValueError("index_name must be provided")

        response = requests.get(
            f"{self._get_url('metadata_index/list', index_name)}",
            headers=self._headers,
            **kwargs
        )
        response.raise_for_status()

        return response.json().get("result", {}).get("metadataIndexes", [])

    # MARK: - alist_metadata_indexes
    async def alist_metadata_indexes(self, index_name: str = None, **kwargs) -> List[Dict[str, str]]:
        """Asynchronously list all metadata indexes for the current index.

        Returns:
            List of metadata indexes with their property names and index types.
        """
        if not index_name:
            raise ValueError("index_name must be provided")

        # Import httpx here to avoid dependency issues
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self._get_url('metadata_index/list', index_name)}",
                headers=self._headers,
                **kwargs
            )
            response.raise_for_status()

            response_data = response.json()

        return response_data.get("result", {}).get("metadataIndexes", [])

    # MARK: - delete_metadata_index
    def delete_metadata_index(self, property_name: str, index_name: str = None, **kwargs) -> Dict[str, Any]:
        """Delete a metadata index.

        Args:
            property_name: The metadata property index to delete.
            index_name: Name of the Vectorize index.

        Returns:
            Response data with mutation ID.
        """
        if not index_name:
            raise ValueError("index_name must be provided")

        response = requests.post(
            f"{self._get_url('metadata_index/delete', index_name)}",
            headers=self._headers,
            json={"property": property_name},
            **kwargs
        )
        response.raise_for_status()

        return response.json().get("result", {})

    # MARK: - adelete_metadata_index
    async def adelete_metadata_index(self, property_name: str, index_name: str = None, **kwargs) -> Dict[str, Any]:
        """Asynchronously delete a metadata index.

        Args:
            property_name: The metadata property to remove indexing for.
            index_name: Name of the Vectorize index.

        Returns:
            Response data with mutation ID.
        """
        if not index_name:
            raise ValueError("index_name must be provided")

        # Import httpx here to avoid dependency issues
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._get_url('metadata_index/delete', index_name)}",
                headers=self._headers,
                json={"propertyName": property_name},
                **kwargs
            )
            response.raise_for_status()

            response_data = response.json()

        return response_data.get("result", {})

    # MARK: - get_index
    def get_index(self, index_name: str, **kwargs) -> Dict[str, Any]:
        """Get information about the current Vectorize index.

        This endpoint returns details about the index configuration.
        https://developers.cloudflare.com/api/resources/vectorize/subresources/indexes/methods/get/

        Returns:
            Dictionary containing index configuration and details.
        """
        if not index_name:
            raise ValueError("index_name must be provided")

        response = requests.get(
            f"{self.base_url}/accounts/{self.account_id}/vectorize/v2/indexes/{index_name}",
            headers=self._headers,
            **kwargs
        )
        response.raise_for_status()

        return response.json().get("result", {})

    # MARK: - aget_index
    async def aget_index(self, index_name: str, **kwargs) -> Dict[str, Any]:
        """Asynchronously get information about the current Vectorize index.

        Returns:
            Dictionary containing index information.
        """
        if not index_name:
            raise ValueError("index_name must be provided")

        # Import httpx here to avoid dependency issues
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/accounts/{self.account_id}/vectorize/v2/indexes/{index_name}",
                headers=self._headers,
                **kwargs
            )
            response.raise_for_status()

            response_data = response.json()

        return response_data.get("result", {})

    # MARK: - create_index
    def create_index(
            self,
            index_name: str,
            dimensions: int = DEFAULT_DIMENSIONS,
            metric: str = DEFAULT_METRIC,
            description: Optional[str] = None,
            include_d1: bool = True,
            **kwargs
    ) -> Dict[str, Any]:
        """Create a new Vectorize index.

        Args:
            index_name: Name for the new index
            dimensions: Number of dimensions for the vector embeddings
            metric: Distance metric to use (e.g., "cosine", "euclidean")
            description: Optional description for the index

        Returns:
            Response data from the API
            :param include_d1:
        """
        # Use provided token or get class level token
        token = self.vectorize_api_token or self.api_token

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        data = {
            "name": index_name,
            "config": {
                "dimensions": dimensions,
                "metric": metric,
            }
        }

        if description:
            data["description"] = description

        # Check if index already exists - but handle various error states
        try:
            r = self.get_index(index_name, **kwargs)
            if r:
                raise ValueError(f"Index {index_name} already exists")
        except requests.exceptions.HTTPError as e:
            # If 404 Not Found or 410 Gone, we can create the index
            if e.response.status_code in [404, 410]:
                pass  # Index doesn't exist or was removed, so we can create it
            else:
                # Re-raise for other HTTP errors
                raise

        # Create the index
        response = requests.post(
            f"{self.base_url}/accounts/{self.account_id}/vectorize/v2/indexes",
            headers=headers,
            json=data,
            **kwargs
        )
        response.raise_for_status()

        if include_d1 and self.d1_database_id:
            # Create D1 table if not exists
            self.d1_create_table(
                table_name=index_name,
                **kwargs
            )

        # todo: add wait

        return response.json().get("result", {})

    # MARK: - acreate_index
    async def acreate_index(
            self,
            index_name: str,
            dimensions: int = DEFAULT_DIMENSIONS,
            metric: str = DEFAULT_METRIC,
            description: Optional[str] = None,
            include_d1: bool = True,
            **kwargs
    ) -> Dict[str, Any]:
        """Asyncronously Create a new Vectorize index.

        Args:
            index_name: Name for the new index
            dimensions: Number of dimensions for the vector embeddings
            metric: Distance metric to use (e.g., "cosine", "euclidean")
            description: Optional description for the index

        Returns:
            Response data from the API
        """
        # Use provided token or get class level token
        token = self.vectorize_api_token or self.api_token

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        data = {
            "name": index_name,
            "config": {
                "dimensions": dimensions,
                "metric": metric,
            }
        }

        if description:
            data["description"] = description

        # Import httpx here to avoid dependency issues
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/accounts/{self.account_id}/vectorize/v2/indexes",
                headers=headers,
                json=data,
                **kwargs
            )
            response.raise_for_status()

            response_data = response.json()

        if include_d1 and self.d1_database_id:
            # create D1 table if not exists
            await self.ad1_create_table(
                table_name=index_name,
                **kwargs
            )

        # todo: add wait

        return response_data.get("result", {})

    # MARK: - list_indexes
    def list_indexes(self, **kwargs) -> List[Dict[str, Any]]:
        """List all Vectorize indexes for an account."""
        # Use provided token or get class level token
        token = self.api_token or self.vectorize_api_token

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        response = requests.get(
            f"{self.base_url}/accounts/{self.account_id}/vectorize/v2/indexes",
            headers=headers,
            **kwargs
        )
        response.raise_for_status()

        return response.json().get("result", [])

    # MARK: - alist_indexes
    async def alist_indexes(
            self,
            **kwargs
    ) -> List[Dict[str, Any]]:
        """Asynchronously list all Vectorize indexes for an account."""
        # Use provided token or get class level token
        token = self.vectorize_api_token or self.api_token

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        # Import httpx here to avoid dependency issues
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/accounts/{self.account_id}/vectorize/v2/indexes",
                headers=headers,
                **kwargs
            )
            response.raise_for_status()

            response_data = response.json()

        return response_data.get("result", [])

    # MARK: - delete_index
    def delete_index(
            self,
            index_name: str,
            include_d1: bool = True,
            **kwargs
    ) -> Dict[str, Any]:
        """Delete a Vectorize index."""
        # Use provided token or get class level token
        token = self.vectorize_api_token or self.api_token

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        response = requests.delete(
            f"{self.base_url}/accounts/{self.account_id}/vectorize/v2/indexes/{index_name}",
            headers=headers,
            **kwargs
        )
        response.raise_for_status()

        if include_d1 and self.d1_database_id:
            # delete D1 table if exists
            self.d1_drop_table(
                table_name=index_name
            )

        return response.json().get("result", {})

    # MARK: - adelete_index
    async def adelete_index(
            self,
            index_name: str,
            include_d1: bool = True,
            **kwargs
    ) -> Dict[str, Any]:
        """Asynchronously delete a Vectorize index."""
        # Use provided token or get class level token
        token = self.vectorize_api_token or self.api_token

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        # Import httpx here to avoid dependency issues
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.delete(
                f"{self.base_url}/accounts/{self.account_id}/vectorize/v2/indexes/{index_name}",
                headers=headers,
                **kwargs
            )
            response.raise_for_status()

            response_data = response.json()

        if include_d1 and self.d1_database_id:
            await self.ad1_drop_table(
                table_name=index_name
            )

        return response_data.get("result", {})

    # MARK: - from_texts
    @classmethod
    def from_texts(
            cls: Type[VST],
            texts: List[str],
            embedding: Embeddings,
            metadatas: Optional[List[dict]] = None,
            ids: Optional[List[str]] = None,
            namespaces: Optional[List[str]] = None,
            insert_only: bool = False,
            account_id: Optional[str] = None,
            d1_database_id: Optional[str] = None,
            index_name: Optional[str] = None,
            dimensions: int = DEFAULT_DIMENSIONS,
            metric: str = DEFAULT_METRIC,
            api_token: Optional[str] = None,
            **kwargs: Any,
    ) -> VST:
        """Create a CloudflareVectorize vectorstore from raw texts."""
        # Check for required parameters
        if not account_id or not index_name:
            raise ValueError("account_id and index_name must be provided")

        if not ids:
            ids = [uuid.uuid4().hex for _ in range(len(texts))]

        ai_api_token = kwargs.pop("ai_api_token", None)
        vectorize_api_token = kwargs.pop("vectorize_api_token", None)
        d1_api_token = kwargs.pop("d1_api_token", None)

        vectorstore = cls(
            embedding=embedding,
            account_id=account_id,
            d1_database_id=d1_database_id,
            api_token=api_token,
            ai_api_token=ai_api_token,
            vectorize_api_token=vectorize_api_token,
            d1_api_token=d1_api_token
        )

        # create vectorize index if not exists
        vectorstore.create_index(
            index_name=index_name,
            dimensions=dimensions,
            metric=metric,
            **kwargs
        )

        vectorstore.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
            namespaces=namespaces,
            insert_only=insert_only,
            index_name=index_name,
            **kwargs,
        )

        return vectorstore

    # Update the afrom_texts method similarly
    async def afrom_texts(
            self: Type[VST],
            texts: List[str],
            embedding: Embeddings,
            metadatas: Optional[List[dict]] = None,
            ids: Optional[List[str]] = None,
            namespaces: Optional[List[str]] = None,
            insert_only: bool = False,
            account_id: Optional[str] = None,
            index_name: Optional[str] = None,
            dimensions: int = DEFAULT_DIMENSIONS,
            metric: str = DEFAULT_METRIC,
            api_token: Optional[str] = None,
            **kwargs: Any,
    ) -> VST:
        """Asynchronously create a CloudflareVectorize vectorstore from raw texts."""
        # Check for required parameters
        if not account_id or not index_name:
            raise ValueError("account_id and index_name must be provided")

        vectorstore = self(
            embedding=embedding,
            account_id=account_id,
            api_token=kwargs.get("vectorize_api_token") or api_token,
            vectorize_api_token=kwargs.get("vectorize_api_token") or None,
            d1_api_token=kwargs.get("d1_api_token") or None,
            ai_api_token=kwargs.get("ai_api_token") or None,
            **kwargs,
        )

        await vectorstore.acreate_index(
            index_name=index_name,
            dimensions=dimensions,
            metric=metric,
            **kwargs
        )

        await vectorstore.aadd_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
            namespaces=namespaces,
            insert_only=insert_only,
            index_name=index_name,
            **kwargs
        )

        return vectorstore

    # MARK: - add_documents
    def add_documents(
            self,
            documents: List[Document],
            namespaces: Optional[List[str]] = None,
            insert_only: bool = False,
            index_name: str = None,
            wait: bool = True,
            **kwargs: Any,
    ) -> Dict:
        """Add documents to the vectorstore.

        Args:
            documents: List of Documents to add to the vectorstore.
            namespaces: Optional list of namespaces for each vector.
            insert_only: If True, uses the insert endpoint which will fail if vectors with
                        the same IDs already exist. If False (default), uses upsert which
                        will create or update vectors.
            index_name: Name of the Vectorize index.
            wait: If True (default), poll until all documents have been added.

        Returns:
            Operation response with ids of added documents.

        Raises:
            ValueError: If the number of documents exceeds MAX_INSERT_SIZE.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # Use document IDs if they exist, falling back to provided ids
        if "ids" not in kwargs:
            ids = [doc.id or str(uuid.uuid4()) for doc in documents]
        else:
            ids = None

        return self.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
            namespaces=namespaces,
            insert_only=insert_only,
            index_name=index_name,
            wait=wait,
            **kwargs
        )

    # MARK: - aadd_documents
    async def aadd_documents(
            self,
            documents: List[Document],
            ids: Optional[List[str]] = None,
            namespaces: Optional[List[str]] = None,
            insert_only: bool = False,
            index_name: str = None,
            include_d1: bool = True,
            **kwargs: Any,
    ) -> Dict:
        """Asynchronously add documents to the vectorstore.

        Args:
            documents: List of Documents to add to the vectorstore.
            ids: Optional list of ids to associate with the documents.
            namespaces: Optional list of namespaces for each vector.
            insert_only: If True, uses the insert endpoint which will fail if vectors with
                        the same IDs already exist. If False (default), uses upsert which
                        will create or update vectors.
            index_name: Name of the Vectorize index.

        Returns:
            List of ids from adding the documents into the vectorstore.

        Raises:
            ValueError: If the number of documents exceeds MAX_INSERT_SIZE.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # Use document IDs if they exist, falling back to provided ids
        if "ids" not in kwargs:
            ids = [doc.id for doc in documents]

        return await self.aadd_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
            namespaces=namespaces,
            insert_only=insert_only,
            index_name=index_name,
            include_d1=include_d1,
            **kwargs
        )

    # MARK: - from_documents
    @classmethod
    def from_documents(
            cls: Type[VST],
            documents: List[Document],
            embedding: Embeddings,
            namespaces: Optional[List[str]] = None,
            insert_only: bool = False,
            account_id: Optional[str] = None,
            d1_database_id: Optional[str] = None,
            index_name: Optional[str] = None,
            dimensions: int = DEFAULT_DIMENSIONS,
            metric: str = DEFAULT_METRIC,
            api_token: Optional[str] = None,
            **kwargs: Any,
    ) -> VST:
        """Create a CloudflareVectorize vectorstore from documents.

        Args:
            documents: List of Documents to add to the vectorstore.
            embedding: Embedding function to use to embed the documents.
            ids: Optional list of ids to associate with the documents.
            namespaces: Optional list of namespaces for each vector.
            insert_only: If True, uses insert instead of upsert (default: False).
            account_id: Cloudflare account ID.
            index_name: Name of the Vectorize index.
            dimensions: Number of dimensions for vectors when creating a new index.
            metric: Distance metric to use when creating a new index.

        Returns:
            CloudflareVectorize vectorstore.

        Raises:
            ValueError: If the number of documents exceeds MAX_INSERT_SIZE.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        if "ids" not in kwargs:
            ids = [doc.id for doc in documents]
        else:
            ids = None

        return cls.from_texts(
            texts=texts,
            embedding=embedding,
            metadatas=metadatas,
            ids=ids,
            namespaces=namespaces,
            insert_only=insert_only,
            account_id=account_id,
            d1_database_id=d1_database_id,
            index_name=index_name,
            dimensions=dimensions,
            metric=metric,
            api_token=api_token,
            **kwargs,
        )

    # MARK: - afrom_documents
    @classmethod
    async def afrom_documents(
            cls: Type[VST],
            documents: List[Document],
            embedding: Embeddings,
            namespaces: Optional[List[str]] = None,
            insert_only: bool = False,
            account_id: Optional[str] = None,
            index_name: Optional[str] = None,
            dimensions: int = DEFAULT_DIMENSIONS,
            metric: str = DEFAULT_METRIC,
            api_token: Optional[str] = None,
            **kwargs: Any,
    ) -> VST:
        """Asynchronously create a CloudflareVectorize vectorstore from documents.

        Args:
            documents: List of Documents to add to the vectorstore.
            embedding: Embedding function to use to embed the documents.
            ids: Optional list of ids to associate with the documents.
            namespaces: Optional list of namespaces for each vector.
            insert_only: If True, uses insert instead of upsert (default: False).
            account_id: Cloudflare account ID.
            index_name: Name of the Vectorize index.
            dimensions: Number of dimensions for vectors when creating a new index.
            metric: Distance metric to use when creating a new index.

        Returns:
            CloudflareVectorize vectorstore.

        Raises:
            ValueError: If the number of documents exceeds MAX_INSERT_SIZE.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        if "ids" not in kwargs:
            ids = [doc.id for doc in documents]
        else:
            ids = None

        return await cls.afrom_texts(
            texts=texts,
            embedding=embedding,
            metadatas=metadatas,
            ids=ids,
            namespaces=namespaces,
            insert_only=insert_only,
            account_id=account_id,
            index_name=index_name,
            dimensions=dimensions,
            metric=metric,
            api_token=cls.vectorize_api_token or api_token,
            **kwargs,
        )

    # https://developers.cloudflare.com/api/resources/vectorize/
