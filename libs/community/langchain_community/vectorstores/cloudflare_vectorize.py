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
            d1_database_id: str,
            api_token: Optional[str] = None,
            base_url: str = "https://api.cloudflare.com/client/v4",
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
    ) -> List[VectorizeRecord]:
        """Combine vector data from Vectorize API with text data from D1 database.
        
        Args:
            vector_data: List of vector data dictionaries from Vectorize API
            d1_response: Response from D1 database containing text data
            
        Returns:
            List of VectorizeRecord objects with combined data
        """
        # Create a lookup dictionary for D1 text data by ID
        d1_data_by_id = {}
        # if d1_response and "results" in d1_response:
        for item in d1_response:
            if "id" in item:
                d1_data_by_id[item["id"]] = item

        # Combine data into VectorizeRecord objects
        records = []
        for vector in vector_data:
            vector_id = vector.get("id")
            values = vector.get("values", [])
            metadata = vector.get("metadata", {})
            namespace = vector.get("namespace", None)

            # Merge with D1 data if available
            if vector_id in d1_data_by_id:
                d1_item = d1_data_by_id[vector_id]
                text = d1_item.get("text", "")

                # Update metadata with any additional fields from D1
                for key, value in d1_item.items():
                    if key not in ["id", "text"] and key not in metadata:
                        metadata[key] = value
            else:
                text = ""  # No matching text found in D1

            # Create VectorizeRecord object
            record = VectorizeRecord(
                id=vector_id,
                text=text,
                values=values,
                namespace=namespace,
                metadata=metadata
            )
            records.append(record)

        return records

    # MARK: - d1_create_table
    def d1_create_table(self, database_id: str, table_name: str) -> Dict[str, Any]:
        """Create a table in a D1 database using SQL schema.
        
        Args:
            database_id: ID of the database to create table in
            table_name: Name of the table to create 
            
        Returns:
            Response data with query results
        """

        table_schema = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id TEXT PRIMARY KEY, 
            text TEXT, 
            namespace TEXT, 
            metadata TEXT
        )"""

        response = requests.post(
            self._get_d1_url(f"database/{database_id}/query"),
            headers=self.d1_headers,
            json={"sql": table_schema},
        )

        return response.json().get("result", {})

    # MARK: - ad1_create_table
    async def ad1_create_table(self, database_id: str, table_name: str) -> Dict[str, Any]:
        """Asynchronously create a table in a D1 database using SQL schema.
        
        Args:
            database_id: ID of the database to create table in
            table_name: Name of the table to create 
            
        Returns:
            Response data with query results
        """

        table_schema = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id TEXT PRIMARY KEY, 
            text TEXT, 
            namespace TEXT, 
            metadata TEXT
        )"""

        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self._get_d1_url(f"database/{database_id}/query"),
                headers=self.d1_headers,
                json={"sql": table_schema},
            )
            response.raise_for_status()
            response_data = response.json()

        return response_data.get("result", {})

    # MARK: - d1_drop_table
    def d1_drop_table(self, database_id: str, table_name: str) -> Dict[str, Any]:
        """Asynchronously delete a table from a D1 database.
        
        Args:
            database_id: ID of the database containing the table
            table_name: Name of the table to delete
            
        Returns:
            Response data with query results
        """
        drop_query = f"DROP TABLE IF EXISTS {table_name}"

        response = requests.post(
            self._get_d1_url(f"database/{database_id}/query"),
            headers=self.d1_headers,
            json={"sql": drop_query},
        )

        return response.json().get("result", {})

    # MARK: - ad1_drop_table
    async def ad1_drop_table(self, database_id: str, table_name: str) -> Dict[str, Any]:
        """Asynchronously delete a table from a D1 database.
        
        Args:
            database_id: ID of the database containing the table
            table_name: Name of the table to delete
            
        Returns:
            Response data with query results
        """
        import httpx

        drop_query = f"DROP TABLE IF EXISTS {table_name}"

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self._get_d1_url(f"database/{database_id}/query"),
                headers=self.d1_headers,
                json={"sql": drop_query},
            )
            response.raise_for_status()
            response_data = response.json()

        return response_data.get("result", {})

    # MARK: - d1_upsert_texts
    def d1_upsert_texts(self, database_id: str, table_name: str, data: List[VectorizeRecord]) -> Dict[str, Any]:
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
            for k, v in record_dict['metadata'].items():
                record_dict['metadata'][k] = v.replace("'", "''") if v else None

            statements.append(
                f"INSERT INTO {table_name} (id, text, namespace, metadata) " +
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

        response = requests.post(
            self._get_d1_url(f"database/{database_id}/query"),
            headers=self.d1_headers,
            json={
                "sql": ";\n".join(statements),
            },
        )

        return response.json().get("result", {})

    # MARK: - ad1_upsert_texts
    async def ad1_upsert_texts(self, database_id: str, table_name: str, data: List[dict]) -> Dict[str, Any]:
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

        # Generate parameterized SQL statement for insert or update
        # Assuming first dict in data has all the fields we need
        fields = list(data[0].keys())
        placeholders = [f":{field}" for field in fields]

        # Create upsert query
        # This assumes an id field exists and is used as primary key
        sql = f"""
        INSERT INTO {table_name} ({', '.join(fields)})
        VALUES ({', '.join(placeholders)})
        ON CONFLICT (id, index_name) DO UPDATE SET
        {', '.join([f"{field} = :{field}" for field in fields if field != 'id'])}
        """

        import httpx

        # Execute with parameters
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self._get_d1_url(f"database/{database_id}/query"),
                headers=self.d1_headers,
                json={
                    "sql": sql,
                    "params": data,
                    "batch": True,  # Process as a batch operation
                },
            )
            response.raise_for_status()
            response_data = response.json()

        return response_data.get("result", {})

    # MARK: - d1_get_by_ids
    def d1_get_by_ids(self, index_name: str, ids: List[str]) -> List:
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
            SELECT * FROM {index_name}
            WHERE id IN ({placeholders})
        """

        response = requests.post(
            self._get_d1_url(f"database/{self.d1_database_id}/query"),
            headers=self.d1_headers,
            json={"sql": sql, "params": ids},
        )

        response_data = response.json()
        d1_results = response_data.get("result", {})
        if len(d1_results) == 0:
            return []

        d1_results_records = d1_results[0].get("results", [])

        return d1_results_records

    # MARK: - ad1_get_texts
    async def ad1_get_by_ids(self, database_id: str, table_name: str, filter_params: Optional[Dict[str, Any]] = None) -> \
            Dict[str, Any]:
        """Asynchronously retrieve text data from a D1 database table.
        
        Args:
            database_id: ID of the database to query
            table_name: Name of the table to query
            filter_params: Optional dictionary of filter parameters
            
        Returns:
            Response data with query results
        """

        # TODO: refactor this

        # Start with a basic query
        sql = f"SELECT * FROM {table_name}"
        params = {}

        # Add WHERE clauses if filter_params provided
        if filter_params:
            where_clauses = []
            for key, value in filter_params.items():
                where_clauses.append(f"{key} = :{key}")
                params[key] = value

            if where_clauses:
                sql += " WHERE " + " AND ".join(where_clauses)

        import httpx

        # Execute the query
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self._get_d1_url(f"database/{database_id}/query"),
                headers=self.d1_headers,
                json={
                    "sql": sql,
                    "params": params,
                },
            )
            response.raise_for_status()
            response_data = response.json()

        return response_data.get("result", {})

    # MARK: - d1_delete
    def d1_delete(self, index_name: str, ids: List[str]) -> Dict[str, Any]:
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
                    DELETE FROM {index_name}
                    WHERE id IN ({placeholders})
                """

        response = requests.post(
            self._get_d1_url(f"database/{self.d1_database_id}/query"),
            headers=self.d1_headers,
            json={"sql": sql, "params": ids},
        )

        return response.json().get("result", {})

    # MARK: - ad1_delete
    async def ad1_delete(self, database_id: str, table_name: str, filter_params: Dict[str, Any]) -> Dict[str, Any]:
        """Asynchronously delete data from a D1 database table.
        
        Args:
            database_id: ID of the database containing the table
            table_name: Name of the table to delete from
            filter_params: Dictionary of parameters to filter rows to delete
            
        Returns:
            Response data with deletion results
        """
        # Build DELETE query with WHERE clauses
        where_clauses = []
        params = {}

        for key, value in filter_params.items():
            where_clauses.append(f"{key} = :{key}")
            params[key] = value

        sql = f"DELETE FROM {table_name}"
        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)

        import httpx

        # Execute the deletion
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self._get_d1_url(f"database/{database_id}/query"),
                headers=self.d1_headers,
                json={
                    "sql": sql,
                    "params": params,
                },
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
            **kwargs: Any,
    ) -> List[str]:
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
        embeddings = self.embedding.embed_documents(texts_list)

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

        # Copy headers and set correct content type for NDJSON
        headers = self._headers.copy()
        headers["Content-Type"] = "application/x-ndjson"

        # Make API call to insert/upsert vectors
        response = requests.post(
            self._get_url(endpoint, index_name),
            headers=headers,  # Use the NDJSON-specific headers
            data=ndjson_data.encode('utf-8'),
        )
        response.raise_for_status()

        # add values to D1Database
        self.d1_upsert_texts(
            database_id=self.d1_database_id,
            table_name=index_name,
            data=vectors,
        )

        return ids

    # MARK: - aadd_texts
    async def aadd_texts(
            self,
            texts: Iterable[str],
            metadatas: Optional[List[dict]] = None,
            ids: Optional[List[str]] = None,
            namespaces: Optional[List[str]] = None,
            insert_only: bool = False,
            index_name: str = None,
            **kwargs: Any,
    ) -> List[str]:
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
        embeddings = await self.embedding.embed_documents(texts_list)

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

            # create D1 table if not exists
            await self.ad1_create_table(
                database_id=self.d1_database_id,
                table_name=index_name
            )

            # add values to D1Database
            await self.d1_upsert_texts(
                database_id=self.d1_database_id,
                table_name=index_name,
                data=vectors,
            )

        return ids

    # MARK: - similarity_search
    def similarity_search(
            self,
            query: str,
            index_name: str,
            k: int = DEFAULT_TOP_K,
            filter: Optional[Dict[str, Any]] = None,
            namespace: Optional[str] = None,
            return_metadata: str = "none",
            return_values: bool = False
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
                return_values=return_values
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
        )
        response.raise_for_status()

        results = response.json().get("result", {}).get("matches", [])

        # query D1 for raw results
        ids = [x.get("id") for x in results]

        d1_results_records = \
            self.d1_get_by_ids(
                index_name=index_name,
                ids=ids
            )

        # Create a mapping of id to text content from D1 results
        id_to_text = {}
        for item in d1_results_records:
            if "id" in item and "text" in item:
                id_to_text[item["id"]] = item["text"]

        documents = []
        scores = []

        for result in results:
            # Create a Document with the complete vector data in metadata
            vector_id = result.get("id")
            vector_data = {
                "id": vector_id,
                "score": result.get("score", 0.0),
            }

            # Add metadata if returned
            if "metadata" in result:
                vector_data["metadata"] = result.get("metadata", {})

            # Add namespace if returned
            if "namespace" in result:
                vector_data["namespace"] = result.get("namespace")

            # Add values if returned
            if "values" in result:
                vector_data["values"] = result.get("values", [])

            # Get the text content from D1 results
            text_content = id_to_text.get(vector_id, "")

            # Create a Document with the text content and vector data as metadata
            documents.append(Document(page_content=text_content, metadata=vector_data))
            scores.append(result.get("score", 0.0))

        return documents, scores

    # MARK: - asimilarity_search
    async def asimilarity_search(
            self,
            query: str,
            k: int = DEFAULT_TOP_K,
            filter: Optional[Dict[str, Any]] = None,
            namespace: Optional[str] = None,
            index_name: str = None,
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
            query=query, k=k, filter=filter, namespace=namespace, index_name=index_name, **kwargs
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

        documents = []
        scores = []

        for result in results:
            # Create a Document with the complete vector data in metadata
            vector_data = {
                "id": result.get("id"),
                "score": result.get("score", 0.0),
            }

            # Add metadata if returned
            if "metadata" in result:
                vector_data["metadata"] = result.get("metadata", {})

            # Add namespace if returned
            if "namespace" in result:
                vector_data["namespace"] = result.get("namespace")

            # Add values if returned
            if "values" in result:
                vector_data["values"] = result.get("values", [])

            # Create a Document with empty page content and the vector data as metadata
            documents.append(Document(page_content="", metadata=vector_data))
            scores.append(result.get("score", 0.0))

        return documents, scores

    # MARK: - delete
    def delete(
            self,
            ids: List[str],
            index_name: str = None,
            **kwargs: Any
    ) -> None:
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
        )
        response.raise_for_status()

        self.d1_delete(
            index_name=index_name,
            ids=ids
        )

        return response

    # MARK: - adelete
    async def adelete(
            self,
            ids: List[str],
            index_name: str = None,
            **kwargs: Any
    ) -> None:
        """Asynchronously delete vectors by ID from the vectorstore.
        
        Args:
            ids: List of ids to delete.
            index_name: Name of the Vectorize index.
        """
        if not index_name:
            raise ValueError("index_name must be provided")

        delete_request = {"ids": ids}

        # Make API call to delete vectors
        response = requests.post(
            self._get_url("delete_by_ids", index_name),
            headers=self._headers,
            json=delete_request,
        )
        response.raise_for_status()

        await self.ad1_delete_texts(
            database_id=self.d1_database_id,
            table_name=index_name,
            ids=ids
        )

    # MARK: - get_by_ids
    def get_by_ids(
            self,
            ids: List[str],
            index_name: str = None
    ) -> List[VectorizeRecord]:
        # TODO: this needs to return a LangChain Document
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
        )
        response.raise_for_status()

        vector_data = response.json().get("result", {})

        # Get text data from D1 database
        d1_response = self.d1_get_by_ids(
            index_name=index_name,
            ids=ids
        )

        # Combine data into VectorizeRecord objects
        records = \
            self._combine_vectorize_and_d1_data(
                vector_data,
                d1_response
            )

        return records

    # MARK: - aget_by_ids
    async def aget_by_ids(
            self,
            ids: List[str],
            index_name: str = None
    ) -> List[Dict[str, Any]]:
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
        response = requests.post(
            self._get_url("get_by_ids", index_name),
            headers=self._headers,
            json=get_request,
        )
        response.raise_for_status()

        return response.json().get("result", {}).get("vectors", [])

    # MARK: - get_index_info
    def get_index_info(self, index_name: str) -> Dict[str, Any]:
        """Get information about the current index.
        
        Returns:
            Dictionary containing index information.
        """
        if not index_name:
            raise ValueError("index_name must be provided")

        response = requests.get(
            self._get_url("info", index_name),
            headers=self._headers,
        )
        response.raise_for_status()

        return response.json().get("result", {})

    # MARK: - aget_index_info
    async def aget_index_info(self, index_name: str) -> Dict[str, Any]:
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
            )
            response.raise_for_status()

            response_data = response.json()

        return response_data.get("result", {})

    # MARK: - create_metadata_index
    def create_metadata_index(self, property_name: str, index_type: str = "string", index_name: str = None) -> Dict[
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
        )
        response.raise_for_status()

        return response.json().get("result", {})

    # MARK: - acreate_metadata_index
    async def acreate_metadata_index(self, property_name: str, index_type: str = "string", index_name: str = None) -> \
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
            )
            response.raise_for_status()

            response_data = response.json()

        return response_data.get("result", {})

    # MARK: - list_metadata_indexes
    def list_metadata_indexes(self, index_name: str = None) -> List[Dict[str, str]]:
        """List all metadata indexes for the current index.
        
        Returns:
            List of metadata indexes with their property names and index types.
        """
        if not index_name:
            raise ValueError("index_name must be provided")

        response = requests.get(
            f"{self._get_url('metadata_index/list', index_name)}",
            headers=self._headers,
        )
        response.raise_for_status()

        return response.json().get("result", {}).get("metadataIndexes", [])

    # MARK: - alist_metadata_indexes
    async def alist_metadata_indexes(self, index_name: str = None) -> List[Dict[str, str]]:
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
            )
            response.raise_for_status()

            response_data = response.json()

        return response_data.get("result", {}).get("metadataIndexes", [])

    # MARK: - delete_metadata_index
    def delete_metadata_index(self, property_name: str, index_name: str = None) -> Dict[str, Any]:
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
        )
        response.raise_for_status()

        return response.json().get("result", {})

    # MARK: - adelete_metadata_index
    async def adelete_metadata_index(self, property_name: str, index_name: str = None) -> Dict[str, Any]:
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
            )
            response.raise_for_status()

            response_data = response.json()

        return response_data.get("result", {})

    # MARK: - get_index
    def get_index(self, index_name: str) -> Dict[str, Any]:
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
        )
        response.raise_for_status()

        return response.json().get("result", {})

    # MARK: - aget_index
    async def aget_index(self, index_name: str) -> Dict[str, Any]:
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
    ) -> Dict[str, Any]:
        """Create a new Vectorize index.
        
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

        # Check if index already exists - but handle various error states
        try:
            r = self.get_index(index_name)
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
        )
        response.raise_for_status()

        # Create D1 table if not exists
        self.d1_create_table(
            database_id=self.d1_database_id,
            table_name=index_name
        )

        return response.json().get("result", {})

    # MARK: - acreate_index
    @classmethod
    async def acreate_index(
            cls,
            account_id: str,
            index_name: str,
            dimensions: int = DEFAULT_DIMENSIONS,
            metric: str = DEFAULT_METRIC,
            base_url: str = "https://api.cloudflare.com/client/v4",
            description: Optional[str] = None,
            api_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Asynchronously create a new Vectorize index."""
        # Use provided token or get class level token
        token = api_token or cls.get_api_token()

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
                f"{base_url}/accounts/{account_id}/vectorize/v2/indexes",
                headers=headers,
                json=data,
            )
            response.raise_for_status()

            response_data = response.json()

            # create D1 table if not exists
            await cls.ad1_create_table(
                database_id=cls.d1_database_id,
                table_name=index_name
            )

        return response_data.get("result", {})

    # MARK: - list_indexes
    @classmethod
    def list_indexes(
            cls,
            account_id: str,
            base_url: str = "https://api.cloudflare.com/client/v4",
            api_token: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List all Vectorize indexes for an account."""
        # Use provided token or get class level token
        token = api_token or cls.get_api_token()

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        response = requests.get(
            f"{base_url}/accounts/{account_id}/vectorize/v2/indexes",
            headers=headers,
        )
        response.raise_for_status()

        return response.json().get("result", [])

    # MARK: - alist_indexes
    @classmethod
    async def alist_indexes(
            cls,
            account_id: str,
            base_url: str = "https://api.cloudflare.com/client/v4",
            api_token: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Asynchronously list all Vectorize indexes for an account."""
        # Use provided token or get class level token
        token = api_token or cls.get_api_token()

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        # Import httpx here to avoid dependency issues
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{base_url}/accounts/{account_id}/vectorize/v2/indexes",
                headers=headers,
            )
            response.raise_for_status()

            response_data = response.json()

        return response_data.get("result", [])

    # MARK: - delete_index
    def delete_index(
            self,
            index_name: str,
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
        )
        response.raise_for_status()

        # delete D1 table if exists
        self.d1_drop_table(
            database_id=self.d1_database_id,
            table_name=index_name
        )

        return response.json().get("result", {})

    # MARK: - adelete_index
    async def adelete_index(
            self,
            index_name: str,
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
            )
            response.raise_for_status()

            response_data = response.json()

            await self.ad1_drop_table(
                database_id=self.d1_database_id,
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
            index_name: Optional[str] = None,
            create_if_missing: bool = False,
            dimensions: int = DEFAULT_DIMENSIONS,
            metric: str = DEFAULT_METRIC,
            api_token: Optional[str] = None,
            **kwargs: Any,
    ) -> VST:
        """Create a CloudflareVectorize vectorstore from raw texts."""
        # Check for required parameters
        if not account_id or not index_name:
            raise ValueError("account_id and index_name must be provided")

        # Use provided token or get class level token
        token = api_token or cls.get_api_token()

        # create vectorize index if not exists
        cls.create_index(
            account_id=account_id,
            index_name=index_name,
            dimensions=dimensions,
            metric=metric,
            api_token=cls.vectorize_api_token or token
        )

        vectorstore = cls(
            embedding=embedding,
            account_id=account_id,
            api_token=cls.vectorize_api_token or token,
            **kwargs,
        )

        vectorstore.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
            namespaces=namespaces,
            insert_only=insert_only,
            index_name=index_name
        )

        return vectorstore

    # Update the afrom_texts method similarly
    @classmethod
    async def afrom_texts(
            cls: Type[VST],
            texts: List[str],
            embedding: Embeddings,
            metadatas: Optional[List[dict]] = None,
            ids: Optional[List[str]] = None,
            namespaces: Optional[List[str]] = None,
            insert_only: bool = False,
            account_id: Optional[str] = None,
            index_name: Optional[str] = None,
            create_if_missing: bool = False,
            dimensions: int = DEFAULT_DIMENSIONS,
            metric: str = DEFAULT_METRIC,
            api_token: Optional[str] = None,
            **kwargs: Any,
    ) -> VST:
        """Asynchronously create a CloudflareVectorize vectorstore from raw texts."""
        # Check for required parameters
        if not account_id or not index_name:
            raise ValueError("account_id and index_name must be provided")

        # Use provided token or get class level token
        token = api_token or cls.get_api_token()

        await cls.acreate_index(
            account_id=account_id,
            index_name=index_name,
            dimensions=dimensions,
            metric=metric,
            api_token=cls.vectorize_api_token or token
        )

        vectorstore = cls(
            embedding=embedding,
            account_id=account_id,
            api_token=cls.vectorize_api_token or token,
            **kwargs,
        )

        await vectorstore.aadd_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
            namespaces=namespaces,
            insert_only=insert_only,
            index_name=index_name
        )

        return vectorstore

    # MARK: - add_embeddings
    def add_embeddings(
            self,
            vectors: List[VectorizeRecord],
            insert_only: bool = False,
            index_name: str = None,
            **kwargs: Any,
    ) -> List[str]:
        """Add pre-computed embeddings to the vectorstore.
        
        This method allows adding vectors directly without computing embeddings.
        
        Args:
            vectors: List of VectorizeRecord objects to add to the vectorstore.
            insert_only: If True, uses the insert endpoint which will fail if vectors with 
                        the same IDs already exist. If False (default), uses upsert which
                        will create or update vectors.
            index_name: Name of the Vectorize index.
            
        Returns:
            List of ids from adding the vectors into the vectorstore.
            
        Raises:
            ValueError: If the number of vectors exceeds MAX_INSERT_SIZE.
        """
        if not index_name:
            raise ValueError("index_name must be provided")

        # Check if the number of vectors exceeds the maximum allowed
        if len(vectors) > MAX_INSERT_SIZE:
            raise ValueError(
                f"Number of vectors ({len(vectors)}) exceeds maximum allowed ({MAX_INSERT_SIZE})"
            )

        # Convert VectorizeRecord objects to dictionaries
        vector_dicts = [vector.to_dict() for vector in vectors]

        # Choose endpoint based on insert_only parameter
        endpoint = "insert" if insert_only else "upsert"

        # Convert vectors to newline-delimited JSON
        ndjson_data = "\n".join(json.dumps(vector_dict) for vector_dict in vector_dicts)

        # Copy headers and set correct content type for NDJSON
        headers = self._headers.copy()
        headers["Content-Type"] = "application/x-ndjson"

        # Make API call to insert/upsert vectors
        response = requests.post(
            self._get_url(endpoint, index_name),
            headers=headers,  # Use the NDJSON-specific headers
            data=ndjson_data.encode('utf-8'),
        )
        response.raise_for_status()

        # Return the IDs of the added vectors
        return [vector.id for vector in vectors]

    # MARK: - aadd_embeddings
    async def aadd_embeddings(
            self,
            vectors: List[VectorizeRecord],
            insert_only: bool = False,
            index_name: str = None,
            **kwargs: Any,
    ) -> List[str]:
        """Asynchronously add pre-computed embeddings to the vectorstore.
        
        This method allows adding vectors directly without computing embeddings.
        
        Args:
            vectors: List of VectorizeRecord objects to add to the vectorstore.
            insert_only: If True, uses the insert endpoint which will fail if vectors with 
                        the same IDs already exist. If False (default), uses upsert which
                        will create or update vectors.
            index_name: Name of the Vectorize index.
            
        Returns:
            List of ids from adding the vectors into the vectorstore.
            
        Raises:
            ValueError: If the number of vectors exceeds MAX_INSERT_SIZE.
        """
        if not index_name:
            raise ValueError("index_name must be provided")

        # Check if the number of vectors exceeds the maximum allowed
        if len(vectors) > MAX_INSERT_SIZE:
            raise ValueError(
                f"Number of vectors ({len(vectors)}) exceeds maximum allowed ({MAX_INSERT_SIZE})"
            )

        # Convert VectorizeRecord objects to dictionaries
        vector_dicts = [vector.to_dict() for vector in vectors]

        # Choose endpoint based on insert_only parameter
        endpoint = "insert" if insert_only else "upsert"

        # Convert vectors to newline-delimited JSON
        ndjson_data = "\n".join(json.dumps(vector_dict) for vector_dict in vector_dicts)

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

        # Return the IDs of the added vectors
        return [vector.id for vector in vectors]

    # MARK: - add_documents
    def add_documents(
            self,
            documents: List[Document],
            ids: Optional[List[str]] = None,
            namespaces: Optional[List[str]] = None,
            insert_only: bool = False,
            index_name: str = None,
            **kwargs: Any,
    ) -> List[str]:
        """Add documents to the vectorstore.
        
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
        doc_ids = ids
        if any(hasattr(doc, "id") and doc.id is not None for doc in documents):
            doc_ids = []
            for i, doc in enumerate(documents):
                if hasattr(doc, "id") and doc.id is not None:
                    doc_ids.append(doc.id)
                elif ids is not None and i < len(ids):
                    doc_ids.append(ids[i])
                else:
                    doc_ids.append(str(uuid.uuid4()))

        return self.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=doc_ids,
            namespaces=namespaces,
            insert_only=insert_only,
            index_name=index_name
        )

    # MARK: - aadd_documents
    async def aadd_documents(
            self,
            documents: List[Document],
            ids: Optional[List[str]] = None,
            namespaces: Optional[List[str]] = None,
            insert_only: bool = False,
            index_name: str = None,
            **kwargs: Any,
    ) -> List[str]:
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
        doc_ids = ids
        if any(hasattr(doc, "id") and doc.id is not None for doc in documents):
            doc_ids = []
            for i, doc in enumerate(documents):
                if hasattr(doc, "id") and doc.id is not None:
                    doc_ids.append(doc.id)
                elif ids is not None and i < len(ids):
                    doc_ids.append(ids[i])
                else:
                    doc_ids.append(str(uuid.uuid4()))

        return await self.aadd_texts(
            texts=texts,
            metadatas=metadatas,
            ids=doc_ids,
            namespaces=namespaces,
            insert_only=insert_only,
            index_name=index_name
        )

    # MARK: - from_documents
    @classmethod
    def from_documents(
            cls: Type[VST],
            documents: List[Document],
            embedding: Embeddings,
            ids: Optional[List[str]] = None,
            namespaces: Optional[List[str]] = None,
            insert_only: bool = False,
            account_id: Optional[str] = None,
            index_name: Optional[str] = None,
            create_if_missing: bool = False,
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
            create_if_missing: If True, creates the index if it doesn't exist (default: False).
            dimensions: Number of dimensions for vectors when creating a new index.
            metric: Distance metric to use when creating a new index.
            
        Returns:
            CloudflareVectorize vectorstore.
            
        Raises:
            ValueError: If the number of documents exceeds MAX_INSERT_SIZE.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return cls.from_texts(
            texts=texts,
            embedding=embedding,
            metadatas=metadatas,
            ids=ids,
            namespaces=namespaces,
            insert_only=insert_only,
            account_id=account_id,
            index_name=index_name,
            create_if_missing=create_if_missing,
            dimensions=dimensions,
            metric=metric,
            api_token=cls.vectorize_api_token or api_token,
            **kwargs,
        )

    # MARK: - afrom_documents
    @classmethod
    async def afrom_documents(
            cls: Type[VST],
            documents: List[Document],
            embedding: Embeddings,
            ids: Optional[List[str]] = None,
            namespaces: Optional[List[str]] = None,
            insert_only: bool = False,
            account_id: Optional[str] = None,
            index_name: Optional[str] = None,
            create_if_missing: bool = False,
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
            create_if_missing: If True, creates the index if it doesn't exist (default: False).
            dimensions: Number of dimensions for vectors when creating a new index.
            metric: Distance metric to use when creating a new index.
            
        Returns:
            CloudflareVectorize vectorstore.
            
        Raises:
            ValueError: If the number of documents exceeds MAX_INSERT_SIZE.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return await cls.afrom_texts(
            texts=texts,
            embedding=embedding,
            metadatas=metadatas,
            ids=ids,
            namespaces=namespaces,
            insert_only=insert_only,
            account_id=account_id,
            index_name=index_name,
            create_if_missing=create_if_missing,
            dimensions=dimensions,
            metric=metric,
            api_token=cls.vectorize_api_token or api_token,
            **kwargs,
        )

    # https://developers.cloudflare.com/api/resources/vectorize/
