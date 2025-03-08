import itertools
import asyncio
import warnings
from datetime import datetime
import pandas as pd
import os
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

from langchain_community.embeddings.cloudflare_workersai import CloudflareWorkersAIEmbeddings
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from libs.community.langchain_community.vectorstores.cloudflare_vectorize import CloudflareVectorize, VectorizeRecord

# MARK: - PARAMS
MODEL_WORKERSAI = "@cf/baai/bge-large-en-v1.5"
vectorize_index_name = "test-langchain2"
d1_database_id = "8ce9ce08-8961-475c-98fb-1ef0e6e4ca40"

# load_dotenv("/Users/collierking/Documents/langchain/libs/community/tests/integration_tests/vectorstores/.env")
load_dotenv("/Users/collierking/Desktop/chartclass/langchain/libs/community/tests/integration_tests/vectorstores/.env")

cf_acct_id = os.getenv("cf_acct_id")
cf_ai_token = os.getenv("cf_ai_token")
cf_vectorize_token = os.getenv("cf_vectorize_token")
cf_d1_token = os.getenv("d1_api_token")

# MARK: - PULL DATA
docs = WikipediaLoader(query="Cloudflare", load_max_docs=2).load()

# MARK: - CHUNK
# recursive character splitter
text_splitter = \
    RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=100,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
texts = text_splitter.create_documents([docs[0].page_content])

running_section = ""
for idx, text in enumerate(texts):
    if text.page_content.startswith("="):
        running_section = text.page_content
        running_section = running_section.replace("=", "").strip()
    else:
        if running_section == "":
            text.metadata = {"section": "Introduction"}
        else:
            text.metadata = {"section": running_section}

# MARK: - UTILS
embedder = \
    CloudflareWorkersAIEmbeddings(
        account_id=cf_acct_id,
        api_token=cf_ai_token,
        model_name=MODEL_WORKERSAI
    )

cfVect = \
    CloudflareVectorize(
        embedding=embedder,
        account_id=cf_acct_id,
        api_token=cf_vectorize_token,
        d1_database_id=d1_database_id,
        ai_api_token=cf_ai_token,
        d1_api_token=cf_d1_token,
        vectorize_api_token=cf_vectorize_token,
    )

# MARK: - LIST INDEXES
print(f"Step: list_indexes -- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
arr_indexes = cfVect.list_indexes()
arr_indexes = [x for x in arr_indexes if "test-langchain" in x.get("name")]
print(len(arr_indexes))

# break into chunks
arr_indexes_chunks = [arr_indexes[i:i + 10] for i in range(0, len(arr_indexes), 10)]

# MARK: - Async Deletes
print(f"Step: adelete_index -- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
for idx, chunk in enumerate(arr_indexes_chunks):
    print("Deleting Indexes {}".format(idx))
    # if idx == 1:
    #     break
    arr_async_requests = [
        cfVect.adelete_index(
            index_name=x.get("name")
        )
        for x in chunk
    ]

    r = asyncio.get_event_loop().run_until_complete(asyncio.gather(*arr_async_requests))

# MARK: - CREATE INDEX
print(f"Step: delete_index -- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
try:
    cfVect.delete_index(
        index_name=vectorize_index_name
    )
except Exception as e:
    print(e)

print(f"Step: create_index -- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
r = \
    cfVect.create_index(
        index_name=vectorize_index_name,
    )

print(r)

# MARK: - CREATE MD INDEXES
print(f"Step: create_metadata_index -- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
cfVect.create_metadata_index(
    property_name="section",
    index_type="string",
    index_name=vectorize_index_name,
)

print(cfVect.list_metadata_indexes(index_name=vectorize_index_name))

# MARK: - ADD DOCUMENTS
print(f"Step: add_documents -- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
# split into chunks of 1000
r = cfVect.add_documents(
    index_name=vectorize_index_name,
    documents=texts
)

# MARK: - QUERY/SEARCH
print(f"Step: similarity_search -- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
query_documents = \
    cfVect.similarity_search(
        index_name=vectorize_index_name,
        query="california",
        k=100,
        filter=None,
        return_metadata="none",
    )

arr_records = [dict(x) for x in query_documents]
df_records = pd.DataFrame(arr_records)
print(df_records)

query_documents, query_scores = \
    cfVect.similarity_search_with_score(
        index_name=vectorize_index_name,
        query="california",
        k=100,
        filter=None,
        return_metadata="all",
    )

arr_records = [dict(x) for x in query_documents]
df_records = pd.DataFrame(arr_records)
print(df_records)

# MARK: - QUERY SEARCH WITH METADATA
print(f"Step: similarity_search (metadata filter) -- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
query_documents = \
    cfVect.similarity_search(
        index_name=vectorize_index_name,
        query="california",
        k=100,
        filter={"section": "Introduction"},
        return_metadata="all",
        return_values=True
    )

arr_records = [dict(x) for x in query_documents]
df_records = pd.DataFrame(arr_records)
print(df_records)

arr_sample_ids = df_records['id'].tolist()[:3]
print(len(arr_sample_ids))

# MARK: - GET BY IDS
print(f"Step: get_by_ids -- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
query_documents = \
    cfVect.get_by_ids(
        index_name=vectorize_index_name,
        ids=arr_sample_ids
    )

arr_records = [dict(x) for x in query_documents]
df_records = pd.DataFrame(arr_records)
print(df_records)

# MARK: - DELETE BY IDS
print(f"Step: delete -- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
r = \
    cfVect.delete(
        index_name=vectorize_index_name,
        ids=arr_sample_ids
    )

print(r.json())

# MARK: - GET BY IDS again
print(f"Step: get_by_ids -- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
query_documents = \
    cfVect.get_by_ids(
        index_name=vectorize_index_name,
        ids=arr_sample_ids
    )

arr_records = [dict(x) for x in query_documents]
df_records = pd.DataFrame(arr_records)
print(df_records)

assert len(df_records) == 0

# MARK: - GET INDEX
print(f"Step: get_index -- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
r = \
    cfVect.get_index(
        index_name=vectorize_index_name,
    )

print(r)

# MARK: - GET INDEX INFO
print(f"Step: get_index_info -- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
r = \
    cfVect.get_index_info(
        index_name=vectorize_index_name,
    )

print(r)

# MARK: - DELETE INDEX
print(f"Step: delete_index -- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
r = \
    cfVect.delete_index(
        index_name=vectorize_index_name,
    )

print(r)

# MARK: - FROM DOCUMENTS
print(f"Step: from_documents -- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
cfVect = \
    CloudflareVectorize.from_documents(
        account_id=cf_acct_id,
        index_name=vectorize_index_name,
        documents=texts,
        embedding=embedder,
        # api_token=cf_vectorize_token,
        # d1_database_id=d1_database_id,
        ai_api_token=cf_ai_token,
        # d1_api_token=cf_d1_token,
        vectorize_api_token=cf_vectorize_token
    )

# MARK: - QUERY/SEARCH
print(f"Step: similarity_search -- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
query_documents = \
    cfVect.similarity_search(
        index_name=vectorize_index_name,
        query="california",
        k=100,
        filter=None,
        return_metadata="none",
    )

arr_records = [dict(x) for x in query_documents]
df_records = pd.DataFrame(arr_records)
print(df_records)

# MARK: - DELETE INDEX
print(f"Step: delete_index -- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
r = \
    cfVect.delete_index(
        index_name=vectorize_index_name,
    )

# MARK: - FROM TEXTS
print(f"Step: from_texts -- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
cfVect = \
    CloudflareVectorize.from_texts(
        account_id=cf_acct_id,
        index_name=vectorize_index_name,
        texts=[x.page_content for x in texts],
        metadatas=[x.metadata for x in texts],
        embedding=embedder,
        api_token=cf_vectorize_token,
        # d1_database_id=d1_database_id,
        ai_api_token=cf_ai_token,
        # d1_api_token=cf_d1_token,
        vectorize_api_token=cf_vectorize_token
    )

# MARK: - QUERY/SEARCH
print(f"Step: similarity_search -- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
query_documents = \
    cfVect.similarity_search(
        index_name=vectorize_index_name,
        query="california",
        k=100,
        filter=None,
        return_metadata="none",
    )

arr_records = [dict(x) for x in query_documents]
df_records = pd.DataFrame(arr_records)
print(df_records)





# MARK: - ASYNC TESTS
# TODO: REPEAT TEST SCENARIOS FOR ASYNC
arr_indexes = cfVect.list_indexes()
arr_indexes = [x for x in arr_indexes if "test-langchain" in x.get("name")]
print(len(arr_indexes))

# break into chunks
arr_indexes_chunks = [arr_indexes[i:i + 10] for i in range(0, len(arr_indexes), 10)]

for idx, chunk in enumerate(arr_indexes_chunks):
    print(f"Deleting chunk {idx} of {len(chunk)} indexes")
    # if idx == 1:
    #     break
    arr_async_requests = [
        cfVect.adelete_index(
            index_name=x.get("name")
        )
        for x in chunk
    ]

    r = asyncio.get_event_loop().run_until_complete(asyncio.gather(*arr_async_requests))

# MARK: - ASYNC CREATE INDEXES
arr_index_names = []
arr_async_requests = []
for i in range(2):
    print(i)
    index_name = f"{vectorize_index_name}{i}"
    arr_index_names.append(index_name)
    arr_async_requests.append(
        cfVect.acreate_index(
            index_name=index_name,
        )
    )

r = asyncio.get_event_loop().run_until_complete(asyncio.gather(*arr_async_requests))

# MARK: - ASYNC GET INDEXES
arr_async_requests = []
for i in range(2):
    arr_async_requests.append(cfVect.alist_indexes())

r = asyncio.get_event_loop().run_until_complete(asyncio.gather(*arr_async_requests))

arr_indexes = list(itertools.chain(*r))
arr_index_names = list(set([x.get('name') for x in arr_indexes if "test-langchain" in x.get("name")]))
len(arr_index_names)

# MARK: - ASYNC CREATE MD INDEXES
arr_async_requests = []
for idx, index in enumerate(arr_index_names):
    arr_async_requests.append(
        cfVect.acreate_metadata_index(
            property_name="subject",
            index_type="string",
            index_name=index,
        )
    )

r = asyncio.get_event_loop().run_until_complete(asyncio.gather(*arr_async_requests))

# MARK: - ASYNC ADD DOCUMENTS
