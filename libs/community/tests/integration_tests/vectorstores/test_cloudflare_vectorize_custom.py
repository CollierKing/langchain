import warnings
warnings.filterwarnings('ignore')

from langchain_community.embeddings.cloudflare_workersai import CloudflareWorkersAIEmbeddings
from langchain_core.documents import Document
from libs.community.langchain_community.vectorstores.cloudflare_vectorize import CloudflareVectorize

import json
import pandas as pd
from fusetools.cloud_tools import AWS
from fusetools.db_conn_tools import Postgres
import os

# MARK: - PARAMS
secret_region_name = "us-east-1"
MODEL_WORKERSAI = "@cf/baai/bge-large-en-v1.5"
vectorize_index_name = "cc_kt_docs_test"
d1_database_id = "8ce9ce08-8961-475c-98fb-1ef0e6e4ca40"

# MARK: - SECRETS
str_secret_cc = \
    AWS.get_secret_manager(
        secret_name="cc_ccdo_creds",
        region_name=secret_region_name,
        pub=os.environ['cc_aws_sm_key'],
        sec=os.environ['cc_aws_sm_secret']
    )

json_secret_rds_creds_cc = json.loads(str_secret_cc)

# WorkersAI
str_secret = \
    AWS.get_secret_manager(
        secret_name="cc_cf_creds",
        region_name=secret_region_name,
        pub=os.environ['cc_aws_sm_key'],
        sec=os.environ['cc_aws_sm_secret']
    )

json_secret_cf = json.loads(str_secret)
cf_acct_id = json_secret_cf.get("CC_CF_ACCT_NUM")
cf_ai_token = json_secret_cf.get("CC_CF_AI_API_KEY")
cf_vectorize_token = json_secret_cf.get("CC_CF_VECTORIZE_KEY")
d1_api_token = json_secret_cf.get("CC_D1_API_TOKEN")

# MARK: - CLIENTS
cursor_p_cc, conn_cc = \
    Postgres.con_postgres(
        host=json_secret_rds_creds_cc['do_host_cc'],
        db=json_secret_rds_creds_cc['do_db_cc'],
        usr=json_secret_rds_creds_cc['do_user_cc'],
        pwd=json_secret_rds_creds_cc['do_pwd_cc'],
        port=json_secret_rds_creds_cc['do_port_cc']
    )

# MARK: - PULL DATA
# query PG for saved and not yet embedded
tgt_tbl_name = "cc_kt_docs"
sql = f'''
SELECT * FROM {tgt_tbl_name}
LIMIT 1000
'''
df_kt_chunks = pd.read_sql_query(sql=sql, con=conn_cc)

# MARK: - UTILS
embedder = CloudflareWorkersAIEmbeddings(
    account_id=cf_acct_id,
    api_token=cf_ai_token,
    model_name=MODEL_WORKERSAI
)

cfVect = CloudflareVectorize(
    embedding=embedder,
    account_id=cf_acct_id,
    api_token=cf_vectorize_token,
    d1_database_id=d1_database_id,
    ai_api_token=cf_ai_token,
    d1_api_token=d1_api_token,
    vectorize_api_token=cf_vectorize_token,
)

# MARK: - PREP DATA
# explode data
df_kt_chunks_exp = df_kt_chunks.explode("arr_knowledge_triples").reset_index(drop=True)

df_kt_chunks_exp['kt_str'] = \
    df_kt_chunks_exp.apply(
        lambda
            x: f"{x['arr_knowledge_triples'].get('subject')} {x['arr_knowledge_triples'].get('predicate')} {x['arr_knowledge_triples'].get('object')}",
        axis=1
    )

df_kt_chunks_exp['kt_id'] = \
    df_kt_chunks_exp.apply(
        lambda x: f"{x['arr_knowledge_triples'].get('kt_id')}",
        axis=1
    )

df_kt_chunks_exp['chunk_id'] = \
    df_kt_chunks_exp.apply(
        lambda x: f"{x['arr_knowledge_triples'].get('chunk_id')}",
        axis=1
    )

df_kt_chunks_exp['metadata'] = \
    df_kt_chunks_exp.apply(
        lambda x: {
            "ticker": x['ticker'],
            "subject": x['arr_knowledge_triples'].get('subject'),
            "predicate": x['arr_knowledge_triples'].get('predicate'),
            "object": x['arr_knowledge_triples'].get('object'),
            "doc_id": x['doc_id'],
            "doc_type": x['doc_type'],
            "chunk_id": x['chunk_id'],
        },
        axis=1
    )

# MARK: - CREATE INDEX
try:
    cfVect.delete_index(
        index_name=vectorize_index_name
    )
except Exception as e:
    print(e)

r = \
    cfVect.create_index(
        index_name=vectorize_index_name,
    )

# MARK: - CREATE MD INDEXES
cfVect.create_metadata_index(
    property_name="subject",
    index_type="string",
    index_name=vectorize_index_name,
)

cfVect.create_metadata_index(
    property_name="object",
    index_type="string",
    index_name=vectorize_index_name,
)

cfVect.create_metadata_index(
    property_name="predicate",
    index_type="string",
    index_name=vectorize_index_name,
)

cfVect.create_metadata_index(
    property_name="ticker",
    index_type="string",
    index_name=vectorize_index_name,
)

print(cfVect.list_metadata_indexes(index_name=vectorize_index_name))

# MARK: - ADD DOCUMENTS
# split into chunks of 1000
df_kt_chunks_exp_chunks = [df_kt_chunks_exp[i:i + 1000] for i in range(0, len(df_kt_chunks_exp), 1000)]

for idx, chunk in enumerate(df_kt_chunks_exp_chunks):
    if idx == 1:
        break
    print(f"Adding chunk {chunk.index} of {len(df_kt_chunks_exp_chunks)}")
    r = cfVect.add_documents(
        index_name=vectorize_index_name,
        documents=[
            Document(
                page_content=text,
                metadata=metadata
            )
            for text, metadata in zip(
                chunk['kt_str'].tolist(),
                chunk['metadata'].tolist()
            )
        ],
        ids=chunk['kt_id'].tolist(),
        namespaces=[x.get("doc_type") for x in chunk['metadata'].tolist()],
    )

# MARK: - QUERY/SEARCH
query_documents = \
    cfVect.similarity_search(
        index_name=vectorize_index_name,
        query="airplanes",
        k=100,
        filter=None,
        return_metadata="none",
    )

arr_records = [dict(x) for x in query_documents]
df_records = pd.DataFrame(arr_records)

query_documents, query_scores = \
    cfVect.similarity_search_with_score(
        index_name=vectorize_index_name,
        query="airplanes",
        k=100,
        filter=None,
        return_metadata="all",
    )

arr_records = [dict(x) for x in query_documents]
df_records = pd.DataFrame(arr_records)

df_records['id'] = \
    df_records.apply(
        lambda x: x['metadata'].get('id'),
        axis=1
    )

# MARK: - QUERY SEARCH WITH METADATA
query_documents = \
    cfVect.similarity_search(
        index_name=vectorize_index_name,
        query="",
        k=100,
        filter={
            "ticker": "HII"
        },
        return_metadata="all",
    )

arr_records = [dict(x) for x in query_documents]
df_records = pd.DataFrame(arr_records)

arr_sample_ids = df_records['id'].tolist()[:3]
print(len(arr_sample_ids))

# MARK: - GET BY IDS
r = \
    cfVect.get_by_ids(
        index_name=vectorize_index_name,
        ids=arr_sample_ids
    )

arr_records = [x.to_dict() for x in r]
df_records = pd.DataFrame(arr_records)

# MARK: - DELETE BY IDS
r = \
    cfVect.delete(
        index_name=vectorize_index_name,
        ids=arr_sample_ids
    )

print(r)

# MARK: - GET BY IDS again
r = \
    cfVect.get_by_ids(
        index_name=vectorize_index_name,
        ids=arr_sample_ids
    )

arr_records = [x.to_dict() for x in r]
df_records = pd.DataFrame(arr_records)

assert len(df_records) == 0

# MARK: - GET INDEX
r = \
    cfVect.get_index(
        index_name=vectorize_index_name,
    )

print(r)

# MARK: - GET INDEX INFO
r = \
    cfVect.get_index_info(
        index_name=vectorize_index_name,
    )

print(r)

# MARK: - DELETE INDEX
r = \
    cfVect.delete_index(
        index_name=vectorize_index_name,
    )

print(r)

# todo FROM DOCUMENTS

# todo ADD EMBEDDINGS

# TODO: REPEAT TEST SCENARIOS FOR ASYNC
