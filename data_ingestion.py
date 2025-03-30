import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_astradb import AstraDBVectorStore

load_dotenv()

def ingest_documents(docs):
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=os.getenv("HF_TOKEN"),
        model_name="BAAI/bge-base-en-v1.5"
    )

    vstore = AstraDBVectorStore(
        embedding=embeddings,
        collection_name="flipkart01",
        api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
        token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
        namespace=os.getenv("ASTRA_DB_KEYSPACE")
    )

    vstore.add_documents(docs)
    return vstore
