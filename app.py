from fastapi import FastAPI
from pydantic import BaseModel
from data_converter import load_and_convert_data
from data_ingestion import ingest_documents
from retrieval_generation import build_chain

app = FastAPI()

class QueryRequest(BaseModel):
    session_id: str
    input: str

# Ingest once when FastAPI starts
@app.on_event("startup")
def initialize():
    global chain
    docs = load_and_convert_data("Data/flipkart_product_review.csv")
    vstore = ingest_documents(docs)
    chain = build_chain(vstore)

@app.post("/query")
def query_chain(request_data: QueryRequest):
    response = chain.invoke(
        {"input": request_data.input},
        config={"configurable": {"session_id": request_data.session_id}}
    )
    return {"answer": response["answer"]}
