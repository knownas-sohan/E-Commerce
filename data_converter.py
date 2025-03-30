import pandas as pd
from langchain_core.documents import Document

def load_and_convert_data(csv_path):
    data = pd.read_csv(csv_path)
    data = data[["product_title", "review"]]
    
    product_list = []
    for _, row in data.iterrows():
        product_list.append({
            "product_name": row["product_title"],
            "review": row["review"]
        })
    
    docs = [Document(page_content=obj["review"], metadata={"product_name": obj["product_name"]}) for obj in product_list]
    return docs
