import openai
import os
import re
import requests
import sys
from num2words import num2words
import os
import pandas as pd
import numpy as np
from openai.embeddings_utils import get_embedding, cosine_similarity
import tiktoken

target_input=input()

# 環境変数 OPENAI_API_KEY 読み込み
api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("AZURE_ENDPOINT")


# OPENAI 接続情報を設定
openai.api_type = "azure"
openai.api_key = api_key
openai.api_base = api_base
openai.api_version = "2022-12-01"

url = openai.api_base + "/openai/deployments?api-version=2022-12-01" 

r = requests.get(url, headers={"api-key": api_key})

print(r.text)

df=pd.read_csv(os.path.join(os.getcwd(),'doc_class.csv'))
df_bills=df[['@id','rdfs:label']]
df_bills.columns=["id","label"]
print(df_bills)

tokenizer = tiktoken.get_encoding("cl100k_base")
df_bills['n_tokens'] = df_bills["label"].apply(lambda x: len(tokenizer.encode(x)))
df_bills = df_bills[df_bills.n_tokens<8192]
len(df_bills)
print(df_bills)

df_bills['ada_v2'] = df_bills["label"].apply(lambda x : get_embedding(x, engine = 'text-embedding-ada-002-v2')) # engine should be set to the deployment name you chose when you deployed the text-embedding-ada-002 (Version 2) model

# search through the reviews for a specific product
def search_docs(df, user_query, top_n=3, to_print=True):
    embedding = get_embedding(
        user_query,
        engine="text-embedding-ada-002-v2" # engine should be set to the deployment name you chose when you deployed the text-embedding-ada-002 (Version 2) model
    )
    df["similarities"] = df.ada_v2.apply(lambda x: cosine_similarity(x, embedding))

    res = (
        df.sort_values("similarities", ascending=False)
        .head(top_n)
    )
    if to_print:
        print(res)
    return res


res = search_docs(df_bills, f"""以下の text はある本の概要説明文です。この説明文の内容に適切な分類項目をidに含まれる数字で提示してください。また、分類の根拠も併せて出力してください。. text:{target_input}""", top_n=4)