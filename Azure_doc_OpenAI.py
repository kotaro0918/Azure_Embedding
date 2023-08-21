import openai
import os
import re
import requests
from langchain.vectorstores import FAISS
import sys
from num2words import num2words
import os
import pandas as pd
import numpy as np
from langchain.chat_models import AzureChatOpenAI
from openai.embeddings_utils import get_embedding, cosine_similarity
import tiktoken
from langchain.chains.question_answering import load_qa_chain
# Embedding用
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
# Vector 格納 / FAISS
from langchain.vectorstores import FAISS
# テキストファイルを読み込む
from langchain.document_loaders import TextLoader
# Q&A用Chain
from langchain.chains.question_answering import load_qa_ch

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
from langchain.document_loaders import CSVLoader
# CSVLoaderを使用してCSVファイルからデータを読み込む
loader = CSVLoader("doc_class.csv")
documents = loader.load()

elements=[]
# 各ドキュメントのコンテンツとメタデータにアクセスする
for document in documents:
    content = document.page_content
    elements.append(content)
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(deployment="text-embedding-ada-002-v2")
db =FAISS.from_texts

query = f"""あなたは図書館で書籍の分類を担当する司書です。

以下の text はある本の概要説明文です。地名に注目してこの説明文の内容に適切な分類項目をidに含まれる数字で提示してください。
答えに至る過程も出力してください
text: {target_input}"""
embedding_vector = embeddings.embed_query(query)
docs_and_scores = db.similarity_search_by_vector(embedding_vector)
llm = AzureChatOpenAI(
    deployment_name="gpt-35-turbo",
    openai_api_base=openai.api_base,
    openai_api_version=openai.api_version ,
    openai_api_key=openai.api_key ,
    temperature=0,
    request_timeout=180,
)
from langchain.callbacks import get_openai_callback
# load_qa_chainを準備

with get_openai_callback() as cb:
    chain = load_qa_chain(llm, chain_type="stuff", verbose=True)

# 質問応答の実行
    print(chain({"input_documents": docs_and_scores, "question": query},return_only_outputs=True))
    print(cb)
