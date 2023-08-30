import openai
import os
import re
import requests
import pandas as pd
from openai.embeddings_utils import get_embedding, cosine_similarity
from langchain.vectorstores import FAISS
from num2words import num2words
import os
from langchain.chat_models import AzureChatOpenAI
import tiktoken
from langchain.chains import RetrievalQA
# Embedding用
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
# Vector 格納 / FAISS
from langchain.vectorstores import FAISS
# テキストファイルを読み込む
from langchain.document_loaders import TextLoader
# Q&A用Chain

target_input="埼京線"

# 環境変数 OPENAI_API_KEY 読み込み
api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("AZURE_ENDPOINT")


# OPENAI 接続情報を設定
openai.api_type = "azure"
openai.api_key = api_key
openai.api_base = "https://azure-openai-935953.openai.azure.com/"
openai.api_version = "2023-06-01-preview"
from langchain.document_loaders import CSVLoader
# CSVLoaderを使用してCSVファイルからデータを読み込む
loader = TextLoader('doc_class.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)
from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model_kwargs={"deployment_id":"text-embedding-ada-002-v2"})
for i in range(3):
    mylist = docs[16*i:16*(i+1)-1]
    db =FAISS.from_documents(mylist,embeddings)
query = f"""あなたは図書館で書籍の分類を担当する司書です。

以下の text はある本の概要説明文です。地名に注目してこの説明文の内容に適切な分類項目をidに含まれる数字で提示してください。
答えに至る過程も出力してください
text: {target_input}"""
embedding_vector = get_embedding(query,engine ="text-embedding-ada-002-v2" )
docs_and_scores = db.similarity_search_by_vector(embedding_vector)
chat = AzureChatOpenAI(
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
    qa = RetrievalQA.from_chain_type(
    llm=chat, 
    chain_type="stuff", 
    retriever=retriever
)

# 質問応答の実行
    print(qa.run({"input_documents": docs_and_scores, "question": query},return_only_outputs=True))
    print(cb)
