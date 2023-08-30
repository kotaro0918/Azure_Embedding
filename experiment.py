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

target_input=input()

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
mylist = docs[0:15]
db =FAISS.from_documents(mylist,embeddings)
print(db)
for i in range(12):
    mylist = docs[16*(i+1):16*(i+2)-1]
    db.add_documents(mylist)
    if i == 11:
        mylist = docs[16*(i+1):len(docs)]
        db.add_documents(mylist)
retriever = db.as_retriever()
query = f"""あなたは図書館で書籍の分類を担当する司書です。

以下の text はある本の概要説明文です。この説明文の内容に適切な分類項目をidに含まれる数字で３つ提示してください。
答えに至る過程と根拠も出力してください
text: {target_input}"""
embedding_vector = get_embedding(query,engine ="text-embedding-ada-002-v2" )
docs_and_scores = db.similarity_search_by_vector(embedding_vector)
llm = AzureChatOpenAI(
    deployment_name="gpt-35-turbo",
    temperature=0,
    openai_api_base=openai.api_base,
    openai_api_version= openai.api_version
)
from langchain.callbacks import get_openai_callback
# load_qa_chainを準備

with get_openai_callback() as cb:
    chain = RetrievalQA.from_chain_type(
        llm=llm,chain_type="stuff", 
        retriever=retriever,
        verbose=True,
         chain_type_kwargs={"verbose": True})

# 質問応答の実行
    result=chain.run(query)
    print(result)
    print(cb)
