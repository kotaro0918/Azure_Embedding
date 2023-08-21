import os
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage

# 環境変数 OPENAI_API_KEY 読み込み
api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("AZURE_ENDPOINT")


model = AzureChatOpenAI(
    openai_api_base = api_base,
    openai_api_version = "2023-06-01-preview",
    deployment_name = "gpt-4",
    openai_api_key = api_key,
    openai_api_type = "azure",
)

result = model(
    [
        HumanMessage(content="What comes after 1,2,3 ?"),
    ]
)

print(result)