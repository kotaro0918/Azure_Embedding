import openai
import os

# 環境変数 OPENAI_API_KEY 読み込み
api_key = os.getenv("OPENAI_API_KEY")

# OPENAI 接続情報を設定
openai.api_type = "azure"
openai.api_key = api_key
openai.api_base = "https://azure-openai-935953.openai.azure.com/"
openai.api_version = "2023-06-01-preview"

def generate_text(message):
    completion = openai.ChatCompletion.create(
        # OpenAIの場合
        # model="gpt-4"
        # Azure OpenAI gpt-4の場合
        deployment_id = "gpt-4",
        # Azure OpenAI gpt-3.5-turboの場合
        # deployment_id = "gpt-3.5-turbo",
        messages = [
            {
                "role": "user",
                "content": message
            },
        ],
        max_tokens=1000,
        n=1,
        temperature=0.8,
    )

    response = completion.choices[0].message.content
    return response

if __name__ == "__main__":
    message = "What is the 2nd highest mountain in Japan?"

    response = generate_text(message)
    print(response)

    
