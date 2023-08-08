import openai
import os

# localで実行中のllama2
openai.api_base = "http://host.docker.internal:8000/v1"
# openai.api_version = "2023-06-01-preview"

def generate_text(message):
    completion = openai.ChatCompletion.create(
        # Llma2の場合　（文字列は適当でも動くっぽい）
        model="/models/llama-2-7b-chat.ggmlv3.q5_K_M.bin",
        # Azure OpenAI gpt-4の場合
        # deployment_id = "gpt-4",
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
    message = "What is the highest mountain in Japan?"

    response = generate_text(message)
    print(response)

    
