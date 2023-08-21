# Azure OpenAI Client (python)

python から Azure OpenAI Service あるいは　ローカル PC 上で実行中の Llama2 サービスに接続するサンプルコード

### 事前準備

- vscode に拡張 "Remote Development" をインストールしておく
- Docker Desktop をインストールして起動しておく

### devcontainer.env 　（Azure OpenAI 　 Service に接続する場合）

↓ のような中身の `.devcontainer/devcontainer.env` ファイルを作成して同じディレクトリに置いておく

```
OPENAI_API_KEY="44d44b78f8254775abd91619f164xxxx"
```

### 使い方

1. devcontainer で開く
1. main.py 編集
1. 実行
   OpenAI に接続する

```
python openai-test.py
```

Llama2 に接続する

```
python llama2-test.py
```

### その他

PC やブラウザなどから llama-cpp-sever の URL は `http://localhost:8000/` だが、
devcontainer 内からだと `http://host.docker.internal:8000/`　になる

## Azure Openaiを用いたEmbedding

今回用いたデータセット
```
curl "https://raw.githubusercontent.com/Azure-Samples/Azure-OpenAI-Docs-Samples/main/Samples/Tutorials/Embeddings/data/bill_sum_data.csv" --output bill_sum_data.csv
```

入力トークン数上限が8192なので、上のデータからトークン数が8192以下のものだけを抽出しembedding  

Embeddingに使用できるモデルには限りがあるので注意
(gpt3.5turboやgpt4は使用不可)

### 　動作原理  

1.  bill_sum_data.csvのデータを加工したdf_billのtextをEmbedding

1.  user_queryをベクトル化する

1.  user-queryとのコサイン類似度が高いものを順番に４つ出力

### 苦労した点

1. Embeddingのengine名に指定するのはモデル名ではなくデプロイ名