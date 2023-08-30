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


## Azure OpenAI とOpenAI のEmbedding 比較


今回は前回本の分類項目で使用したものと同じテキストデータを用いた　　

##　 動作原理
1. テキストデータを500トークン数ごとに分けて chunk_overlapを50トークンに設定して分割する
1. 分割したデータを16個ごとにembeddingしてFAISSに格納
1. queryをembeddingして、コサイン類似度の高いものをプロンプト内に組み込む

##　比較

・Azure OpenAIとOpenAIとの間に大きな違いは見られなかった  

・Embeddingしているエンジンはtext-embedding-ada-002で同じ

・二つの出力結果について同プロンプトで試してみたところ、ほぼ同じ出力が帰ってきた  
またプロンプトの条件を変化させて比較した場合でも、同じ傾向が見られた  
(根拠や答えに至る過程を合わせて出力させると精度が上がるなど)



### 苦労した点

 AzureOpenAIはEmbeddingする際に最大16個配列しか一度に入力できないので、16個ごとに分割して処理する点

 今回はコード内のdocsから16個ごと取り出して、その都度dbにdocumentとして追加する方法をとった  
 
 Faiss.mergeなども試してみたが、出力がNoneになりうまくいかなかった。

 



