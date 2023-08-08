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
