# Tsumuji

Qwen3.8-Flash-Next専用のRust＋CUDA推論エンジンです。
Hyper-Connection、PLE、Gated DeltaNet、QSA、NVFP4、MTPの固有実装を保持しています。
CUDA Driver API、重み読取、トークナイザー、JSON、HTTPフロントエンドは`common/`を使います。

リポジトリのルートから実行してください。

```sh
nix develop
./darask build tsumuji
./darask serve tsumuji
```

OpenAI形式の`/v1/chat/completions`と`/v1/completions`、SSE、ツール呼び出し、思考の分離、会話先頭の再利用に対応します。
`--`以降でネイティブオプションを渡せます。

```sh
./darask serve tsumuji -- --model /path/to/checkpoint --port 8090 --ctx 131072
```

モデルの構造・各段階の記録は`docs/`、APIの詳細は[docs/m5.md](docs/m5.md)、統合前の使い方は[docs/legacy-readme.md](docs/legacy-readme.md)です。
過去のパス・出力ディレクトリは当時の記録です。統合後のビルドはルートのCargoワークスペースから行います。
