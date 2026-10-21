# Darask Inference Engine

RTX PRO 6000 Blackwell向けの、Rust＋CUDAによるローカル推論エンジンです。
共通基盤を一つにし、モデルごとの実行経路と最適化は分けています。

| エンジン | モデル | 固有の実装 |
|---|---|---|
| Hayate | GLM-5.3-Flash | MLA、エキスパートのGPU/CPU分担、EXL3/LOWBIT/NVFP4/FP8 |
| Kogarashi | Ornith-1.5-122B | Qwen3.5 MoE、GGUF Q4_K/Q6_Kの直接実行、Gated DeltaNet |
| Tsumuji | Qwen3.8-Flash-Next | Hyper-Connection、PLE、QSA、NVFP4、MTP |

```text
common/          Rust CLI・JSON・トークナイザー・重み形式・HTTP API・CUDA Driver API
engines/hayate/   GLMの実行器と専用カーネル
engines/kogarashi/Ornithの実行器と専用カーネル、比較用バックエンド
engines/tsumuji/  Qwenの実行器と専用カーネル
local/           重み・Studio・データ・元Git履歴（公開対象外）
```

## ビルドと起動

NixOS / Linux x86_64、NVIDIA driver、CUDA sm_120aを対象にしています。
モデル重みは含みません。Pythonは変換・評価・量子化実験の補助ツール用です。
推論のCLI・サーバ・ネイティブデコーダはRust＋CUDAです。

```sh
nix develop
./darask list
./darask build hayate
./darask build kogarashi
./darask build tsumuji
./darask doctor
```

```sh
./darask serve hayate
./darask serve kogarashi
./darask serve tsumuji
```

`--`以降で各エンジンのオプションを指定できます。`--dry-run`はコマンドの表示だけを行います。

```sh
./darask serve kogarashi --dry-run -- --context 8192 --port 8080
./darask run kogarashi -- --prompt 'The capital of France is' --tokens 32
```

HayateのCUDAビルドには、初回に`engines/hayate/scripts/fetch-third-party.sh`で固定版の依存ソースを取得します。
TsumujiとKogarashiのネイティブデコーダはllama.cppにリンクしません。

## 画像入力・キャプション

3モデルとも画像からテキストを生成できます。サーバーを起動後、次のコマンドでフォルダ内の画像を2件並列で送信します。

```sh
./darask batch kogarashi --input /path/to/images --out local/captions.jsonl --jobs 2
```

`hayate`、`tsumuji`にも対応します。結果は逐次保存され、同じコマンドで未完了分から再開できます。
GPUは重みを一度ロードして順次推論します。重みの準備・API形式・詳しい使い方は[画像入力](docs/VISION.md)を参照してください。

## 品質と測定

```sh
./darask check
```

既定のチェックはGPUを使わず、共有処理・形式検証・API・既存のCPU回帰テストを実行します。
GPU測定は既存の共有GPU枠を使います。実モデルの回帰確認と性能の記録は[検証記録](docs/VALIDATION.md)に分けています。
異なる量子化や入力条件の速度を同一条件の比較として扱いません。

Kogarashiの元の固定版llama.cppとCUDAパッチは、比較基準として`engines/kogarashi/`に保存しています。
`./darask reference kogarashi`で起動できます。ネイティブ版の状態と制限は[検証記録](docs/VALIDATION.md)を参照してください。

配置と移行については[移行記録](docs/MIGRATION.md)、モデルごとの詳細は各`engines/*/README.md`を参照してください。

## ライセンス

本体は[Apache-2.0](LICENSE)。参考実装と依存ソースの出典は[NOTICE](NOTICE)と`licenses/`に記載しています。
モデルのライセンスはそれぞれの配布元に従います。
