# Tsumuji

Qwen3.8-Flash-Next（`qwen4_exp`）専用の Rust + CUDA 推論エンジンです。RTX PRO 6000 Blackwell 1枚と RAM 128GB の構成で、SGLang より速く、かつ品質を落とさず（decode は W4A16）に動かすことを目標にしています。

- 計画：[docs/plan.md](docs/plan.md)
- M0 の結果と計画の修正点：[docs/m0.md](docs/m0.md)

## 現状

M0（計測と下調べ）まで完了しています。推論はまだ動きません。

| crate | 内容 |
|---|---|
| `tsumuji-cuda` | CUDA Driver API の薄いラッパー（外部 crate なし）。PDL 付き起動、stream capture の graph、mapped pinned メモリ |
| `tsumuji-kernels` | `kernels/<dir>/*.cu` を sm_120a の cubin にして埋め込む |
| `tsumuji-model` | 形状の定数（`kernels/include/q38.h` と一致をテスト）、PLE の n-gram ハッシュ |
| `tsumuji-pack` | safetensors の読み取り、チェックポイントの検査 |
| `tsumuji-bench` | M0 のマイクロベンチ |
| `tsumuji-server` | OpenAI 互換 API サーバ `tsumuji-serve`（トークナイザとチャットテンプレートも Rust、外部 crate なし）。[docs/m5.md](docs/m5.md) |

## 使い方

ツールチェーンは flake にあります（Hayate と同じ nixpkgs を固定しています）。

```bash
nix develop -c cargo build --release
```

```bash
nix develop -c cargo test --release
```

GPU を使うものは、Studio の serve を止めないように Hayate の GPU スロット経由で実行します（Python は Hayate の dev shell のものを使います）。

```bash
nix develop /mnt/NVME4TB/darask-inference-engine/engines/hayate#default -c bash -c 'NEED=14 scripts/gpu-run.sh -- target/release/tsumuji-bench all'
```

チェックポイントの検査：

```bash
nix develop -c target/release/tsumuji-pack inspect /mnt/NVME4TB/darask-inference-engine/local/hayate/src/Qwen3.8-Flash-Next-NVFP4
```

golden fixture の生成（CPU のみ、Hayate の tools シェルの transformers 5.17 を使います）：

```bash
nix develop /mnt/NVME4TB/darask-inference-engine/engines/hayate#tools -c python3 python/golden_tiny.py
```

### API サーバ（OpenAI 互換）

1ユーザー向けの `POST /v1/chat/completions`（SSE ストリーミング、ツール呼び出し、会話の先頭の再利用）、`POST /v1/completions`、`GET /v1/models`、`GET /health` です。配線は Hayate serve の約束に合わせてあり、HTTP 以外の部分は serve から呼べるライブラリです。対応範囲と実測は [docs/m5.md](docs/m5.md) にあります。モデルの読み込みに約 1 分かかります。コンテキストは、空き VRAM に重みと並んで収まる最大の長さ（上限 262,144 トークン）になり、`--ctx` で指定もできます（131,072 トークンで VRAM 約 79 GiB）。

```bash
nix develop -c cargo build --release -p tsumuji-server
```

```bash
nix develop /mnt/NVME4TB/darask-inference-engine/engines/hayate#default -c bash -c 'NEED=80 LABEL="Tsumuji API" scripts/gpu-run.sh -- target/release/tsumuji-serve --model /mnt/NVME4TB/darask-inference-engine/local/hayate/src/Qwen3.8-Flash-Next-NVFP4 --port 8090'
```

```bash
curl -N http://127.0.0.1:8090/v1/chat/completions -d '{"messages":[{"role":"user","content":"こんにちは"}],"stream":true}'
```

思考を切るには `"chat_template_kwargs": {"enable_thinking": false}` を、思考を短くするには `"reasoning_effort": "low"` を付けます。思考の部分は `reasoning_content` に分けて返します。

## 置き場所

- 使う重み：`/mnt/NVME4TB/darask-inference-engine/local/hayate/src/Qwen3.8-Flash-Next-NVFP4`（RadixArk NVFP4、135GB）
- 使わないもの（参照用の BF16 原本など）：`/mnt/SSD16TB/hayate/` 以下の同じ相対パス
