# Hayate（疾風）

自分の **GLM-5.3-Flash 専用** Rust + CUDA 推論エンジン。対象は RTX PRO 6000 Blackwell 96 GB、i9-12900KS、RAM 128 GB、NixOS。CUDA は sm_120a のみをビルドする。

45 層の GLM、LOWBIT / EXL3 / NVFP4 / FP8 の変換済み重み、GPU と CPU のエキスパート並列実行に絞る。頻度の高いエキスパートを VRAM に置き、残りを RAM の AVX2 ワーカーで計算する。画像・動画生成と他の LLM の実装は削除した。モデル重みや Studio・学習ツールは別管理。

## ビルド

```sh
nix develop --impure
scripts/fetch-third-party.sh
scripts/build.sh
```

出力は `target/main/x86_64-unknown-linux-gnu/release/hayate`。直接 `cargo build --release -p hayate-cli --features cuda` でもビルドできる。CUDA なしの `cargo test --workspace` では CPU・形式検証・HTTP ワーカー・チャットテンプレートを確認できる。

## 実行

```sh
hayate serve
hayate capabilities --json
hayate run glm53 --help
```

`serve` は `127.0.0.1:7870`。既定の変換済み重みは `/mnt/NVME4TB/darask-inference-engine/local/hayate/hyt/glm53x3-sc260`、トークナイザーは `/mnt/NVME4TB/darask-inference-engine/local/hayate/src/GLM-5.3-Flash-dealignai-FP8`。`--glm-hyt`、`--glm-source` で変更できる。

```sh
curl http://127.0.0.1:7870/v1/models/load -H 'Content-Type: application/json' -d '{"id":"glm53"}'
curl http://127.0.0.1:7870/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"glm53","messages":[{"role":"user","content":"こんにちは"}],"stream":true}'
```

OpenAI 形式の SSE、思考と本文の分離、ツール呼び出し、会話プレフィクス再利用、GGUF LoRA、接続切断時の停止に対応。外部ジョブが VRAM を申告すると、要求の合間にエキスパートキャッシュを縮め、枠が空くと元の予算まで戻す。

## 実装と検証

- [構成](docs/ARCHITECTURE.md)、[チャット API](docs/GLM53_SERVE.md)、[GPU 共有](docs/GPU_SHARING.md)
- [GLM 実装と過去の実測](crates/models/src/glm53/README.md)
- [DeepInfer から参考にした最適化](docs/DEEPINFER.md)
- `python-tools/hayate_tools/glm53_*`: 変換・量子化・参照出力・評価・ベンチマーク

Hayate は [Apache-2.0](LICENSE)。FlashInfer / CUTLASS と参考にした DeepInfer の出典は [NOTICE](NOTICE) を参照。モデル重みにはそれぞれのライセンスが適用される。
