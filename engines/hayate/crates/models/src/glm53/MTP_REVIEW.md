# MTP 差し戻し対応（567a724 の上）

2026-10-15。比較条件と分岐診断を実装した。GPU はこの sandbox から使えないため、
**分岐原因の実証、n=1..3 の速度向上、CPU/GPU 配置の勝者は未確認**。
MTP は `--spec-draft-n 1..3` のオプトインを維持し、expert 配置の既定も従来の GPU 優先。
適応処理は既存の「最初の8 tokenで通常decodeを測り、遅ければMTPを停止」のまま。
n を実行中に探索する機能と、MTP の既定オン化は追加していない。

## 変更ファイル

- `crates/cli/src/glm53.rs`
- `crates/models/src/glm53/executor.rs`
- `crates/models/src/glm53/experts.rs`
- `crates/models/src/glm53/speculation_gpu.rs`
- `python-tools/hayate_tools/glm53_mtp_bench.py`
- `python-tools/tests/test_glm53_mtp.py`
- `crates/models/src/glm53/MTP_REVIEW.md`

担当外の common runtime / offload / expert_cache / CPU pool、他モデル、third_party、
依存・重み形式は変更していない。fmtによる担当外 `models/src/lib.rs` の宣言順だけ復元した。
ネットワーク、新規crate、重み変換、git commitはなし。

## 比較と設計上の判断

- `--spec-draft-n 0 --spec-reserve-mtp` はMTP層・検証領域・MTP expertを通常MTPと同様に
  読み込み・配置する。生成とprefillではMTPを実行しない。本体expertの容量を揃え、
  容量と配置IDのfingerprintも検査する。予約bytesを手入力する方法は使わない。
- `--mtp-experts cpu` はlayer45の288 expertを全てpinned RAMに置き、既存のCPUカーネルで
  計算する。`pcie-frac=0` が必要。CPU MTP分のRAMを先に予約し、残るGPU slotを本体で使う。
  RAM不足はエラー。MTPのrouteはGPU昇格スコアに加えないため、保守中もMTPはGPUへ昇格しない。
  頻度保存・route/host統計は引き続き数える。形式変換・再量子化はしない。
- ベンチは code / JSON / prose について、MTPなしn=0、GPU配置の予約n=0とn=1..3、
  CPU配置の予約n=0とn=1..3を測る。in-processの強制受理0..n・自然draftのtoken/logits
  ビット一致gateを**各配置で必須のまま**維持する。
- prefillの比較経路は全構成 `--prefill-legacy --prefill-chunk 128` に統一する。
  第12b段階のprefill変更をMTPの容量診断へ混ぜない。保守0では配置を固定し、
  保守4の別runでは通常の入れ替えによる影響も確認する。
- 本体だけのdecode hit数・route数をMTPから分ける。予約n=0対MTPなしn=0の容量差・hit率差・
  速度比を `cache_cost` に出す。MTP有効時の本体routeは棄却された検証行も含むため、
  hit率低下の容量影響はn=0同士で見る。容量・配置hash・MTP expert bytesはwarmup/prefill後の値。
- 非profileのwarmup2・反復5を既定に維持。profileとlogits診断は別run。
  同一構成の反復・診断・profileは生成ID一致が必須。各実行の引数を `.command.json` に残す。
  同一構成内の非決定性は近似品質gateで許容しない。
- 既存 `--dump-logits PREFIX` をMTPでも使えるようにした。受理されたdraftとcorrection/bonusの
  各行に対応する**本体verifier**のBF16 logitsをFP32へ正確に拡張して保存する。
  追加D2Hとファイル保存を行うので、そのrunを速度判定に使わない。
- 最初の分岐はn=0のtop1/top2 ID・logits・差、候補tokenのlogit regretを標準出力とJSONへ出す。
  分岐後は候補の実token列をn=0でteacher forcingし、全位置で同じprefixのlogitsを比較する。
  prefill直後の1行と後続eval行を接続し、token番号を揃える。異なる生成prefix間のKLDは計算しない。

## 品質の事前閾値

- 同一prefixのtop1一致率 **99%以上**。128位置なら不一致は最大1位置。
- `KLD(P_n0 || P_candidate)` は **全位置で0.01以下**。既存prefillの上限と揃え、平均だけで隠さない。
- 全不一致位置と最初の自由生成分岐で、n=0の `top1-top2 <= ln(1.1) = 0.09531018`。
  softmaxのodds比が高々1.1（10%差）という解釈から、GPU測定前に決めた保守的な閾値。
  候補tokenの `top1-logit(candidate)` も同じ上限を要求し、無関係な下位tokenへの交代を通さない。

各構成についてMTPなしn=0・同容量n=0の両方と比較する。不合格構成には速度summaryを付けない。
生の測定値と不合格理由は残し、全構成を確認して非zero終了する。途中例外でもreportを保存する。
合格構成には同容量n=0比・MTPなしn=0比を出し、両方より速い構成だけを配置/nの採用候補にする。
同容量で全nがexact、MTPなし対予約n=0だけ分岐する保守0の結果には
`consistent_with_capacity_drift` と付記する。CPU/GPU誤差の発生箇所を直接実証した意味ではない。
この小さいgreedy評価だけで長文・タスク正答率を保証しない。

## 同期1回は未達・他担当への必要な変更

検証後のGPU受理・復元・MTP cache修復・20 bytesの結果回収は既に1待機に統合済み。
**ステップ全体は1回ではない**。steady state / CPU hybrid / pcie-frac=0のコード上の内訳は、
本体route/CPU入力42回、疎選択11回、最終回収1回、自然draft 3n回で **54+3n回/step**。
profile、初期化、キャッシュ保守、driver内部待機は別。今回この実行経路は変更していない。

担当外 `crates/kernels/src/runtime.rs` には、pinned bufferの寿命と持続CUDAエラー伝播を含む
ホスト処理の非同期投入・完了通知が必要。その上でGLMのroute解決・CPU expert計算・GPU画像選択、
疎KVの和集合/gatherと動的GEMM投入を接続する必要がある。単なるasync copyへの置換では、
未完了データをCPUが読む問題が残る。第11段階と同じ境界で、担当外は変更していない。
担当外 `models/src/lib.rs` のmod順整形も統合側の対応事項。

## 統合担当がそのまま実行するコマンド

既存実機runの `.command.json` を読み、重みと頻度ファイルの実パスを下記へ反映した。
出力先は新しいディレクトリが必要。GPUテストはignore指定なので明示的に実行する。
重み再変換は不要。

```sh
nix develop .#tools --offline
export PYTHONPATH=python-tools
export GLM53_MTP_HYT=/mnt/NVME4TB/hayate/hyt/glm53-mtp-stage11
cargo build --release --features cuda --offline
cargo test -p hayate-models --release --features cuda --offline \
  glm53_mtp_reserved_baseline_and_diagnostic_rows -- --ignored --nocapture --test-threads=1
cargo test -p hayate-models --release --features cuda --offline \
  glm53_mtp_cache_only_repair_bits -- --ignored --nocapture --test-threads=1
cargo test -p hayate-models --release --features cuda --offline \
  glm53_mtp_real_greedy_all_acceptance_counts -- --ignored --nocapture --test-threads=1
python3 -m hayate_tools.glm53_mtp_bench --binary target/release/hayate \
  --hyt "$GLM53_MTP_HYT" --frequencies /mnt/NVME4TB/hayate/spill/glm53-stage6/frequencies.bin \
  --out /mnt/NVME4TB/hayate/spill/glm53-mtp-review-fixed \
  --tokens 128 --warmups 2 --repeats 3 --maintenance-interval 0 --mtp-experts gpu cpu
python3 -m hayate_tools.glm53_mtp_bench --binary target/release/hayate \
  --hyt "$GLM53_MTP_HYT" --frequencies /mnt/NVME4TB/hayate/spill/glm53-stage6/frequencies.bin \
  --out /mnt/NVME4TB/hayate/spill/glm53-mtp-review-maint4 \
  --tokens 128 --warmups 2 --repeats 3 --maintenance-interval 4 --mtp-experts gpu cpu
```

反復3は差し戻し元の実機runと揃えた明示設定。最終採用時は `--repeats 5` でも確認する。
各ベンチは両配置で全受理数のin-process gateを行うため、元の長時間テストと同様に時間が掛かる。
`report.json` の `prompts.<prompt>.configurations.<gpu|cpu>-n<n>` にqualityとspeed summary、
`cache_cost` に容量の影響、`capacity_hypothesis` に切り分け結果、`measured_recommendation` に
両基準より速い配置/nを保存する。最初の分岐のgapは `vs_full_n0.first_divergence` と
`vs_matched_n0.first_divergence`。summaryがない構成は採用不可。

## ローカル検証

PATHにcargo/nvcc/python3がなかったため、既存 `/tmp/glm53-env.sh` をbashでsourceした。
Rust1.98.1 / CUDA13.4 / toolsのtorch・numpy・safetensorsを使い、新規依存は追加していない。

```sh
cargo fmt --all
cargo clippy --workspace --all-targets --offline -- -D warnings
cargo clippy --workspace --all-targets --features cuda --offline -- -D warnings
cargo test --workspace --offline
cargo test --workspace --features cuda --offline
PYTHONPATH=python-tools python3 -m unittest tests.test_glm53_prefill tests.test_glm53_eval \
  tests.test_glm53_decode tests.test_glm53_mtp tests.test_glm53_nvfp4 -v
PYTHONPATH=python-tools python3 -m hayate_tools.glm53_mtp_bench \
  --hyt /tmp/unused-glm53 --out /tmp/glm53c-mtp-revision-no-gpu
git diff --check
```

| 確認 | 結果 |
|---|---|
| fmt | 成功、担当外mod順差分だけ復元 |
| clippy 通常/CUDA | ともに成功、警告0 |
| workspace 通常 | 249成功 / 0失敗 / 19 ignore、doc tests成功 |
| workspace CUDA | 268成功 / 0失敗 / 154 ignore、doc tests成功 |
| Python 5モジュール | 35件中33成功 / GPU 2 skip / 0失敗 |
| GPUなしベンチ | 正常終了、CUDA unavailableのskip JSON、性能値なし |
| diff check | 成功 |

CPUテストではCPU専用MTPのRAM予約とGPU slot返却、昇格候補への混入防止、
45番目の僅差分岐、同prefixのKLD、無関係な下位token・大margin・大KLDの拒否、
GPU/CPU両配置を走らせるベンチ全体の模擬runと、不合格時のsummary非公開を確認した。
新しい実重みignoreテストは、予約n=0と通常MTPの容量/配置、n=1..3の自然生成、
受理/correction/bonusのdump行を通常decodeのlogitsとバイト比較する。
GPUテスト本体・GPU tok/s・ITL・受理率・実VRAM・実分岐gap・実KLDは未測定。
既定の変更に必要な実機根拠はまだない。

参照: `docs/ARCHITECTURE.md` §1/4.3/4.4/5/10、`docs/MODELS.md`、既存GLM実装、
第6〜12段階報告、`spill/glm53-mtp-stage11b/code-n0-r0.command.json` と `spill/gpuq33/bench.log`。
外部Web参照URLなし。ローカルの実機記録は読み取りのみ。
