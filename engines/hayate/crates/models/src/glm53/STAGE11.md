# 第9段階の続き: MTP の疎 KV 一括射影と GPU 復元・cache 修復

**MTP 完成の要件は未達**。疎 KV の一括射影と検証後の同期統合は実装したが、
ステップ全体のホスト同期1回と共通 runtime の非同期化は残る。
共通 runtime は第3〜10段階の担当外であり、今回も変更していない。
GPU はサンドボックスから使用できない。実重みの一致・tok/s・受理率は未確認。

## 変更ファイル

- `crates/kernels/cu/glm53/decode.cu`
- `crates/kernels/src/glm53.rs`
- `crates/models/src/glm53/{gpu.rs,gpu_tests.rs,speculation.rs,speculation_gpu.rs,verification.rs,STAGE11.md}`
- `python-tools/hayate_tools/glm53_mtp_bench.py`
- `python-tools/tests/test_glm53_mtp.py`

qwen38、共通 runtime、offload、expert_cache、依存、third_party は変更していない。
`cargo fmt --all` の担当外 `models/src/lib.rs` の宣言順差分だけは元に戻した。
git commit、重み変換本体、元ファイルへの書き込み、ネットワーク利用は行っていない。

## 設計と理由

- 本体の疎 attention は M=2..4 行の causal な選択 ID と latent append を先に投入する。
  選択 ID をまとめて1回回収し、昇順の和集合を1回の `kv_b` GEMM に渡す。
  重複 latent を繰り返し射影せず、重みの unpack も1回。
  従来と同じ BF16 復元・CUTLASS GEMM を使う。活性化の再量子化はしない。
- 和集合の出力から各 query の K/V を集める小カーネルを追加。
  query ごとの時系列順と件数を戻し、softmax・V 合成は通常 decode のカーネルを使う。
  未来 ID、重複 ID、空集合、不正な負 ID を拒否する。
- 検証後は GPU の受理数で KDA/conv/partial-pool snapshot を選んで復元する。
  最後に使用する本体 hidden も GPU で選ぶ。受理数を先にホストへ戻さない。
- MTP 修復で必要なのは latent KV と index pool。entry norm/eh_proj、attention input norm、
  KV/index key/gate 射影、cache append だけを実行する。
  修復で使わない query/selection/attention/output、MoE、最終 norm/head は実行しない。
  正常 draft の forward と expert の式は従来どおり。
- 修復は最大 n+1 行を投入する。partial pool の上書きだけを `row <= accepted` で制御する。
  棄却行の latent と完成 pool は可視長の外にあり、次の append で上書きされる。
  本体/ドラフトの position は最後の結果回収後に受理長へ合わせる。
  n+1 行分の entry/KV/index 射影は棄却時も計算するため、そのコストは残る。
- 検証・GPU受理・復元・MTP修復を投入してから20 bytesの結果を1回読む。
  旧実装の受理前 D2H と修復後 synchronize を統合した。
  アンカー/強制 draft ID は起動引数を GPU に書く小カーネルにし、pageable H2D 同期も除去。
- CPU expert の M 行共有読み出し、GPU expert 和集合、回転0の kernel省略、E23のホスト配置、
  8 P-core の park 待機は第9〜10段階のものを維持した。

## 残る同期と他担当への必要な変更

検証後の待機は1回になったが、**ステップ全体では1回ではない**。

- 本体 verifier は MoE 42層でルート/CPU入力の D2H 完了を待つ。
- 疎 attention 11層で和集合を作るために選択 ID の D2H 完了を待つ。
- 自然 draft は1トークンにつき sparse 選択回収、latent H2D、ルート回収の3同期が残る。

CPU hybrid、pcie-frac=0、steady state の直接の同期箇所は **54+3n 回/step**。
これはコード上の内訳で実機カウンタ値ではない。初回 profile 設定、キャッシュ転送、
driver内部の暗黙同期などは別。`verifier_d2h_syncs` は verifier の D2H 待ちだけを数え、
draft、修復、最終回収、初回 profile の待機を含めない。

必要な変更:

1. 共通 `crates/kernels/src/runtime.rs` に、ホスト処理の非同期投入・完了通知と、
   pinned buffer の寿命/エラー伝播を扱う手段が必要。持続 CUDA エラーでは投入を止める。
2. GLM のルート解決/CPU expert 計算をその非同期経路へ接続する。
   GPU resident expert のアドレス/配置も、ホストでルートを読まず実行できる形が必要。
   同じ CPU expert を全 M 行で1回読む現在の方式と E23/P-core 分離を維持する。
3. 疎 KV の和集合・gather も非同期実行へ移す。同期 API を単に async copy に変えるだけでは、
   未完了の ID/latent を CPU が読んでしまう。Tensor Core GEMM の動的行数を含む投入方法を決める。

今回はユーザーの「担当外に必要なら変更せず報告」に従い、共通 runtime は変更していない。
GPU 実行部分をそのまま Graph に capture 可能になったとは主張しない。

## 容量と計測上の注意

選択集合は query ごとに変わり、和集合を2051行に制限できない。
起動時に各疎層へ `C = clamp(context, 2051, 4*2051)` 行の KV/latent 領域と
選択 ID・写像を確保する。通常 decode の K/V 領域は再利用する。

| context | 第9段階からの追加 GPU bytes | 追加 pinned bytes |
|---|---:|---:|
| 4096 | 1,497,989,152 | 46,498,320 |
| 8192 | 4,496,916,512 | 92,635,664 |
| 131072 | 4,505,702,432 | 92,770,832 |

11疎層の allocation サイズからの計算。GPU の速度/driver 使用量の測定ではない。
context4096 の `MTP_added_VRAM_bytes` は第9段階の1,293,559,088から
**2,791,548,240 bytes**となる見込み。起動時の実 allocation 集計をログで確認する。
expert cache がこの分小さくなるため、KV射影の削減だけで速度向上とは判断できない。
GPU/CPU の配置で演算順が変わる第8段階の性質も残り、n=0との出力比較が失敗する可能性がある。
その場合は配置/CPU-GPUの数値差を先に切り分け、bit一致の条件は緩めない。

## 正しさ

- CPU: 和集合の順序・復元写像・未来/重複ID拒否、4-token pool境界と2048-tokenの
  選択予算境界、2051行を超える和集合を確認。
- CPU: n=1..3 × 受理0..n × 8開始位置の72条件で、最大長の修復から可視prefixと
  partial poolを取り出し、受理行だけの逐次処理と比較。
  第9段階の160token以上の状態/greedy一致テストも維持する。
- GPU `#[ignore]`: snapshot/hiddenの動的復元とpartial-poolの条件付きコピーを72条件で比較。
  非256倍数の299 bytesを使い、末尾も確認する。
- GPU `#[ignore]`: 実KV形状32768×512、非単位FP8 scale、1/3/127/131行、
  128行タイル境界を跨ぐ union からの gather と通常射影の K/V・attention をバイト比較。
- 実重み `#[ignore]`: full forward と cacheだけのMTP修復を72条件でバイト比較。
- 実重み `#[ignore]`: n=1..3、強制受理0..nと自然draftで、通常decodeとのtoken/logits完全一致。
  既存3promptに2051tokenのpromptを追加し、選択予算超過も通す。
  同じExecutor内で配置を固定して比較する。

## 実機の変換・検証・計測コマンド

メインPCで実行する。既存重みを上書きしない新しい出力先を使う。
下記dense指定は例であり、本体の採用品質設定と揃える。

```sh
nix develop .#tools --offline
export PYTHONPATH=python-tools
export GLM53_MTP_HYT=/mnt/NVME4TB/hayate/hyt/glm53-mtp-stage11
python3 -m hayate_tools.glm53_convert --layers 45 --mtp --expert-quant nvfp4 \
  --dense shared=nvfp4 --dense kda=fp8 --dense sparse=fp8 \
  --dense dense-ffn=fp8 --dense lm-head=fp8 --device cuda \
  --out "$GLM53_MTP_HYT" --estimate
python3 -m hayate_tools.glm53_convert --layers 45 --mtp --expert-quant nvfp4 \
  --dense shared=nvfp4 --dense kda=fp8 --dense sparse=fp8 \
  --dense dense-ffn=fp8 --dense lm-head=fp8 --device cuda --out "$GLM53_MTP_HYT"
cargo build --release --features cuda --offline
cargo test -p hayate-kernels --release --features cuda --offline \
  glm53_gpu_prefix_restore_and_partial_pool -- --ignored --nocapture --test-threads=1
cargo test -p hayate-kernels --release --features cuda --offline \
  glm53_verification_batch_dense_bits_and_gpu_acceptance -- --ignored --nocapture --test-threads=1
cargo test -p hayate-models --release --features cuda --offline \
  glm53_sparse_union_projection_and_attention_bits -- --ignored --nocapture --test-threads=1
cargo test -p hayate-models --release --features cuda --offline \
  glm53_mtp_cache_only_repair_bits -- --ignored --nocapture --test-threads=1
cargo test -p hayate-models --release --features cuda --offline \
  glm53_mtp_real_greedy_all_acceptance_counts -- --ignored --nocapture --test-threads=1
python3 -m hayate_tools.glm53_mtp_bench --binary target/release/hayate \
  --hyt "$GLM53_MTP_HYT" --out /mnt/NVME4TB/hayate/spill/glm53-mtp-stage11 \
  --tokens 128 --warmups 2 --repeats 5
```

頻度で配置する場合は、最後のコマンドに `--frequencies <採用した頻度ファイル>` を追加する。
モデルの変換元は既定の `/mnt/NVME4TB/hayate/src/GLM-5.3-Flash-Uncensored-FP8`。
別の変換元なら変換器の `--source` とベンチの `--model-dir` を揃える。

ベンチはcode/JSON/proseで正しさgateを先に実行し、n=0..3を非profileで比較する。
profileは別runにし、その出力IDもn=0と比較する。失敗時は速度reportを公開しない。
`report.json` の `prompts.<prompt>.summary` に非profileのmedian tok/s、n=0比、加重受理率、
run内平均step msのmedianを保存する。`profiles.<n>` は各stepと45層の内訳。
CPU expert数/bytes/wall/worker/dispatch、KV選択行数/和集合行数、D2H待機回数も含む。
CUDA event区間とhost投入/待機区間は重なるので合算しない。
GPUなしではskip理由だけを保存し、速度値を生成しない。

## ローカルのコマンドと結果

初期PATHにcargo/nvcc/python3がなかったため、既存 `/tmp/glm53-env.sh` をsourceし、
インストール済みのNix Rust1.98.1/CUDA13.4/Python3.14とtorch/numpy/safetensorsを使用した。

```sh
cargo fmt --all
cargo clippy --workspace --all-targets --offline -- -D warnings
cargo clippy --workspace --all-targets --features cuda --offline -- -D warnings
cargo test --workspace --offline
cargo test --workspace --features cuda --offline
PYTHONPATH=python-tools python3 -m unittest tests.test_glm53_mtp tests.test_glm53_decode \
  tests.test_glm53_eval tests.test_glm53_nvfp4 -v
PYTHONPATH=python-tools python3 -m hayate_tools.glm53_mtp_bench \
  --hyt /tmp/unused-glm53-stage11 --out /tmp/glm53-stage11-no-gpu
git diff --check
```

最終結果（2026-10-15）。GPU ignoreは実行成功に数えない。

| 確認 | 結果 |
|---|---|
| fmt | 成功。担当外の宣言順差分のみ復元 |
| clippy 通常/CUDA | ともに成功、`-D warnings` で警告0 |
| workspace 通常 | **238成功 / 0失敗 / 19 ignore**、doc tests成功 |
| workspace CUDA | **257成功 / 0失敗 / 137 ignore**、doc tests成功 |
| Python 4モジュール | **25件 / 23成功 / GPU 2 skip / 0失敗**（2.128秒） |
| GPUなしの計測器 | 正常終了、`CUDA unavailable; no performance measurements` のskip JSON |
| diff check | 成功 |

CUDA sm_120aのカーネルと新しいignoreテストはコンパイルした。
途中のclippyで定数幅`chunks_exact`の指摘を受け、既存の`as_chunks`形式へ修正して再実行した。
実重み変換、GPUテスト本体、tok/s、受理率、TTFT、実機VRAM、追加領域による命中率の変化、
Graphでの実行とステップ全体の同期1回は確認できていない。新たな実機の速度測定値はない。

## 参照

`docs/ARCHITECTURE.md`（特に§1/4.3/4.4/5/10）、`docs/MODELS.md`、GLM第6〜10段階。
Qwen `STAGE13.md`、および別worktreeの
`../qwen38/crates/models/src/qwen38/STAGE14.md`〜`STAGE16.md` を読み取りのみで参照した。
cache修復で予測forwardを実行しないこと、固定M、FMA/BF16順序、GPU受理/復元、
host同期回数とprofile区間を混同しないことを反映した。外部Web参照URLはなし。
