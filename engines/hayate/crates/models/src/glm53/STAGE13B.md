# 第13b段階: 配置の決定性と prefill 品質ゲート

基点 `25f0cd1`。`docs/ARCHITECTURE.md` §1/4.3/4.4/5/10 と
`docs/MODELS.md` を確認した。担当内だけを変更し、crate 追加・外部通信・commit はしていない。
GPU は使えないため、以下の GPU テストと Gen5 性能は未確認。

## 配置

旧ローダは resident 確保後の `cudaMemGetInfo` から cache 容量を引き算していた。
driver/context、他プロセス、再ロード時の残存 allocation により1スロットでも
容量が変わると、頻度順で選ぶ expert 集合も変わっていた。

今回の容量は次の固定値から計算する。

```
prefill の自動容量用予算 = (MAIN.planned_vram() - resident の所有バイト数) / 4
expert 予算 = MAIN.planned_vram() - resident の所有バイト数 - 8 GiB - IMAGE_BYTES
```

resident は globals、全層の重み・state・workspace、MTP 検証領域、
executor の input/logits、確保済み prefill 領域の実際の `DeviceBuffer::bytes()` の合計。
`MAIN.planned_vram()` は既存の 97,887 MiB - 2 GiB。
従来の8 GiB一時領域と staging image の余白も維持した。
自動 prefill 容量も固定予算に変え、そこから配置へ空き容量の揺れが伝わる経路を除いた。
sync/async の mailbox は cache 計画を変えない。MTP 予約ありの n=0 と n=1..3 も
同じ resident 領域を持つ。頻度順・同頻度時の順番・HYT の量子化形式は従来どおり。

実際の free は admission にだけ使う。不足時には容量を縮めず `Error::OutOfMemory`。
scratch や転送バッファの後続 allocation が不足した場合も、その確保エラーを返す。
ロードログに `GLM fixed memory plan` と resident/capacity/expert_budget/free を出す。
従来の配置 hash/バイト数との一致は要求しない。新方式で比較する全ロード同士の一致を要求する。

CPU 回帰テストは42層×288 expert、同頻度を含む頻度表を使い、free を
1 byte、1 image、512 MiB、2 GiB など動かして配置全体の一致を確認する。
予算不足時に小さい配置で続行しないことも確認する。
既存の実機配置テストには、sync 解放後・async 再ロード前の64 MiB確保を加えた。

async ベンチは品質時と各速度測定時の **hash と GPU bytes の両方**を確認し、
n=0..3 間も同じ配置を要求する。各 n の固定配置を JSON に保存する。
生成 ID・全 logits bytes の一致、async の1同期/stepのゲートは維持した。
CPU モックで正常完走、hash が同じでも容量が変わる失敗、n 間の配置が変わる失敗を検証した。

## all=true の KLD

統合担当の `25f0cd1` 実機結果は all=false が KLD=0、all=true の
prefill_start=0/suffix_row=0 が **0.01435**。
短い previous は `cpu_short_prompt` により既存の逐次 `step` を使うため、
all=false の一致だけでは GPU batch の数値一致を示さない。

既存コードを確認した結果:

- routed NVFP4 は E2M1×E4M3 を BF16 に正確に展開し、grouped GEMM の
  FP32 accumulator に行列 scale を掛けてから BF16 にする。前段階で余分な
  重み BF16 丸めを除く修正が入っており、今回は戻していない。
- decode/CPU は行列 scale を含む値を FP32 積和する。行列 scale の掛ける位置と
  GEMM/warp/CPU の FP32 加算順は異なる。dense の batch GEMM と GEMV も加算順が異なる。
- expert 合算は両経路とも expert ID 昇順で、積と部分和の BF16 丸めも維持されていた。
- 既存の小行列 GPU テスト `glm53_prefill_nvfp4_global_scale` は前回実機で成功済み。
  CPU の独立 FP64 参照でも scale 分離方式の誤差が小さいことを今回再確認した。

この結果とユーザーから提示された Qwen3.8 の同種の0.013〜0.0145を根拠に、
許可された **行単位 KLD≤0.02** を GPU all の Rust テストと Python prefill 比較に適用した。
Rust の previous ケースは0.01を維持する。Python は prefill 末尾 logits と
全 teacher-forced 後続行を個別に0.02で確認し、平均で外れ値を隠さない。
FP8対NVFP4の量子化品質ゲートは変更していない。
CPU 回帰で0.01〜0.02の差を許容し、一行だけ0.02を超える差を prefix/suffix の双方で拒否する。

**FP32 積和差が45層 KLD差の唯一の原因だとは、GPUなしでは確定できない。**
同じ all batch を reset 後に再実行して5行の logits の完全一致を要求する検査を追加した。
これで再利用/DMAに伴う非再現的な差も実機で検出する。計算カーネルを
一致目的で遅い経路に置き換えることや、未計測の最適化はしていない。

## ローカル検証

PATH に cargo/nvcc/python3 がなかったため、既存の `/tmp/glm53-env.sh` を
bash で読み込んだ。既存 Nix store の tools 環境(Rust 1.98.1/CUDA 13.4/
torch/numpy/safetensors)を使用し、追加取得はない。

```sh
cargo fmt --all
cargo clippy --workspace --all-targets --offline -- -D warnings
cargo clippy --workspace --all-targets --features cuda --offline -- -D warnings
cargo test --workspace --offline
cargo test --workspace --features cuda --offline
PYTHONPATH=python-tools python3 -m unittest tests.test_glm53_async tests.test_glm53_eval \
  tests.test_glm53_prefill tests.test_glm53_decode tests.test_glm53_mtp tests.test_glm53_nvfp4 -v
git diff --check
```

| 確認 | 結果 |
|---|---|
| fmt / clippy 通常・CUDA | 成功、`-D warnings` で警告0 |
| workspace 通常 | 253成功 / 0失敗 / 19 ignore、doc tests成功 |
| workspace CUDA | 272成功 / 0失敗 / 160 ignore、doc tests成功 |
| Python 6モジュール | 43件中41成功 / GPU 2 skip / 0失敗 |
| CPU 配置 | 12,096 expert、6種類のfree量で配置一致、不足時エラー |
| CPU KDA 独立数式 | 最大相対 L2=9.89216574e-7 |
| CPU NVFP4 スケール | FP32 rescale L2=7.03489689e-8 / 旧方式0.00156952193 |
| CPU dense | 9ケース、最大相対 L2=2.78416453e-8 |
| GPUなし prefill ベンチ | CUDA unavailable の skip JSON、性能値なし |

GPUなしベンチの確認コマンド:

```sh
PYTHONPATH=python-tools python3 -m hayate_tools.glm53_prefill_bench \
  --hyt /tmp/unused-glm53 --out /tmp/glm53-stage13b-no-gpu
```

fmt の既存 module 順序変更だけが出る担当外 `crates/models/src/lib.rs` は元に戻した。

## 実機コマンド(Gen5、統合担当)

前回と同じ HYT・frequencies を使い、出力先は新しくする。
frequencies を使う場合、両ベンチに同じ `--frequencies FILE` を加える。

```sh
nix develop .#tools --offline
export PYTHONPATH=python-tools
export GLM53_HYT=/mnt/NVME4TB/hayate/hyt/glm53-stage6-fp8-dense
export GLM53_MTP_HYT=/mnt/NVME4TB/hayate/hyt/glm53-mtp-stage11
export GLM53_SRC=/mnt/NVME4TB/hayate/src/GLM-5.3-Flash-Uncensored-FP8
cargo build --release -p hayate-cli --features cuda --offline

# 配置・sync/async・MTP全長・全logits bytes・同期1回/step。64 MiBの確保差も加える。
cargo test -p hayate-models --release --features cuda --offline \
  glm53_async_same_placement_decode_and_mtp_bits -- --ignored --nocapture --test-threads=1
cargo test -p hayate-models --release --features cuda --offline \
  glm53::async_tests -- --ignored --nocapture --test-threads=1

# n=0単独、MTP予約なし。
python3 -m hayate_tools.glm53_async_bench \
  --binary target/release/hayate --hyt-dir "$GLM53_MTP_HYT" --model-dir "$GLM53_SRC" \
  --n 0 --tokens 128 --warmup 2 --repeats 5 \
  --out /mnt/NVME4TB/hayate/spill/glm53-stage13b-gen5-n0

# n=0にもMTPを予約して配置を揃える。CPU MTPは別出力先で --mtp-experts cpu を加える。
python3 -m hayate_tools.glm53_async_bench \
  --binary target/release/hayate --hyt-dir "$GLM53_MTP_HYT" --model-dir "$GLM53_SRC" \
  --n 0 1 2 3 --tokens 128 --warmup 2 --repeats 5 \
  --out /mnt/NVME4TB/hayate/spill/glm53-stage13b-gen5-mtp

# unscaled W4 GEMM/GEMVの局所比較と、pinned DMAのGen5再計測を含む。
cargo test -p hayate-kernels --release --features cuda --offline \
  glm53_prefill_ -- --ignored --nocapture --test-threads=1
# previousの0.01、allの0.02、およびallの同一形状再実行一致。
cargo test -p hayate-models --release --features cuda --offline \
  glm53_prefill_matches_tokenwise_prefix_logits -- --ignored --nocapture --test-threads=1
python3 -m hayate_tools.glm53_prefill_bench --hyt "$GLM53_HYT" \
  --out /mnt/NVME4TB/hayate/spill/glm53-stage13b-gen5-prefill-smoke \
  --lengths 2048 --warmups 0 --repeats 1 --expert-policies previous all
```

過去の転送・prefill・ロード時間は **Gen1** の値であり、Gen5 の比較値にしない。
今回の Gen5 tok/s、ITL、帯域、ロード時間、実重み KLD、sync/async bit一致は未計測。
all再実行が不一致なら、閾値をさらに広げず転送/状態再利用を調べる。

## 変更ファイルと他担当への依頼

- `crates/models/src/glm53/executor.rs`
- `crates/models/src/glm53/experts.rs`
- `crates/models/src/glm53/gpu_tests.rs`
- `crates/models/src/glm53/speculation_gpu.rs`
- `crates/models/src/glm53/STAGE13B.md`
- `python-tools/hayate_tools/glm53_async_bench.py`
- `python-tools/hayate_tools/glm53_eval.py`
- `python-tools/tests/test_glm53_async.py`
- `python-tools/tests/test_glm53_eval.py`

担当外への必須コード変更はない。統合担当は上記実機検証と Gen5 再計測を実行し、
新しい固定予算の slot 数・配置 hash/GPU bytes が各ロードで一致することを確認する。
外部 URL は参照していない。仕様の確認にはローカルの既存実装のみを用いた。
