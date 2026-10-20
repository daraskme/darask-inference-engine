# 第9段階: MTP と greedy 投機的デコード

CPU 検証と sm_120a ビルドを実施。GPU が使えないため、実重みの greedy/logits 一致、
受理率、n=0..3 の tok/s、追加 VRAM の実測は未確認。
**高速化の完了条件は未達**。受理判定は GPU 上だが、CPU expert のルート回収、疎 KV
の gather、MTP 修復に待ちがあり、要求されたホスト同期 1 回/step は実現していない。
疎 attention の選択後 kv_b 射影も従来の各行 GEMM を維持する。

## 変更ファイル

- `crates/cli/src/glm53.rs`
- `crates/kernels/cu/glm53/{decode.cu,dense.cuh}`
- `crates/kernels/src/{glm53.rs,cpu/glm53.rs}`
- `crates/models/src/glm53/{mod.rs,executor.rs,experts.rs,gpu.rs,cpu_experts.rs}`
- `crates/models/src/glm53/{speculation.rs,speculation_gpu.rs,verification.rs,STAGE9.md}`（新規）
- `python-tools/hayate_tools/{glm53_convert.py,glm53_mtp_bench.py}`（後者は新規）
- `python-tools/tests/test_glm53_mtp.py`（新規）

qwen38、共通 runtime/offload/expert_cache、third_party、依存 crate は変更しない。
fmt が触る担当外 `crates/models/src/lib.rs` の既存 mod 順差分は最後に復元する。
commit なし。大きな重み変換・GPU 実行・元ファイルへの書き込みなし。

## 実装と判断

- `--mtp` 変換は 45 層本体に層45を追加する。NVFP4 experts、dense 形式、回転、MSE、
  down FP8 は既存の指定を引き継ぐ。層45は sparse と判定し、`eh_proj` も sparse の
  dense 形式を使う。norm/router は既存形式。未知の NextN 層数、45層未満を拒否する。
- Transformers 5.17 の `glm5_next` を読んだが、この版は層45と shared_head を ignore
  しており、MTP の forward は存在しない。式を同版で確認済みとはしない。
  実 safetensors ヘッダと、下記の公開一次実装で補った。
- MTP は本体の最終正規化後 hidden と次 ID の共有 embedding を、それぞれ hnorm/enorm
  で RMSNorm し、embedding 側を先に連結して eh_proj に渡す。
  mHC のない sparse attention と通常の残差、MoE を通し、専用 shared_head.norm と
  共有 lm_head で予測する。再帰 draft には MTP の正規化後 hidden を渡す。
  prompt は本体 prefill の各行 hidden と次 ID で MTP KV を初期化する。
- 本体検証は `[anchor, draft0, ...]` の n+1 行を層ごとに1回通す。
  dense と router は 1..4 行の専用 FMA 累積で、重みの復号・読み出しを共有する。
  BF16 境界、warp reduction 順、routed NVFP4 の未丸め係数を通常 decode に合わせる。
  W4A4、activation 量子化、prefill grouped GEMM、全モデル decode の n+1 回反復は使わない。
- routed experts は和集合を作り、各 expert を1回ずつ処理する。GPU resident と shared
  を先に投入し、CPU miss は同じ重みを読みながら最大4行の AVX2/FMA を計算する。
  CPU 入力と全ルートはまとめて回収、出力は pinned buffer から非同期 H2D する。
  最後は通常 decode と同じ expert ID 昇順、multiply/add ごとの BF16 丸めで合算する。
  回転0では回転 kernel を起動しない。
- CPU の最初の動的行数ループは実重みベンチで遅かったため、1..4 行を固定長で
  コンパイルする形に修正。将来の一般形状や別 CPU 用の分岐は追加しない。
- KDA の F32 state と q/k/v short-conv の履歴を prefix 0 と各入力後に D2D 保存。
  sparse latent KV/完成 pool は position で末尾を不可視にする。未完成 pool の
  key/gate は上書きされるため各 prefix を保存する。受理 a 個なら anchor+a 入力後へ戻す。
- greedy と一致 prefix の判定は GPU。受理 ID と correction/bonus だけをまとめて回収。
  MTP は draft KV を戻して、受理した本体 hidden で cache を修復する。
  修復を終える待ちまで step 時間に含める。
- `--spec-draft-n 0..3`、既定0。混合 PCIe 比率の error-diffusion が通常 decode と検証で
  異なる割り当てになるのを避け、MTP は `--pcie-frac 0|1` に限る。
  既定では通常 decode 8 token の基準を測り、8 spec step 後に時間/token が基準以上なら
  残りの生成で投機を停止。比較用 `--spec-no-adaptive` はこの停止を無効にする。
- `--profile-spec` は CUDA event による層内訳と D2H 待ち、draft/verify の host 投入時間、
  acceptance 完了待ち、repair、完了までの step wall を出す。CUDA event 区間には CPU
  処理/投入の隙間が入り、純 kernel 時間ではない。内訳は合算・二重加算しない。

## 正しさと計測のコマンド

メイン PC で以下を実行する。既存 HYT を上書きせず新しい出力先を指定する。
`--dense` は第8段階で採用したものと揃える。以下は shared NVFP4 / 他 dense FP8 の例。

```sh
nix develop .#tools --offline
export PYTHONPATH=python-tools
export GLM53_MTP_HYT=/mnt/NVME4TB/hayate/hyt/glm53-stage9-mtp
python3 -m hayate_tools.glm53_convert --layers 45 --mtp --expert-quant nvfp4 \
  --dense shared=nvfp4 --dense kda=fp8 --dense sparse=fp8 \
  --dense dense-ffn=fp8 --dense lm-head=fp8 --device cuda \
  --out "$GLM53_MTP_HYT" --estimate
# 実変換は --estimate を外す。元 FP8 を復元して変換する。全 source SHA256 と I/O を伴う。
python3 -m hayate_tools.glm53_convert --layers 45 --mtp --expert-quant nvfp4 \
  --dense shared=nvfp4 --dense kda=fp8 --dense sparse=fp8 \
  --dense dense-ffn=fp8 --dense lm-head=fp8 --device cuda --out "$GLM53_MTP_HYT"
cargo build --release --features cuda --offline
cargo test -p hayate-kernels --release --features cuda --offline \
  glm53_verification_batch_dense_bits_and_gpu_acceptance -- --ignored --nocapture --test-threads=1
cargo test -p hayate-models --release --features cuda --offline \
  glm53_mtp_real_greedy_all_acceptance_counts -- --ignored --nocapture --test-threads=1

# 3系統のprompt、強制受理0..nと自然draftを先に検証、warmup2/repeat5で全nを測る。
python3 -m hayate_tools.glm53_mtp_bench --binary target/release/hayate \
  --hyt "$GLM53_MTP_HYT" --frequencies /path/to/stage8/frequencies.bin \
  --out /mnt/NVME4TB/hayate/spill/glm53-stage9 --tokens 128 --warmups 2 --repeats 5

# 通常利用。遅ければ当該生成中に自動停止する。
target/release/hayate run glm53 --hyt-dir "$GLM53_MTP_HYT" \
  --prompt 'Write a Rust JSON parser.' --max-tokens 128 --spec-draft-n 3
# 内訳取得は別実行に --profile-spec を付ける。速度の採否は profile なしで判定する。
```

CPU 参照は n=1..3 × 受理0..n × prompt境界8種類の72条件で160 token以上、
出力と再帰/conv/KV/partial-pool 状態の完全一致を確認する。
実重みテストは各 n、全強制受理数、自然 draft で token と同一 prefix logits bytes の
完全一致を要求する。各条件でリセットし、3種類の固定 ID prompt を使う。
validator では配置の違いを避けるため maintenance を停止する。
計測器は maintenance4 に戻し、n=0 を含む全 run の生成 ID を照合する。
失敗時はログと command を残して失敗終了し、速度の report.json を公開しない。
GPU 不可の場合は skip JSON だけを作り、架空の速度を報告しない。

`report.json` は各 run の TTFT、tok/s、ITL、受理率、step ms、追加 allocation、MTP expert
GPU bytes、peak VRAM と、n=0 を上回った最速の n（なければ0）を記録する。
投機の ITL は step wall/出力数で、実際の token 到着は burst になる。

## 容量

実ヘッダの estimate（変換本体・GPU実行の測定ではない）。上記 dense、回転0の場合。

| 項目 | bytes |
|---|---:|
| MTP routed NVFP4 payload | 4,076,866,944 |
| MTP dense/norm/router payload | 175,028,916 |
| MTP payload 合計 | 4,251,895,860 |
| 本体+MTP HYT 全体（table/alignment込み） | 184,821,608,632 |
| 本体 verifier の追加 buffer/snapshot（最大4行） | 810,104,192 |
| MTP 層の dense/state/workspace/verifier（context4096） | 482,035,584 |
| executor speculative buffers（vocab154880） | 1,419,312 |
| expert cache を除く追加 device allocation 合計 | 1,293,559,088（1.205 GiB） |

末尾4行は Rust の allocation サイズからの計算で、driver 実測ではない。
`MTP_added_VRAM_bytes` は上記 allocation 合計、`MTP_expert_GPU_bytes` は測定時の層45の
resident routed画像容量（0..約4.077GB）。これらは追加で必要な領域の内訳であり、
モデル全体の expert cache を再配分するため、n=0 との差の driver使用量とは一致しない。
`peak_VRAM_bytes` も記録する。context 依存 KV、profileイベント/driver overhead は実機で確認する。
MTP latent KV の追加 pinned RAM は context×1024 bytes。検証用 pinned buffers は別。

## この環境の検証

PATH に cargo/nvcc/python3 がなかったため、既存 `/tmp/qwen38-stage4-env.sh` の Nix
ツールチェーンを使用。外部 crate の追加・取得なし。Python は torch/numpy/safetensors
と標準ライブラリのみ。Transformers はソースを読んだだけで実行依存には追加しない。

```sh
cargo fmt --all
cargo clippy --workspace --all-targets --offline -- -D warnings
cargo test --workspace --offline
cargo clippy --workspace --all-targets --features cuda --offline -- -D warnings
cargo test --workspace --features cuda --offline
PYTHONPATH=python-tools python3 -m unittest tests.test_glm53_mtp tests.test_glm53_decode \
  tests.test_glm53_eval tests.test_glm53_nvfp4 -v
nvcc -c -O3 -std=c++20 --use_fast_math -gencode=arch=compute_120a,code=sm_120a \
  -Xptxas=-v crates/kernels/cu/glm53/decode.cu -o /tmp/glm53-stage9-decode.o
```

最終結果とCPU実重みベンチは末尾に追記する。
CUDA dense batch の2..4行は40 registers、spill stores/loads 0。
これはコンパイル資源量であり GPU 速度の実測ではない。

## 未確認・残る必要な変更

- GPU 変換本体、実機 greedy/logits、受理率、tok/s、TTFT、peak VRAM、adaptive の実測。
- ホスト同期1回/step は未達。現行の host-cache resolver と RAM sparse KV gather が
  GPU結果を層途中で待つ。これを無くすには GPU/CPU 間の常駐キュー・非同期完了通知と
  KV gather の実行方式の変更が必要。共通 runtime は担当外なので変更していない。
- sparse selected KV の kv_b GEMM は通常 decode の演算を守って各入力で実行する。
  全行の選択集合の和集合を1回展開する実装と、GEMM行配置を変えた際の bit比較が残る。
  mHC と recurrent/short-conv の小処理も位置ごとに実行。Tensor Core高速化は未実装。
- CPU と GPU の既存 GEMV は加算順が異なる（第8段階の仕様）。MTP のVRAM確保や
  maintenance で配置が変われば、通常 decode 側も結果が変わる可能性がある。
  計測器の n=0 比較が失敗したら、まず第8段階のCPU/GPUの一致/配置を切り分ける。
  誤差許容で本検証の bit一致要求を緩めない。
- MTP prompt初期化と受理後修復は逐次。追加head計算やhost投入の時間を計測して判断する。
- 共通 `models/src/lib.rs` の mod 順の整形は統合担当で対応が必要（既存差分）。

## 参照（名前・形状・式のみ。コードのコピーなし）

- 指定の `docs/ARCHITECTURE.md`（§1/4.3/4.4/5/10）、`docs/MODELS.md`。
- GLM 第6〜8段階、Qwen `speculation.rs` / `STAGE13.md`。STAGE14/15.md はこのworktreeにない。
  ユーザー提示の第14〜15段階の失敗点も制約として反映した。
- ローカル Transformers **5.17.0** `models/glm5_next/modeling_glm5_next.py`。
- 元 FP8 config と層45の safetensors ヘッダ（1760 tensor、mHCなし）。
- [llama.cpp GLM5-Next 本体とMTPの一次実装](https://github.com/ggml-org/llama.cpp/blob/master/src/models/glm5-next.cpp)
  （本体 post-final-norm hidden、concat 順、通常 residual、shared head norm）。

## 最終結果（2026-10-15）

- `cargo fmt --all`: 成功。担当外 lib.rs の既存 mod 順のみ復元。
- clippy 通常/CUDA: 成功、`-D warnings` で警告0。
- `cargo test --workspace --offline`: **233 passed / 0 failed / 19 ignored**、doc tests成功。
- 同 `--features cuda`: **252 passed / 0 failed / 133 ignored**、doc tests成功。
- 上記 Python 4モジュール: **23 tests / 21成功 / GPU2 skip / 0失敗**。
- 新規CPU試験3件に加え、既存CPU expert試験に複数行FFNの単一行とのbyte一致を追加。
  新規 GPU 試験2件は ignore。ignore/skip をGPU実行成功とは数えない。
- 実ヘッダ estimate は上記容量。GPUなしの計測器は skip JSON を出して正常終了。
- nvcc は成功。batch dense 2..4行は40 registers、spill stores/loads 0。
- `git diff --check`: 成功。

CPU 実重みベンチは stage8 と同じ layer3 の32画像、453,009,408 bytes（LLCより大きい）。
2 sweep warmup + 5 sweep 測定、8 P-core固定、SSD読込を時間から除外。
FFN全体とpool同期を含む。M行は同じexpertを選んだM tokenを模擬する。
自分の全体テスト/ビルド終了後に1..4行を連続実行した。外部ジョブは制御できず、
loadavgは約26.6だった。これはモデルの受理率・tok/sではない。

```sh
cargo test -p hayate-models --release --offline --no-run
# 今回の生成binaryを直接使用し、途中の再ビルドや自分の他ベンチとの重なりを除いた。
for rows in 1 2 3 4; do
  GLM53_CPU_ROWS=$rows target/release/deps/hayate_models-b80a6f1dbb0dfc6a \
    glm53_cpu_expert_bandwidth --ignored --nocapture --test-threads=1
done
# 統合時は上記binaryの代わりに次を使用できる:
# GLM53_CPU_ROWS=4 cargo test -p hayate-models --release --offline \
#   glm53_cpu_expert_bandwidth -- --ignored --nocapture --test-threads=1
```

| 行数 | 5回のwall秒/32 experts | 中央値秒 | 有効GB/s | ms/expert/token |
|---|---|---:|---:|---:|
| 1 | .106747/.082466/.088054/.098760/.067672 | .088054 | 5.144677 | 2.751688 |
| 2 | .096044/.097378/.098765/.100334/.086418 | .097378 | 4.652071 | 1.521531 |
| 3 | .142516/.091650/.103471/.099850/.091383 | .099850 | 4.536899 | 1.040104 |
| 4 | .100352/.107305/.125404/.113046/.160890 | .113046 | 4.007302 | .883172 |

有効GB/sの分子は重みを1回読む容量であり、M倍にはしない。
初期の動的行数版やビルド/他のCPU試験と重なった測定では約.8〜1秒、別runでは.05〜.3秒と
大きく変動した。最終行数固定版ではscale関数をinlineし、内側loopでのscale関数呼び出しを除く。
この揺れがあるため、第8段階の記録（.041322秒）との回帰/改善の断定はしない。
本表は固定した手順で得た今回の値であり、GLM本体の高速化を示すものではない。

`cargo run -p hayate-cli --features cuda --offline -- run glm53 --help` も成功し、
新しい spec/validate/profile/output-ids オプションの表示を確認した。
