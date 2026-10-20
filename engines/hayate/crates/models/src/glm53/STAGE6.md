# 第6段階: デコード計測、転送・起動削減、dense直接GEMV

実装、CPU検証、sm_120aビルド済み。GPUを利用できないため、8 tok/s / ITL <=125 ms、
変更後のKLD/top-1、Graphの実行結果は未確認。既定の重み形式は変更しない。

統合担当が提供した第5段階の基準は、45層NVFP4 plain、warmup2・頻度ファイルで
2.55〜2.57 tok/s、ITL p50 374〜383 ms / p95 554〜577 ms、prefill 1.7〜1.9 tok/s、
読込66 s。KLD 0.0355 / top-1 90.4% / NLL差 -0.014で合格。
GPU hit 93.4%、GPU+host hit 98.5%、disk 830 reads / 127 steps、VRAM peak 96.2 GB。
これらは今回のworktreeで再測定した数値ではない。

## 変更ファイル

- `crates/cli/src/glm53.rs`: 新しい計測・比較フラグのhelp。
- `crates/models/src/glm53/executor.rs`: GPU内の層間接続、head GEMV、profile出力、CLI。
- `crates/models/src/glm53/gpu.rs`: decodeの一括MoE、融合、Graph、圧縮denseロード。
- `crates/models/src/glm53/experts.rs`: 同時選択8画像のscratch、非同期H2D、昇格D2H計数。
- `crates/models/src/glm53/load.rs`: 圧縮residentの検証・アップロード。
- `crates/models/src/glm53/mod.rs`: PerTensor FP8検証、profileモジュール。
- `crates/models/src/glm53/profile.rs`: CUDAイベントとD2H待ちの計数。
- `crates/models/src/glm53/gpu_tests.rs`: legacy/eager/Graphのforced-token比較。
- `crates/models/src/glm53/README.md`、`STAGE6.md`: 第6段階への案内と本報告。
- `crates/kernels/cu/glm53/decode.cu`、`dense.cuh`: dense GEMV/復元、並列router、一括MoE、latent非同期コピー。
- `crates/kernels/cu/glm53/ops.cu`: mHC collapse+RMSNorm、KDA+gated norm。
- `crates/kernels/cu/glm53/math.cuh`、`math_cpu_test.cpp`: clipping、加算順、圧縮係数のCPU検証。
- `crates/kernels/src/glm53.rs`: FFIとcheckpoint不要のGPUテスト。
- `python-tools/hayate_tools/glm53_convert.py`: denseのグループ別形式選択。
- `python-tools/hayate_tools/glm53_profile.py`: 層種別・1トークン平均の集計。
- `python-tools/tests/test_glm53_decode.py`: CPU演算、変換、profileログの検証。

qwen38、共通expert_cache/offload、third_party、外部crateは変更していない。git commitなし。

## 設計と残る同期

元のexecutorは、embeddingと45層のhiddenを毎回D2H/H2Dしていた。
各層に固定した入出力を維持し、前のstreamの完了イベントを次のstreamで待ち、D2Dで渡す。
headだけが4-byte greedy IDを同期取得する。独立stream間のscratch再利用もこの依存で保護する。
GPUバッファとpinnedバッファを解放する前にstreamをdrainし、Graphを破棄する。

routerは従来の1 CTA内の288回逐次射影・reduceから、warp GEMV射影とGPU top-kの2起動にする。
sigmoid、correction biasは選択だけ、元scoreの正規化×2.5、同点時の小さいID優先を維持。
ルートIDはpinned RAMへ非同期コピーする。CPUのキャッシュ解決前に一度待つ。
**部分常駐の全ヒット層でも、このルート待ちは残る。ミス時だけ待つ経路は未実装。**
Qwen3.8の全expert常駐slabを、GLMの部分常駐キャッシュにそのまま適用していない。

8個のrouted画像はそれぞれ別scratch領域を使い、host hitは非同期にまとめて転送する。
disk stagingは一つなので、再利用前だけ同期する。既存の配置・頻度・昇格規則は維持する。
次トークン予測による先読みは、内訳の実測前なので追加していない。

NVFP4、block-FP8、混合down FP8を読み、sharedを9番目としてgate/up・down・reduceを実行する。
回転0ならMoE本体は3起動。回転ありならrouted入力とdown入力のHadamardが追加される。
sharedは回転しない。gate/upのBF16出力、gate最大10 / up[-10,10]、SiLU後のBF16境界、
productのBF16境界、down出力のBF16境界を維持する。
reduceは元のtop-k順からexpert ID昇順をGPUで作り、各multiplyとaddをBF16に丸める。
最後にsharedを足す。加算の再結合やatomicは使わない。

mHCはmapを作った後のcollapse+RMSNormを1起動にし、collapseのBF16境界を保持する。
KDAは1-token recurrence+gated normを1起動にし、F32 stateとBF16 coreを保持する。
疎アテンションのpool用key/gateはGPUに保存し、CPUへの往復を除く。
latent KVは引き続きpinned RAMに置き、appendを非同期D2Hする。
選択IDをダウンロードする際の同期でlatent appendも完了させ、その後CPUでgatherする。

キャッシュ昇格・logits保存を除く45層の明示的D2H待ちは、コード上、
legacy 133回（embedding1 + 層出力45 + router42 + sparse44 + head1）から
54回（router42 + sparse選択11 + head1）。新経路のD2Hコピー自体は65回で、
11回のlatent appendが選択IDの待ちに重なる。実機カウンタで確認する必要がある。
H2DやGraph初回capture、disk staging再利用の同期はこの回数に含めない。

KDAのattention+mHCと、MoE+FFN側のexpandを別々のGraphにする。
ルートID・画像ポインタのdevice配列は毎回更新し、Graphは固定アドレスの配列を読むため、
選択expertやcache slotが変わっても再captureしない。captureはwarmな起動後に一度行う。
Sparse attention、router D2H、cache解決を含む全45層Graphは未実装。
`--no-graphs`で同じ演算のeager経路、`--decode-reference`で旧演算・層間往復を選べる。
referenceは未変更形式のHYTで使う。新しい圧縮denseはreferenceでも直接GEMVを使う。

## dense形式と容量

`--dense shared|kda|sparse|dense-ffn|lm-head=bf16|fp8|nvfp4`を繰り返し指定する。
未指定は元形式を保持する。router、mHC、norm、conv、embedding、補助行列は対象外。
既存HYT v2のPerTensor E4M3 / NVFP4を使い、runtimeの量子化フラグは不要。
元block-FP8のdenseは復元してBF16に丸めてから量子化する。
routed expertは従来どおりF32復元から量子化し、このdense選択によって回転やMSE設定を変えない。

denseはBF16入力、F32積和、BF16出力のwarp GEMVで直接読む。FP8/FP4はCUDA native命令で復号する。
denseの復元係数をBF16に丸め、BF16 GEMMとの境界を揃える。
routed NVFP4は既存GEMVに合わせて係数をBF16には丸めない。
warp内の積和順は旧CUTLASS GEMM/旧block-FP8 GEMVと異なるため、モデル品質の再評価が必要。

疎アテンションの選択後KV展開だけは複数行のBF16 GEMMを保つ。
圧縮kv_bをこのGEMM直前にscratchへ復元し、圧縮重みのBF16コピーを常駐させない。
このscratchは33,554,432 bytes/疎層（45層では369,098,752 bytes）。
新しいTensor Core GEMM、activation量子化、形状のチューニングは追加していない。

実ヘッダだけから取得した45層の容量（bytes）。速度・品質の測定ではない。

| 候補 | resident payload | HYT全体 | routed payload |
|---|---:|---:|---:|
| 元形式/NVFP4 plain | 15,225,993,464 | 186,607,781,888 | 171,228,411,648 |
| shared NVFP4 | 14,763,313,904 | 186,145,359,872 | 同上 |
| shared NVFP4・他dense FP8 | 9,182,544,236 | 180,566,032,568 | 同上 |

resident payloadはembedding・metadataを含む。scratch、state、runtime imageの整列は別。
変換は元sourceから行い、新規出力先を要求する。既存のNVFP4画像を再利用する変換器は追加していない。
全source hash、全expertの再変換とファイル書き込みを伴うため、denseだけの変換時間ではない。

## profileの読み方

`--profile`はwarmupと測定prefillの後に、decodeの各層・head・cache maintenanceを記録する。
evalで指定した場合もstepだけを報告する。prefill・logits保存・重みロードは内訳の対象外。
CUDAイベントは事前確保し、層ごとに追加のGPU完了待ちは行わない。
`--profile`はGraphを無効にする。Graph再生中のイベント計測は今回追加していない。
**速度の合否はprofileなしで測る。**

- `mhc_ms`: attnのcollapse/norm、attention expand+FFN collapse/norm、FFN expand。
- `attention_ms`: KDA全体または疎アテンション全体。疎層ではlatent/選択D2HとCPU gatherの隙間も含む。
- `dense_ffn_ms`: 最初の3層のFFN。
- `routing_ms`: 射影とGPU top-k。
- `transfer_ms`: ルートD2H、CPU cache解決・read、expert H2D、pointer配列H2D。
- `gate_up_ms` / `down_ms` / `reduce_ms`: shared込み。Hadamardがある場合はgate_up区間に含める。
- **referenceの`gate_up_ms`は逐次gate/up/down/reduce/sharedの全体**。前後は3項目の合計で比較する。
- `lm_head_ms` / `greedy_ms`: head射影とgreedy。
- `maintenance_ms`: token末尾のcache昇格のホストwall時間。
- `d2h_count` / `d2h_sync_count` / `d2h_wait_ms`: コピー回数 / CPUで待つ回数 / そのホストwall時間。
- `token_wall_ms`: forward本体のwall時間。profileログ出力の時間は含めない。

CUDAイベントの区間は、前のイベント以降のホスト投入の隙間や、未完了コピーも含む。
D2H待ち時間はCUDA区間と重なるので加算しない。純粋なカーネル時間ではない。
集計器は一つの測定runを受け付け、欠けたtokenや重複したposition/layer、NaN/負値を拒否する。

## 統合担当の実機コマンド

```sh
nix develop .#tools --offline
export PYTHONPATH=python-tools
cargo build --release --features cuda --offline
cargo test --workspace --release --features cuda glm53_dense_decode_gpu -- --ignored --nocapture --test-threads=1
cargo test --workspace --release --features cuda glm53_router_decode_gpu -- --ignored --nocapture --test-threads=1
cargo test --workspace --release --features cuda glm53_moe_decode_gpu -- --ignored --nocapture --test-threads=1
cargo test --workspace --release --features cuda glm53_decode_fusions_gpu -- --ignored --nocapture --test-threads=1
GLM53_HYT=/mnt/NVME4TB/hayate/hyt/glm53-nvfp4-plain \
  cargo test --workspace --release --features cuda glm53_decode_stage6_graph -- --ignored --nocapture --test-threads=1

GLM_H=/mnt/NVME4TB/hayate/hyt
GLM_S=/mnt/NVME4TB/hayate/spill/glm53-stage6
mkdir -p "$GLM_S"
# 評価に使わないpromptで頻度収集。以後は全候補に同じ頻度を使う。
target/release/hayate run glm53 --layers 45 --hyt-dir "$GLM_H/glm53-nvfp4-plain" \
  --prompt 'Explain how to sort an array.' --max-tokens 128 --host-gb 64 --allow-disk \
  --prefill-chunk 128 --warmup 2 --save-frequencies "$GLM_S/frequencies.bin"

# 一つのbinaryで同じHYTを比較。bashの配列を使用する。
for mode in legacy eager graphs; do
  GLM_FLAGS=()
  case "$mode" in
    legacy) GLM_FLAGS=(--decode-reference) ;;
    eager) GLM_FLAGS=(--no-graphs) ;;
  esac
  for repeat in 1 2 3; do
    target/release/hayate run glm53 --layers 45 --context 4096 \
      --hyt-dir "$GLM_H/glm53-nvfp4-plain" --prompt Hello --max-tokens 128 \
      --host-gb 64 --allow-disk --prefill-chunk 128 --warmup 2 \
      --frequencies "$GLM_S/frequencies.bin" "${GLM_FLAGS[@]}" \
      2>"$GLM_S/$mode-$repeat.bench"
  done
  target/release/hayate run glm53 --layers 45 --context 4096 \
    --hyt-dir "$GLM_H/glm53-nvfp4-plain" --prompt Hello --max-tokens 128 \
    --host-gb 64 --allow-disk --prefill-chunk 128 --warmup 2 \
    --frequencies "$GLM_S/frequencies.bin" "${GLM_FLAGS[@]}" --profile \
    2>"$GLM_S/$mode.profile"
  python3 -m hayate_tools.glm53_profile "$GLM_S/$mode.profile" \
    >"$GLM_S/$mode-breakdown.json"
done
# graphs.profileもGraph無効。eagerと同じ演算の計測である。

python3 -m hayate_tools.glm53_convert --layers 45 --expert-quant nvfp4 \
  --device cuda --dense shared=nvfp4 --out "$GLM_H/glm53-stage6-shared" --estimate
python3 -m hayate_tools.glm53_convert --layers 45 --expert-quant nvfp4 \
  --device cuda --dense shared=nvfp4 --out "$GLM_H/glm53-stage6-shared"
python3 -m hayate_tools.glm53_convert --layers 45 --expert-quant nvfp4 --device cuda \
  --dense shared=nvfp4 --dense kda=fp8 --dense sparse=fp8 --dense dense-ffn=fp8 \
  --dense lm-head=fp8 --out "$GLM_H/glm53-stage6-fp8-dense"

python3 -m hayate_tools.glm53_eval --corpus-out "$GLM_S/corpus" --tokens 512
GLM_IDS="$GLM_S/corpus/input_ids.i32.bin"
# 同条件の既存FP8 referenceがあれば再利用できる。
target/release/hayate run glm53 --layers 45 --context 512 --hyt-dir "$GLM_H/glm53-fp8" \
  --allow-disk --host-gb 64 --prefill-chunk 128 --decode-reference \
  --eval-ids "$GLM_IDS" --eval-prefix 64 --out "$GLM_S/fp8-reference"
for variant in glm53-nvfp4-plain glm53-stage6-shared glm53-stage6-fp8-dense; do
  target/release/hayate run glm53 --layers 45 --context 512 --hyt-dir "$GLM_H/$variant" \
    --allow-disk --host-gb 64 --prefill-chunk 128 \
    --eval-ids "$GLM_IDS" --eval-prefix 64 --out "$GLM_S/$variant-eval"
  python3 -m hayate_tools.glm53_eval --reference "$GLM_S/fp8-reference" \
    --actual "$GLM_S/$variant-eval" --out "$GLM_S/$variant-quality.json" --check
done
# KLD <=0.15 / top-1 >=82%を通った候補だけ、上のbench/profileのHYTを差し替えて測る。
```

実測表にはlegacy/eager/Graph、plain/shared/FP8-denseのそれぞれで、KDA・疎attention・
dense FFN・routing・transfer・MoE合計・mHC・lm_head・D2H count/wait、tok/s、ITL p50/p95、
KLD/top-1/NLL、実decode_steps、cache hit/read、prefill速度、cold load、peak VRAMを記録する。
前後の時間・tok/s・品質の欄は、この環境では未測定。

## このworktreeの検証

ローカルPATHにはcargo/nvcc/python3がなかったため、既存の
`source /tmp/qwen38-stage4-env.sh`でNix toolchainとinclude/library設定を読み込んだ。
最初の簡易環境ではCUDA headerのinclude設定がなくビルドに失敗し、上記環境で解消した。
コード追加中のfmt引数・unused-mutのエラーは修正して再実行した。

- `cargo fmt --all`: 成功。担当外`crates/models/src/lib.rs`の既存mod順の差分だけ復元する。
- `cargo clippy --workspace --all-targets --offline -- -D warnings`: 成功、警告ゼロ。
- 同 `--features cuda`: 成功、警告ゼロ、sm_120aコンパイル・リンク成功。
- `cargo test --workspace --offline`: 211成功 / 17 ignore。
- 同 `--features cuda`: 230成功 / 123 ignore。
- `PYTHONPATH=python-tools python3 -m unittest tests.test_glm53_decode tests.test_glm53_eval tests.test_glm53_nvfp4 -v`:
  19件、17成功 / GPU2 skip。新しい4件ではtorch/numpy/safetensorsと標準ライブラリのみを使用。
- dense CPU検証: 9ケース、N=7/129/33、K=16/80/256、partial scale tileと元block-FP8を含む。
  独立torch復元+F64 matvecに対する圧縮denseの最大相対L2 **2.78416453e-08**。
  これは圧縮レイアウトの読出し精度で、元モデルのKLDではない。
- 実sourceヘッダによる`convert(... estimate=True ...)`: 上の3構成の容量を取得、元ファイルは読取りのみ。
- `nvcc -c -O3 -std=c++20 --use_fast_math -gencode=arch=compute_120a,code=sm_120a -Xptxas=-v crates/kernels/cu/glm53/decode.cu -o /tmp/glm53-stage6-decode.o`:
  成功。dense/gate_up/reduce各34、down39、router射影37 / 選択20 registers、全kernelでspill 0。
  選択のshared memoryは2304 bytes。これはコンパイル資源量で速度・占有率の実測ではない。
- `git diff --check`: 成功。外部crate追加、Web参照、git commitなし。

初回の既存`test_glm53*.py`全体のdiscoverは、既存oracleがtransformersを使うことに気付かず
実行した。今回の依存制約に合わないため検証手順から除き、上記3モジュールで再確認した。
新実装にはtransformersへの依存を加えていない。

## 確認できなかったこと・他担当への必要な変更

GPUのnative復号、融合のbit比較、checkpointを使うGraph再生、KLD/top-1/NLL、greedy、
GPU内訳、tok/s >=8 / ITL <=125 ms、変換時間、prefill速度、VRAM peakは未確認。
ignore/skipのGPUテストをCPUで実行済みとは扱わない。

担当外の必須コード変更はない。統合担当には上記GPUテスト・品質・速度・内訳の実測が必要。
`models/src/lib.rs`のmod順は統合時にfmtと合わせる必要がある。
部分常駐時の全ヒット層でルート待ちをなくす処理、全45層Graph、次トークンのexpert先読みは
未実装で、内訳取得後の残作業。今回の局所Graphだけで全ヒットのtokenをGraph化したとは主張しない。

参照は指定された設計文書、既存GLM実装と
`qwen38/{resident.rs,gpu.rs,STAGE12.md}`、`cu/qwen38/{dense.cu,dense.cuh,nvfp4.cu,decode.cu}`のみ。
Web参照URLはなし。H3の重み・参考実装は今回のGLM作業では使用していない。
