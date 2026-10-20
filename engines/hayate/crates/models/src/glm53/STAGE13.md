# 第13段階: デコードのホスト往復を除去

基点 b51a07f。GPU のない sandbox で実装・CPU 検証した。
**実機のビット一致、同期1回、ITL・tok/s改善は未確認。**
MTP b51a07f / prefill 567a724 の統合担当の実機検証結果も、この作業中には届いていない。

## 選択と設計

`--decode-host async` を追加した。既定の `sync` は比較・従来の実験用に維持する。
非同期経路は `pcie-frac=0`、全ミスをRAMに保持する構成に限定する。
`--allow-disk`、`--expert-prefetch`、`--cpu-spin` との併用は明示的エラー。
これらは従来の `--decode-host sync` で使える。対象外GPU/CPUへのフォールバックは追加していない。

速度優先で、既存のCPUエキスパートプールと圧縮画像の計算をそのまま使う。
CPU/GPUの重み形式、BF16丸め、エキスパートID昇順の合算を変更しない。
新規crate、再量子化、推定値によるタイルの自動選択はない。

- runtime は実ツリーの `crates/kernels/src/runtime.rs` に追加した。
  既存の同期コピーとイベントAPIは維持し、mapped pinned allocation、raw pointer、
  stream wait/write value32、スレッドごとの明示的CUDA同期カウンタを追加した。
  Driver entry point は既存のcudartから取得し、新しいリンク依存を要求しない。
- 各層に起動時確保のmailboxと288ポインタのmapped配置表を持つ。
  GPUカーネルがルーティングIDと回転済み入力をmailboxへ直接書く。
  `cuStreamWriteValue32` のメモリバリア付き通知をEコア22のスレッドが acquire で監視する。
- 監視スレッドは既存プールへ仕事を渡す。計算はPコア0,2,...,14、
  既定の待機方式はpark。監視のスピンとCPU結果待ちはEコア上で行う。
  メインスレッドの既定コア23は従来どおり。
- GPUは配置表からヒット画像だけを選んで計算する。CPUはミスのBF16結果を
  元と同じslotへ書き、release storeで世代付き完了フラグを立てる。
  GPUは同じストリーム上の `cuStreamWaitValue32(EQ)` の後でミスslotだけを読み、
  既存のreductionへ渡す。GPUヒット計算はCPU完了待ちより先に投入する。
- サンプリング読み戻しまで配置を固定する。画像のArcは両経路の完了まで保持し、
  route統計を境界で反映してから既存の保守処理へ進む。保守方針は変更していない。
  世代カウンタと同一ストリームの順序で、MTPの連続ドラフトでもmailboxの早期再利用を防ぐ。
  CPU側の通常エラー・unwindでは待機フラグも解放し、境界でErrorを返す。
- KDAの前半・MoEの既存グラフは使い、async経路のグラフ捕捉前の不要なhost syncを除いた。
  prefillは従来経路を維持する。resetやprefillへの切替では未完了ジョブを退役する。

疎アテンションの11層にも選択IDの読み戻しがあったため、ここも除去した。
GPU上で選択IDを時系列順に整列し、pinned latent KVを直接gatherする。
MTP検証ではGPU上で最大4クエリの和集合と逆引きを作る。
KV射影は既存と同じCUTLASS BF16 GEMMのkernelを使い、実行時のGPU上の和集合行数を渡す。
上限のgridを投入するが、和集合外のCTAは処理に入らず、余分なKV行を計算しない。
選択行数は文脈長から確定するので、attentionの形状にもhost readは不要。
MTPのエキスパートもGPU上でID順の和集合と行割当を作り、既存の `dot_rows<1..4>` で
同じエキスパートをまとめて計算する。CPUミスも同じ行集合で既存のforwardを呼ぶ。

## 計測

`glm_decode_sync` が `steps`、`host_syncs`、`host_syncs_per_step` を出す。
`step` と `spec_step` 内の主スレッドの明示的な stream/event/device 同期を数える。
prefill、reset、ロード、外側のlogits dump、保守ワーカー内部の同期は含まない。
CUDAドライバ内部の暗黙の待ち時間や、保守完了のchannel待ちまで数えた指標ではない。
メモリコピー・API投入・CPU待ちを含む通し時間は既存のITL/tok/sで評価する。
同期数の目標はasyncのn=0/1/2/3でステップあたり1回。実機達成は未確認。

`--profile` / `--profile-spec` に `cpu_tail_wait_ms` と `cpu_merge_ms` を追加した。
前者はGPUヒット計算の後に露出したCPU完了待ち、後者はmapped結果の取り込み時間。
既存のreduce/MoE区間にも含まれるので加算しない。CPU wall時間はGPU区間と重なる。
プロファイル時だけ疎KVの実行時行数をmapped領域へ記録し、最終読み戻し後に報告する。

比較ベンチ `hayate_tools.glm53_async_bench` は、同期版/非同期版で
配置hash・容量・全生成ID・全logitsバイトの一致を先に要求する。
速度測定はdumpなし、各プロセス内2 warmup、5反復、実行順を交互にする。
保守は0で固定し、ITL p50/p95、tok/s、同期数と速度比をJSONへ保存する。
非同期の同期数が1回/stepでなければ失敗する。
`--n 0` 単独ではMTPなし、n>=1を含めるとn=0にもMTPを予約して容量を揃える。
`--reserve-mtp` でn=0単独の予約も指定可能。
既存のキャリブレーション配置は `--frequencies FILE` で引き継げる。
実行失敗・品質不一致の場合はcompleteにならず、そのnの速度要約を公開しない。

```sh
cargo build --release -p hayate-cli --features cuda --offline
export GLM53_MTP_HYT=/mnt/NVME4TB/hayate/hyt/glm53-mtp-stage11
export GLM53_SRC=/mnt/NVME4TB/hayate/src/GLM-5.3-Flash-Uncensored-FP8

# モデル不要: mapped通知の32回再利用、疎KV GEMM、grouped FFNの実GPUテスト
cargo test -p hayate-models --features cuda --offline glm53::async_tests -- --ignored --nocapture --test-threads=1

# 同じ配置で旧n=0と新n=0/1/2/3の生成ID・BF16 logitsを照合、同期数も検査
cargo test -p hayate-models --features cuda --offline glm53_async_same_placement_decode_and_mtp_bits -- --ignored --nocapture --test-threads=1

# 通常n=0: MTP予約なし。事前に固定したfrequenciesがあれば --frequencies FILE を追加
PYTHONPATH=python-tools python3 -m hayate_tools.glm53_async_bench \
  --binary target/release/hayate --hyt-dir "$GLM53_MTP_HYT" --model-dir "$GLM53_SRC" \
  --n 0 --tokens 128 --warmup 2 --repeats 5 --out /tmp/glm53-stage13-n0

# 同容量のMTP比較。CPU MTP配置も別の出力先で --mtp-experts cpu として実行する
PYTHONPATH=python-tools python3 -m hayate_tools.glm53_async_bench \
  --binary target/release/hayate --hyt-dir "$GLM53_MTP_HYT" --model-dir "$GLM53_SRC" \
  --n 0 1 2 3 --tokens 128 --warmup 2 --repeats 5 --out /tmp/glm53-stage13-mtp

# 重なり確認: このrunの速度を非profileの速度と混同しない
nsys profile --trace=cuda,osrt --force-overwrite=true --output=/tmp/glm53-stage13-overlap \
  target/release/hayate run glm53 --hyt-dir "$GLM53_MTP_HYT" --model-dir "$GLM53_SRC" \
  --prompt 'Write a Rust sorting function.' --raw --max-tokens 128 --ignore-eos \
  --warmup 2 --maintenance-interval 0 --decode-host async --profile
```

NsightではCPUスレッド `glm-doorbell-22` / `glm-cpu-*` と、GPUのgate_up/down、
wait-valueの区間を照合する。重なりは投入順として確保したが、実機での割合は未計測。

## 検証結果

最終確認結果は以下。GPUを使う新規4テストはすべてignore。

- `cargo fmt --all`: 成功。元から整列されていなかった担当外の
  `crates/models/src/lib.rs` は差分を戻し、変更対象に含めていない。
- `cargo clippy --workspace --all-targets --offline -- -D warnings`: 成功、警告0。
- `cargo clippy --workspace --all-targets --features cuda --offline -- -D warnings`: 成功、警告0。
- `cargo test --workspace --offline`: **251 passed / 0 failed / 19 ignored**。
- `cargo test --workspace --features cuda --offline`: **270 passed / 0 failed / 158 ignored**。
  CUDAソースのsm_120aビルド・リンクを含むが、ignoreのGPUテストは実行していない。
- `PYTHONPATH=python-tools python3 -m unittest discover -s python-tools/tests -p test_glm53_async.py -v`:
  **4 passed**。CUDAと共有する整列・和集合ヘルパー24ケース、group行割当80ケース、
  ベンチのバイト一致・同期数の厳密ゲートを検査。Rustでは1,024回のpayload再利用と中止も検査。
- 同じunittestコマンドの `-p test_glm53_decode.py`: **6 passed**。
  既存CPU dense比較9ケースの最大相対L2 **2.78416453e-08**。
- 同じunittestコマンドの `-p test_glm53_mtp.py`: **8 passed**。
- `git diff --check`: 成功。

PATHには指定のツールが実際にはなく、`nix develop` はsandboxのdaemon接続制限で
起動できなかった。既存Nix storeのcargo/rustfmt/clippy/GCC/CUDA 13.4/Pythonを
PATH・CUDA_HOMEへ設定して、すべてofflineで実行した。追加取得・crate追加はない。
Pythonの既存回帰テストにはtools環境と同じ既存torch/numpy/safetensorsを使った。

性能値はまだない。ITL p50/p95、tok/s、実機の同期数、GPUとCPUの実際の重なり、
実重みでの新経路の全logits一致を確認していない。MTP用のgrouped作業領域と疎KVの
小さな整列領域を追加しており、MTPの検証領域は比較両モードで同容量を予約する。
実際のcache容量・配置hash・ピークVRAMはベンチの出力で再確認すること。

## 変更ファイル

- `crates/cli/src/glm53.rs`
- `crates/kernels/src/runtime.rs`
- `crates/kernels/src/glm53.rs`
- `crates/kernels/cu/glm53/host_runtime.cu`
- `crates/kernels/cu/glm53/async.cu`
- `crates/kernels/cu/glm53/async_sparse.cu`
- `crates/kernels/cu/glm53/async_math.cuh`
- `crates/kernels/cu/glm53/async_cpu_test.cpp`
- `crates/models/src/glm53/mod.rs`
- `crates/models/src/glm53/async_decode.rs`
- `crates/models/src/glm53/async_tests.rs`
- `crates/models/src/glm53/executor.rs`
- `crates/models/src/glm53/experts.rs`
- `crates/models/src/glm53/gpu.rs`
- `crates/models/src/glm53/verification.rs`
- `crates/models/src/glm53/speculation_gpu.rs`
- `crates/models/src/glm53/STAGE13.md`
- `python-tools/hayate_tools/glm53_async_bench.py`
- `python-tools/hayate_tools/glm53_profile.py`
- `python-tools/tests/test_glm53_async.py`

## 他担当への必要な変更・確認

担当外コードへの必須変更はない。runtimeは追加APIで、qwen38の既存呼出しは変更不要。
統合担当は上記ignoreテストと固定配置ベンチをRTX PRO 6000で実行し、
ビット一致・同期1回・ITL/throughputを確認してから既定のasync化を判断する。
計測後のREADME・性能表更新は統合側へ依頼する。git commitはしていない。

## 参照仕様

CUDA公式 [Stream Memory Operations](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__MEMOP.html)
とローカルCUDAヘッダで、mapped device pointer、EQ待機、writeのバリア、
異なるGPUストリーム間ではeventも必要なことを確認した。
既存の層間eventは維持し、CPU完了フラグでGPUストリーム間の依存を代用していない。
Strataなど外部実装のコードは参照・コピーしていない。
