# 第12c段階: プレフィル転送・借用・短入力の修正

ベースは第13段階の `35cc632`。2026-10-15。
**GPU は使用できないため、実重み KLD ≤ 0.01、層転送 ≥30 GB/s、
2,048 / 8,192 トークンで ≥1,000 tok/s は達成確認していない。**

## 調査と設計

`docs/ARCHITECTURE.md`（§1/4.3/4.4/5/10）と `docs/MODELS.md` に従い、
既存の量子化済み HYT を変更せず、転送待ち・借用量・計算の丸め差を修正した。
新規 crate、ネットワークアクセス、コミットはない。

読み取りで確認した前回の記録:

- `/mnt/NVME4TB/hayate/spill/gpuq34/t-match.log`: start=0 の後続 row=1 で KLD=0.017066321。
- `/mnt/NVME4TB/hayate/spill/glm53-prefill-stage12-smoke/batch-256-profile.log`:
  expert image=14,156,544 bytes、GPU=5,440 slots、host=94,225,956,864 bytes。
  707 slots=10,008,676,608 bytes を借用し、D2H 9,529 ms / 復元 H2D 4,084 ms。

### pinned 転送

既存の host expert は `cudaMallocHost` で確保され、元から直接
`cudaMemcpyAsync` に渡されていた。127 MB の `pinned_staging_bytes` は
読み取り用・保守用の一時領域だけを数えた値で、host expert 本体の量ではない。
起動ログに **`pinned_expert_bytes`** を追加して区別した。

- expert 全体（NVFP4 の今回の実重みでは 14.2 MB の連続領域）を直接 DMA する。
  CPU memcpy を介して集約する処理は入れていない。別 allocation の expert 同士を
  さらに連結するにはホスト配置の変更が必要で、今回その追加コピーは行わない。
- コピー stream を2本にし、二重バンクと偶数/奇数の次層先読みを割り当てた。
- 元画像の immutable Arc を prefill 終了まで保持する。bank の再利用を
  ready/used event で GPU 上で順序付けし、毎回の `ready.synchronize()` を除いた。
- ディスク経由だけが staging を使う。1画像の pinned staging を再利用する前に待つ。
  disk は既存どおり明示的な opt-in。
- profile に `host_h2d_bytes` と `disk_h2d_bytes` を追加した。
  Python report の `pinned_gb_s` / `at_least_30_gb_s` はディスクのない区間だけで算出する。
  複数 stream の区間を合算する値は、重複があれば実際の合計帯域より保守的になる。

### スロットの借用と返却

設定上限ではなく `min(入力長, chunk容量)` で workspace を確保する。
1チャンクで予測が空の previous は、空の先読みバンクに288画像ずつの容量を確保しない。
予測が既にあれば必要な和集合数、複数チャンクまたは all なら最大288を使う。
まず既存の空き VRAM を使い、cuBLAS/driver用に2 GiBを残して不足分だけ借りる。

借用する画像の host コピーが既にあれば D2H を省き、返却時も元 host コピーを保持する。
GPU にしかない画像は全ての pinned 退避先を先に確保し、2 stream に非同期 D2H を投入する。
返却も全宛先の GPU allocation を先に確保してから pinned H2D をまとめて投入し、
expert ごとのホスト同期を除いた。失敗時も2本とも退役させてからバッファを解放する。

退避が必要で独立した layer arena が空き VRAM に入るときは、その arena と ID buffer を
先に確保し、**最初の dense 層を退避と同時に実行する**。退避中の slot と host 画像には触れない。
退避完了後に借用 buffer を workspace に使い、最初のチャンクは2層目から継続する。
独立 arena が2 GiBの余裕を残して入らない場合は、退避を先に完了させる。
`stage=backup` の `dma_ms` / `first_layer_overlap_ms` で実際の CUDA 区間を確認できる。
`stage=borrow` の D2H bytes と `stage=backup` は同じコピーを表すので二重加算しない。
借用 wall は allocation と先行する最初の層の時間も含む。

### 少ないトークン

既定の previous は、入力が512未満で、推定 expert H2D bytes が入力活性化 bytes より
大きく、CPU miss 処理が有効なときに既存の `step` 経路を使う。
GPU resident hit は GPU、host miss はデコードと同じ CPU カーネルで処理し、借用を行わない。
`kind=cpu` / `glm_prefill_policy method=cpu` を出力する。
512 は今回指定された短入力の保守的な境界で、実測した最適な分岐点とは主張しない。
`--prefill-experts all` では短入力でも GPU batch を強制し、品質と速度を比較できる。
`--pcie-frac 1`、reference、legacy はこの CPU 選択を使わない。

### 品質の丸め差

NVFP4 の decode/CPU expert は、行列全体の FP32 multiplier を含めて積和する。
旧 prefill はこの重みを先に BF16 に丸め、元の NVFP4 にもう一度量子化誤差を加えていた。
今回確認した**計算上の不一致**であり、実機 KLD 0.017 の唯一の原因と確定したわけではない。
加算順序は decode も prefill も expert ID の昇順であり、原因としての当初の仮説を撤回した。

E2M1 × E4M3 の積は BF16 で正確に表せるため、行列全体の multiplier を除いた重みで
既存 CUTLASS grouped GEMM を実行し、FP32 accumulator に multiplier を掛けた後で
BF16 出力にする。FP8 の経路、保存形式、重み自体は変更しない。
FP32 の積和順序の違いは残るため、bit一致ではなく既存の KLD ≤0.01 で確認する。

CPU の全 FP4/FP8 有限スケール組み合わせのテストと、非自明な multiplier を使う独立した
FP64 参照を追加した。相対 L2 は新方式 **7.03489689e-8**、旧方式 **0.00156952193**。
これは CUDA の速度や45層の KLD の計測値ではない。
GPU ignore テストでは実カーネルを unrounded decode の1〜4行と比較する。
既存の prefix KLD テストは default start=0/1/3/127 に加え、**all 強制・start=0** も確認する。
短入力の逐次経路だけを通して batch の品質を見落とさない構成にした。

## ローカル検証

PATH にツールがなかったため、既存 `/tmp/glm53-env.sh` を bash で読み込んだ。

```sh
cargo fmt --all
cargo clippy --workspace --all-targets --offline -- -D warnings
cargo clippy --workspace --all-targets --features cuda --offline -- -D warnings
cargo test --workspace --offline
cargo test --workspace --features cuda --offline
PYTHONPATH=python-tools python3 -m unittest tests.test_glm53_prefill tests.test_glm53_eval \
  tests.test_glm53_decode tests.test_glm53_mtp tests.test_glm53_nvfp4 -v
PYTHONPATH=python-tools python3 -m hayate_tools.glm53_prefill_bench \
  --hyt /tmp/unused-glm53 --out /tmp/glm53c-stage12c-no-gpu
git diff --check
```

| 確認 | 結果 |
|---|---|
| fmt / clippy 通常・CUDA | 成功、`-D warnings` で警告0 |
| workspace 通常 | 252成功 / 0失敗 / 19 ignore、doc tests成功 |
| workspace CUDA | 271成功 / 0失敗 / 160 ignore、doc tests成功 |
| Python 5モジュール | 37件中35成功 / GPU 2 skip / 0失敗 |
| 最新の Python prefill 単独 | 7成功 / 0失敗 |
| GPUなしベンチ | CUDA unavailable の skip JSON、性能値なし |
| CPU KDA 独立数式 | 最大相対 L2=9.89216574e-7 |
| CPU NVFP4 スケール | 新7.03489689e-8 / 旧0.00156952193 |

`cargo fmt --all` が触る担当外 `models/src/lib.rs` の既存 module 順は最後に復元する。

## 実機での確認コマンド

前回と同じ重みを使い、出力先は新規ディレクトリにする。

```sh
nix develop .#tools --offline
export PYTHONPATH=python-tools
export GLM53_HYT=/mnt/NVME4TB/hayate/hyt/glm53-stage6-fp8-dense
cargo build --release --features cuda --offline
# checkpoint不要。新しいunscaled GEMMと1 GiBの直接pinned DMAも含む。
# DMAはwarmup 2回/反復5回、main PCだけ30 GB/s以上を要求する。
cargo test -p hayate-kernels --release --features cuda --offline \
  glm53_prefill_ -- --ignored --nocapture --test-threads=1
# 借用、既存hostコピーの再利用、返却、先読み画像の正しさ
cargo test -p hayate-models --release --features cuda --offline \
  glm53_async_expert_transfers -- --ignored --nocapture --test-threads=1
cargo test -p hayate-models --release --features cuda --offline \
  glm53_boundary_expert_transfers -- --ignored --nocapture --test-threads=1
# 報告された短prefixのKLD。all強制でも後続5行を確認。
cargo test -p hayate-models --release --features cuda --offline \
  glm53_prefill_matches_tokenwise_prefix_logits -- --ignored --nocapture --test-threads=1
# まず2048のみ。品質ゲート後にlegacy / previous / allを各1回比較。
python3 -m hayate_tools.glm53_prefill_bench --hyt "$GLM53_HYT" \
  --out /mnt/NVME4TB/hayate/spill/glm53-prefill-stage12c-2048-smoke \
  --lengths 2048 --warmups 0 --repeats 1 --expert-policies previous all
# 2048の速度・転送・品質に問題がなければ、本計測。
python3 -m hayate_tools.glm53_prefill_bench --hyt "$GLM53_HYT" \
  --out /mnt/NVME4TB/hayate/spill/glm53-prefill-stage12c-2048 \
  --lengths 2048 --warmups 2 --repeats 5 --expert-policies previous all
# 最後に8192。逐次参照とlegacyも走るため、遅いときはここまで進めない。
python3 -m hayate_tools.glm53_prefill_bench --hyt "$GLM53_HYT" \
  --out /mnt/NVME4TB/hayate/spill/glm53-prefill-stage12c-8192 \
  --lengths 8192 --warmups 2 --repeats 5 --expert-policies previous all
```

頻度ありの比較は全方式に同じ `--frequencies FILE` を付ける。
ベンチの引数は **`--expert-policies`**、内部で呼ぶ CLI の引数は `--prefill-experts`。
ベンチの既定長を2048だけに変更した。品質ゲート失敗時は速度を採用しない。

`report.json` の profile で層ごとの pinned GB/s と disk bytes、workspace の実D2H/H2D量、
退避/最初の層の重複を確認する。1 GiB の DMA 単独テストの成功だけでは
**1層30 GB/s** を達成した証拠にしない。tok/s は profile なしの中央値を使う。

expert / restore H2D bytes と、34 GB/s（設計資料§1の実測値）で割った参考時間も保存する。
旧配置の94.226 GBを全て送る場合、それだけで約2.77秒、2048トークンなら約739 tok/s相当。
34 GB/sを仮定した値であり物理的な上限の断定ではないが、コピー待ちを削っただけで
2048の1000 tok/sが保証されるわけではない。実際の選択expert数と転送量で判定する。

## 変更ファイルと統合担当への依頼

- `crates/cli/src/glm53.rs`
- `crates/kernels/cu/glm53/grouped.cu`
- `crates/kernels/cu/glm53/math.cuh`
- `crates/kernels/src/glm53.rs`
- `crates/kernels/src/glm53_prefill_tests.rs`
- `crates/models/src/glm53/executor.rs`
- `crates/models/src/glm53/experts.rs`
- `crates/models/src/glm53/gpu.rs`
- `crates/models/src/glm53/gpu_tests.rs`
- `crates/models/src/glm53/prefill.rs`
- `crates/models/src/glm53/prefill_gpu.rs`
- `crates/models/src/glm53/STAGE12.md`
- `python-tools/hayate_tools/glm53_prefill_bench.py`
- `python-tools/tests/test_glm53_prefill.py`

担当外ファイルの必要な変更はない。
統合担当には上記の実機テストと、30 GB/s・KLD ≤0.01・2048/8192の1000 tok/s、
実VRAM/RAMピーク、通常の保守ありdecode、第13段階のasync decode継続を確認してほしい。
特にCUDAイベント間の重複と借用なしの経路は実機未検証である。
速度・品質の目標未確認のため、この段階で目標達成扱いにはしない。

外部 Web 参照 URL なし。既存コードと上記ローカル資料・前回実機ログのみ参照。
`third_party/` と重みファイルは変更していない。
