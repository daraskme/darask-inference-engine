# 第10段階: CPU 待機と GPU キャッシュ追従

**現行の既定値・比較手順は末尾の「差し戻し対応」を参照。** 以下の窓8・32交換・
非同期保守は差し戻し前の履歴。第11段階のMTP実装を含むworktreeで修正した。

GPU はサンドボックスから利用できない。10 tok/s、通常 decode の CPU 15 GB/s、
GPU 命中率94%、45層 KLD/top-1 の維持は未確認。実装と CPU 検証を行った。
MTP は既定で無効のまま。重み形式、SIMD 内積、BF16 丸め、GPU 演算順は変更しない。

## 変更ファイル

- `crates/cli/src/glm53.rs`
- `crates/kernels/src/cpu/glm53.rs`
- `crates/models/src/expert_cache.rs`（CPU の追従試験のみ）
- `crates/models/src/glm53/{cpu_experts.rs,executor.rs,experts.rs,gpu_tests.rs,README.md,STAGE10.md}`
- `python-tools/hayate_tools/glm53_profile.py`
- `python-tools/tests/test_glm53_decode.py`

外部依存・ネットワーク・commit なし。担当外の共通 runtime、CPU pool、offload、
CUDA カーネル、量子化器、third_party は変更しない。

## 原因の切り分けと判断

第8段階の pool は worker を P コア0,2,…,14へ固定しており、固定失敗も伝播する。
一方、呼出しスレッドの固定はなく、8 worker は仕事がない間も常時 spin、
呼出し側も各行列の完了まで spin していた。呼出し側が worker と同じ P コアに
スケジュールされれば、行列の3回の barrier ごとに両者が CPU 時間を取り合う。
これはコードから確認できる競合要因。実際の Graph 実行時の2.5倍差の原因の割合は
GPU がないため確定できない。Graph 自体の CUDA driver 内の待機方式は変更していない。

- `Executor::load` の呼出しスレッドを E コア23へ固定。計算の8 P コアと保守のE20を避ける。
  推論中もこのスレッドで呼ぶ。固定エラーは返す。
- worker の idle と呼出し側の completion は既定で `park/unpark`。
  release/acquire による既存のジョブ公開と回収を維持し、全 worker の完了まで借用を保持する。
  park の前に公開された job / stop を再確認するので、先行する unpark と spurious wake を扱える。
  worker は呼出しスレッドのハンドルを clone してから完了を公開する。
- `--cpu-spin` は旧待機との比較用。`--cpu-threads 1..8` は P コア列の先頭から指定本数。
  4/6/8本の CPU ベンチでは8本が最速だったため、既定は8本を維持する。
- 通常実行でも `decode_cpu_worker_ms` と `decode_cpu_dispatch_ms` を追加。
  前者は行列ごとの最長 worker wall の和（preemption を含む）。後者は公開から全完了まで。
  差は起床・dispatch・呼出し側の待機／スケジューリングの寄与で、純粋な futex 時間ではない。
  `decode_cpu_ms` は非線形演算と入出力の BF16 変換も含む FFN 全体。
- `decode_maintenance_bytes` と `decode_maintenance_worker_ms` も通常実行で記録。
  bytes は投入量、worker ms は完了した交換の wall、終了 drain も含む。
  prefill から持ち越した交換の完了が decode に含まれる場合がある。
  `--maintenance-interval 0` との比較で帯域競合を調べる。
- profile 集計器は CPU 件数・bytes・wall・worker・dispatch を取り込み、
  CPU GB/s を合計bytes÷合計CPU wallで計算する。CPU と GPU の重複時間は token wall に足さない。

層内の GPU cache hits と shared expert の起動は既に `compute_cpu` より先。
CPU 計算はこれらと同時に走り、GPU stream 上の CPU 出力 H2D の後で既存順序の reduce を行う。
この重なりを維持した。別のジョブ抽象化や計算の近似は追加しない。
実際にどちらが合流を待つかは GPU の計測が必要。

## 非同期昇格

既定の直近窓を32→8、交換上限を8→32に変更。
`--maintenance-window 1..128 --maintenance-swaps 1..64` で個別比較できる。
`--maintenance-window 32 --maintenance-swaps 8` は NVFP4 の第8段階と同じ制限。
共通 cache の方針や Qwen は変えず、GLM の呼出し設定だけを変える。

32 token の古い利用を守り続ける遅れと、1 token 336 routes に対して最大8件しか
追従できない制約を切り分ける。候補と被退避のスコアの厳密な大小比較は維持する。
交換は独立した E コア worker / stream で行い、完了後の token 境界でのみ公開。
転送中も旧 GPU／RAM 画像を使い、デコードは worker の完了を待たない。

spare 数を上限に揃える。NVFP4 の32件で最大906,018,816 bytesのH2D+D2H。
第8段階比で VRAM と pinned RAM がそれぞれ339,757,056 bytes増える。
GPU cache slot 数の計画は変えず、既存8 GiB driver/temporary margin の一部を使う。
`memory.experts_gpu` と placement の scratch/staging bytes に増分を計上する。
転送競合や実際の peak は GPU で確認し、8/16/32交換を同じ入力で比較する。

CPU の合成 route shift 試験は128 experts / 64 slots、64 tokens。
32 tokens ごとに hot set が完全に替わり、前の token の準備済み plan を次の境界で公開する。
後半2048 routes の hit は窓32・8交換:736、窓8・8交換:1504、窓8・32交換:1696。
各 plan の上限と重複なしも検証。これは追従の正しさの試験で、実モデルの94%を証明しない。

## CPU 計測

i9-12900KS、実 HYT layer3 の32 expert、453,009,408 bytes > LLC。
読み込みを計測外とし、2 sweep warmup + 5 sweep を測定。gate/up、SwiGLU、down、
pool dispatch／回収を含む。各画像を1回読むとして数える有効 GB/s。
外部 worktree の並列ジョブは制御していないため、異なる時刻の値には負荷の差がある。

最初の4/6/8比較（5 sweep の中央値、ホストE23）:

| P workers | park GB/s | spin GB/s |
|---|---:|---:|
| 4 | 8.437 | 9.340 |
| 6 | 11.485 | 12.779 |
| 8 | 14.626 | 15.490 |

別の時刻の8本の再実行は park 2.374、spin 8.624 GB/sまで低下した。
このため park が常に最速であることや、実機の CPU 15 GB/s達成は主張しない。
最後に8本でE23/P0 × park/spinを順番に3反復し、各実行2 warmup + 5 sweep。
計15 sweep の中央値は次のとおり。worker/dispatchは32 experts分の時間。

| ホスト | 待機 | GB/s | ms/expert | worker ms | dispatch ms |
|---|---|---:|---:|---:|---:|
| E23 | park | 9.694 | 1.460 | 39.681 | 46.292 |
| E23 | spin | 11.334 | 1.249 | 35.903 | 39.586 |
| P0（workerと同じ） | park | 12.485 | 1.134 | 32.546 | 36.028 |
| P0（workerと同じ） | spin | **0.787** | **17.999** | 43.883 | **575.606** |

P0でspinするとworker wallに対してdispatch wallだけが極端に増える現象を3反復とも再現した。
E23へ移すかparkにするとこの競合を回避する。Graph時のホストのCPU移動が実際にこの状態を
作ったかは未確認。E23 parkの各実行中央値は15.322 / 7.262 / 9.694 GB/s、spinは
17.089 / 11.334 / 9.577 GB/sで、外部負荷の変動が大きい。parkの既定はGPU区間の
無用な8 worker spinを止める判断。実機ではspinとのend-to-end比較で採否を決める。

```sh
cargo test -p hayate-models --release --offline glm53_cpu_expert_bandwidth -- --ignored --nocapture
# 固定2 warmup + 5 sweep。4/6/8、park/spin、ホストP0/E23を比較。
GLM53_CPU_THREADS=8 GLM53_CPU_SPIN=1 GLM53_CPU_HOST_CORE=0 \
  cargo test -p hayate-models --release --offline glm53_cpu_expert_bandwidth -- --ignored --nocapture
```

## 実機コマンド

bash、リポジトリルート、既存 tools 環境で実行する。出力先は新規。
第8段階の同じ頻度ファイル・入力・重みを使い、MTP フラグは付けない。
各構成はwarmup2、128生成tokens、反復3。品質評価と速度評価は別実行。

```sh
export GLM53_HYT=/mnt/NVME4TB/hayate/hyt/glm53-stage6-fp8-dense
export GLM53_FREQUENCIES=/path/to/stage6/frequencies.bin
export GLM53_STAGE10_OUT=/mnt/NVME4TB/hayate/spill/glm53-stage10
mkdir "$GLM53_STAGE10_OUT"
export PYTHONPATH=python-tools
cargo build --release --features cuda --offline
cargo test -p hayate-kernels --release --features cuda glm53_cpu_nvfp4_gemv_matches_gpu -- --ignored --nocapture
cargo test -p hayate-models --release --features cuda glm53_cpu_expert_matches_gpu -- --ignored --nocapture
cargo test -p hayate-models --release --features cuda glm53_async_expert_transfers -- --ignored --nocapture
cargo test -p hayate-models --release --features cuda glm53_decode_stage6_graph -- --ignored --nocapture
cargo test -p hayate-models --release --features cuda glm53_decode_cpu_graph_matches_eager -- --ignored --nocapture

# 同じ構成で通常Graph / eager / profileを比較。
for mode in graph eager profile; do
  extra=()
  case "$mode" in eager) extra=(--no-graphs);; profile) extra=(--profile);; esac
  for repeat in 1 2 3; do
    target/release/hayate run glm53 --layers 45 --context 4096 --hyt-dir "$GLM53_HYT" \
      --prompt Hello --max-tokens 128 --warmup 2 --host-gb 100 --prefill-chunk 128 \
      --frequencies "$GLM53_FREQUENCIES" --pcie-frac 0 --maintenance-interval 4 \
      --cpu-threads 8 --maintenance-window 8 --maintenance-swaps 32 "${extra[@]}" \
      >"$GLM53_STAGE10_OUT/$mode-$repeat.out" 2>"$GLM53_STAGE10_OUT/$mode-$repeat.log"
  done
done
python3 -m hayate_tools.glm53_profile "$GLM53_STAGE10_OUT/profile-1.log" \
  >"$GLM53_STAGE10_OUT/profile-1.json"
```

上の3 modeを、まず同じ窓8・交換32で `--cpu-spin` を付けて比較。
次にparkでthreads4/6、窓32・交換8、窓8・交換8/16、interval1/2/0を個別に比較する。
一度に複数の条件を変えない。interval0は配置固定の対照であり、速度採用候補ではない。
`decode_cpu_gb_s`、worker/dispatch、GPU hit、CPU件数÷decode_steps、tok/s、ITL、
maintenance bytes/worker ms、disk reads、peak RSS/VRAMを通常とprofileの両方で記録する。
通常 decode が遅くworker wallも増える場合はCPU競合／帯域を疑い、
dispatchだけ増える場合は起床・待機を疑う。profileはGraphを無効化するため速度を分ける。

品質（固定コーパス、既存 FP8 reference と比較）:

```sh
export GLM53_FP8_REFERENCE=/path/to/fp8-reference
python3 -m hayate_tools.glm53_eval --corpus-out "$GLM53_STAGE10_OUT/corpus" --tokens 512
target/release/hayate run glm53 --layers 45 --context 512 --hyt-dir "$GLM53_HYT" \
  --host-gb 100 --prefill-chunk 128 --frequencies "$GLM53_FREQUENCIES" --pcie-frac 0 \
  --cpu-threads 8 --maintenance-window 8 --maintenance-swaps 32 \
  --eval-ids "$GLM53_STAGE10_OUT/corpus/input_ids.i32.bin" --eval-prefix 64 \
  --out "$GLM53_STAGE10_OUT/eval"
python3 -m hayate_tools.glm53_eval --reference "$GLM53_FP8_REFERENCE" \
  --actual "$GLM53_STAGE10_OUT/eval" --out "$GLM53_STAGE10_OUT/quality.json" --check
```

既存 gate (KLD<=0.15/top-1>=82%) に加え、現行 KLD 0.041 / top-1 91.1%の維持を確認。
teacher forcing は従来どおり昇格を凍結する。通常Graph/eagerで同じ固定tokenを通す
新規45層CPU/Graph回帰試験（昇格凍結、実際のCPUミスあり）でもbit一致を確認する。
既存stage6 Graph試験のprefetch指定にはpcie-frac=1を明示し、第8段階の入力検証と整合させた。

## 通常実行と profile の実機値

| 状態 | 通常 CPU GB/s | profile CPU GB/s | 通常 GPU hit | profile GPU hit | 通常 tok/s | profile tok/s |
|---|---:|---:|---:|---:|---:|---:|
| 第8段階（統合担当提供） | 6.29 | 15.5 | 84.5% | 未提供 | 5.73 | 未提供 |
| 第10段階 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |

第8段階は通常52 experts/token、CPU117 ms/token、profile49.6 ms/token。
このworktreeの再測定値ではない。実機コマンドで上の未測定欄を埋める。

## 確認できなかったこと・必要な変更

GPUによるGraph遅延の厳密な原因、CPU/GPU overlapの割合、H2DとのDRAM競合、
通常/profileのCPU GB/s・命中率・tok/s・ITL、45層KLD/top-1、実peakメモリは未確認。
GPUテストは既存のignoreを維持し、今回の新規試験はCPUで動く。
担当外コードへの必須変更はない。統合担当は上のGPU試験・性能・品質を実施する。
担当外 `crates/models/src/lib.rs` の既存mod宣言順がfmtと異なるため、統合側で整理が必要。
今回のfmtで生じる担当外差分は元に戻す。
参照URLなし。指定設計資料とローカルのGLM実装のみを使い、外部コードは流用していない。

## 検証結果

既存 `/tmp/qwen38-stage4-env.sh` の Nix toolchain を使用。GPU 実行は行っていない。

| コマンド | 結果 |
|---|---|
| `cargo fmt --all` | 成功。担当外lib.rsの既存mod順差分だけ復元 |
| `cargo clippy --workspace --all-targets --offline -- -D warnings` | 成功、警告0 |
| 同 `--features cuda` | 成功、警告0 |
| `cargo test --workspace --offline` | 235 passed / 0 failed / 19 ignored、doc tests成功 |
| 同 `--features cuda` | 254 passed / 0 failed / 134 ignored、doc tests成功 |
| `PYTHONPATH=python-tools python3 -m unittest discover -s python-tools/tests -p 'test_glm53_decode.py' -v` | 6 tests成功、CUDA math の独立CPU参照9 cases最大相対L2=2.78416453e-08 |
| `cargo test -p hayate-models --offline glm53_short_window -- --nocapture` | 1成功。合成shiftのhit 736→1504→1696 / 2048 |
| CPU bench（上記host0/23 × spin0/1、3反復） | 12実行すべて1 test成功。計60 measured sweep |
| `git diff --check` | 成功 |

新規CPU試験は同期の起床／回収（1/4/8 workersとpark/spinの完全一致）、
route shiftの追従、CPU profileのbytes/wall集計の3件。
新規GPU試験1件はignore。既存stage6 GPU試験のprefetch引数も修正した。

## 差し戻し対応（保守の帯域競合）

統合担当提供の実機値: 第8段階は5.73 tok/s、ITL 164 ms、GPU hit 84.5%。
第10段階は3.2〜3.3 tok/s、ITL 301〜305 ms、GPU hit 82.5〜83.7%。
profileでは128生成tokensに対して保守115.1 GB、worker 34,135 ms。
第10段階の32交換の上限906,018,816 bytes × 127 decode steps = 115,064,389,632 bytesと
ほぼ一致する。CPU expert計算と保守コピーの重複を避け、コピー量と頻度変動への追従を制限する。

推奨の既定値は `--host-core 23`、park（`--cpu-spin`なし）、
`--maintenance-interval 1 --maintenance-window 64 --maintenance-swaps 1 --maintenance-hysteresis 4`。
速度の採否は下の実機比較で決める。5.73 tok/s超え・10 tok/sへの接近は未確認。

- 最大交換数32→1。NVFP4 plain画像は14,156,544 bytesなので、H2D+D2Hは
  最大28,313,088 bytes/token（約28.3 MB、従来の1/32）。127 decode stepsで最大3.596 GB。
  down FP8や層混在時は起動ログの `image_bytes` の2倍×交換数が上限。
  交換数の上限で制限するため、別のバイト予算フラグは追加しない。
- 直近窓8→64。候補の回数−GPU最下位の回数が4以上のときだけ交換。
  同数は常に交換せず、`--maintenance-hysteresis 0`は従来の厳密な大小比較。
  絶対回数差なのでGPU最下位が0でも1回だけの候補を昇格しない。
  窓は1..128、交換数は1..64を比較用に維持し、実機では窓32/64/128・交換1/2/4を比較できる。
- 保守workerと独立streamを維持し、head D2H後のトークン末尾で交換を投入・完了回収・公開する。
  次のトークンへ戻る前にworkerのコピーが完了するので、CPU expert計算と重ならない。
  交換なしの場合は追加のCUDA同期を行わない。待機時間はITL、`maintenance_ms`、token wallに含む。
  `--maintenance-async`で従来の次トークンとの重複を比較できる。
- `--host-core none|0..23`を追加。noneはホストのaffinityを変更せず、第8段階相当の比較用。
  23は既定のEコア。固定失敗はエラー。実行プロセスは呼出し元のaffinityを継承するので、
  noneの比較は通常の固定されていないシェルから起動する。
- spare画像は交換上限に揃うため、32→1でVRAM/pinned RAMをそれぞれ438,852,864 bytes削減。
  GPU slot予算の計算法は従来どおり。空いた分をslotへ回す先回りの変更は行わない。
- `expert_cache::plan_recent`はGLMのみが使うrolling-window計画。差の引数を追加した。
  Qwenのinterval計画・offload・共通runtime・CPU pool・演算式・量子化は変更していない。

decodeでは1token、prefillでは1chunkを窓・intervalの1区切りと数える既存仕様を維持。
warmupの配置・履歴も維持される。MTPは既定無効のまま、今回の速度比較では使わない。
`--maintenance-interval 0`は計画・コピーを停止する。今回の比較では速度採用候補として扱う。

### 実機比較コマンド

bash、リポジトリルート、既存tools環境。同じHYT・頻度ファイル・promptで各構成warmup2＋3反復。
`--max-tokens 128 --ignore-eos`で生成128 tokensを固定（prefillの最初1個＋decode127 steps）。
通常Graphの値を速度判定に使い、profileは別に記録する。統合時は実機の負荷も揃える。

```bash
export GLM53_HYT=/mnt/NVME4TB/hayate/hyt/glm53-stage6-fp8-dense
export GLM53_FREQUENCIES=/path/to/the/same/frequencies.bin
export GLM53_RECHECK_OUT=/mnt/NVME4TB/hayate/spill/glm53-stage10-recheck
mkdir "$GLM53_RECHECK_OUT"
export PYTHONPATH=python-tools
cargo build --release --features cuda --offline

# チェックポイント不要のコピー試験と、実モデルの回帰試験。
cargo test -p hayate-models --release --features cuda glm53_boundary_expert_transfers -- --ignored --nocapture
cargo test -p hayate-models --release --features cuda glm53_async_expert_transfers -- --ignored --nocapture
cargo test -p hayate-models --release --features cuda glm53_decode_cpu_graph_matches_eager -- --ignored --nocapture

for mode in graph profile; do
  mode_flags=()
  if [ "$mode" = profile ]; then mode_flags=(--profile); fi
  for repeat in 1 2 3; do
    for config in a b c d; do
      flags=(--host-core 23 --maintenance-window 64 --maintenance-swaps 1
             --maintenance-hysteresis 4 --maintenance-interval 1)
      case "$config" in
        a) flags=(--host-core none --cpu-spin --maintenance-window 32
                  --maintenance-swaps 8 --maintenance-hysteresis 0
                  --maintenance-interval 1 --maintenance-async);;
        b) ;; # E23 + park + 保守少なめ（推奨既定）
        c) flags+=(--cpu-spin);;
        d) flags+=(--maintenance-interval 0);; # E23 + park + 配置固定
      esac
      target/release/hayate run glm53 --layers 45 --context 4096 \
        --hyt-dir "$GLM53_HYT" --prompt Hello --max-tokens 128 --ignore-eos \
        --warmup 2 --host-gb 100 --prefill-chunk 128 --cpu-threads 8 --pcie-frac 0 \
        --frequencies "$GLM53_FREQUENCIES" "${flags[@]}" "${mode_flags[@]}" \
        >"$GLM53_RECHECK_OUT/$config-$mode-$repeat.out" \
        2>"$GLM53_RECHECK_OUT/$config-$mode-$repeat.log"
    done
  done
done
for config in a b c d; do
  python3 -m hayate_tools.glm53_profile "$GLM53_RECHECK_OUT/$config-profile-1.log" \
    >"$GLM53_RECHECK_OUT/$config-profile-1.json"
done
```

(a)は第8段階の待機・保守設定相当。第11段階までの他変更も含むので、
第8段階7b56b60そのものの再現には統合側の別worktreeのバイナリでも同じ入力を測る。
(b)と(c)でpark/spinを判定し、(b)と(d)で保守の損益を判定する。
必要なら(b)の窓32/128、交換2/4を1条件ずつ変更する。
記録する値はtok/s、ITL p50/p95、decode_steps、GPU hit、CPU experts/step、
CPU GB/s・worker/dispatch ms、maintenance bytes/step・worker ms、最終drain、RSS/VRAM。
profileはeagerなので通常Graphのtok/sと混ぜない。
配置変更でCPU/GPUの加算順が変わり得るため、速度採用前に上記固定コーパスの品質ゲートも実行する。

### 検証・未確認事項

既存 `/tmp/qwen38-stage4-env.sh` のNixツールを使用。ネットワーク・外部依存・commitなし。

| コマンド | 結果 |
|---|---|
| `cargo fmt --all` | 成功。担当外lib.rsの既存mod順差分のみ復元 |
| `cargo clippy --workspace --all-targets --offline -- -D warnings` | 成功、警告0 |
| 同 `--features cuda` | 成功、警告0 |
| `cargo test --workspace --offline` | 239 passed / 0 failed / 19 ignored、doc tests成功 |
| 同 `--features cuda` | 258 passed / 0 failed / 138 ignored、doc tests成功 |
| `git diff --check` | 成功 |

新規CPU試験は同頻度・頻度差1..3の交換抑制、差4の昇格、最大1交換、バイト上限、
窓64の履歴失効と安定配置を確認。新規GPU試験は `#[ignore]`、256-byte画像だけを使い、
token末尾の全交換完了・配置公開・退避画像とH2D画像の一致を確認する形にした。
既存非同期試験は比較フラグを明示して維持した。

コピー量28.3 MBは画像サイズから算出した上限であり、実機の測定値ではない。
変更後のtok/s・ITL・CPU帯域・GPU hit・コピー待ち・KLD/top-1・peakメモリは未確認。
他担当への必須コード変更はなし。統合担当には上のGPU試験・速度比較・品質確認が必要。
担当外 `models/src/lib.rs` のmod順整形は統合側で対応が必要。
変更ファイルは `crates/cli/src/glm53.rs`、`crates/models/src/expert_cache.rs`、
`crates/models/src/glm53/{executor.rs,experts.rs,README.md,STAGE10.md}`。
参照URLなし。指定設計資料とローカル実装のみを参照した。
