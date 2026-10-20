# 第7段階: エキスパート転送とキャッシュ保守

差し戻し後の現行仕様・実測・コマンドは末尾の「差し戻し対応」を参照。
以下の当初仕様（既定32、交換中の画像待ち）は履歴であり、現行コードには適用しない。

CPU検証とCUDAビルドは完了。GPUが使えないため、変更後の速度・品質は未測定。
8 tok/s / ITL <=125 msを達成したとは主張しない。

## 変更ファイル

- `crates/cli/src/glm53.rs`
- `crates/models/src/expert_cache.rs`
- `crates/models/src/glm53/{executor.rs,experts.rs,gpu.rs,gpu_tests.rs,mod.rs,transfer.rs}`
- `crates/models/src/glm53/{README.md,STAGE7.md}`
- `python-tools/hayate_tools/glm53_profile.py`
- `python-tools/tests/test_glm53_decode.py`

qwen38、offload、共通CUDA runtime、CUDAカーネル、量子化、third_party、依存crateは変更しない。
git commitなし。fmtで担当外`models/src/lib.rs`に生じる既存mod順の差分は復元した。

## 実装と判断

### 保守

既定の`--maintenance-interval 32`では、32回の`finish_token`ごとに最大8件を計画する。
decodeは1トークンにつき1回、prefillは1チャンクにつき1回の区切りで数える。
ウォームアップも区切りに含まれる。0は配置固定、1は各区切りで非同期交換。
頻度ファイルありの昇格には候補スコア > 被退避スコア×1.25を要求する。
頻度ファイルなしは従来と同じ厳密な大小比較。直近区間で使われたresidentは退避しない。
キャッシュ全体の走査・ソートを毎トークン行わない。

Eコア20の専用スレッド・独立ストリームで交換する。RAMからの昇格は
GPU→pinned stagingのD2H、pinned RAM→GPUのH2Dを同じストリームへ投入し、
完了後に退避画像を昇格元のpinnedバッファへ移す。RAM画像数を増やさない。
ディスクからの昇格では元GPU画像はディスクtierへ戻し、元HYTから再読込可能なままにする。

GPUスロットと昇格元のRAM画像は所有権ごとworkerへ渡す。キャッシュの新しい対応を
予約し、完了通知を受けるまでGPUポインタや退避RAM画像を公開しない。
その画像が実際に選択された場合だけホストが完了を待つ。他層の計算は進められる。
保守が未完了の間は次の計画を作らない。worker失敗時は共通の失敗状態を設定し、
後続コピー・層への投入を止め、待機ジョブには失敗と所有バッファを返す。
持続CUDAエラーは`Error::CudaAsync`として伝播する。greedy/headのD2H完了後に交換を開始し、
先行トークンが使用中のスロットを書き換えない。resetと終了時にはworkerをdrainする。

### 転送

1層の実top-kのGPUミスをまとめて解決する。ホスト画像は専用コピーストリームから
pinned H2Dを非同期投入し、計算ストリームはイベントで待つ。
ディスク画像はEコア16〜19の常駐4リーダーへまとめて依頼する。
各リーダーは独立ストリームと事前確保したpinned stagingを持ち、O_DIRECTの大きな
positional read、画像のpacking、非同期H2Dを行う。CPUは層のバッチ完了をまとめて回収する。
従来の「disk missのたびに計算streamを同期して単一stagingを再利用」を除いた。
4リーダーは各自のstaging再利用前に自身のコピーストリームを待つ。

HYT entryは4096-byte境界から、最大約8 MiBを整列して読む。
最終entryが整列ブロックより短い場合もpayloadが揃えば受け付け、途中の短読込はエラー。
NVFP4、元block-FP8、down FP8のスケールと256-byte runtime paddingを保持する。
形式の再変換、別量子化、演算変更はしない。

転送scratchは8画像×2bank。前の層の計算中に次のbankを準備できる。
古いbankのGPU読取りは、現在層のルート取得時の依存・同期で完了している。
実ルートと先読みを照合し、命中したscratchをすべて予約してから外れのscratchを割り当てる。
先読みが外れても実ルートIDの画像を渡す。ホストコピーはイベントで順序を保証し、
ディスク画像はworkerのH2D完了通知後だけ利用する。エラー時もコピーストリームをdrainする。

NVFP4 plainではscratchは従来8画像の113,252,352 bytesから、legacy/prefill用1画像を含む
17画像の240,661,248 bytesへ増える（+127,408,896 bytes）。
定常pinned stagingは6画像と5個の直接I/Oバッファ、計126,923,259 bytes。
計画RAM tierの64 GBとは別。GPU scratchは既存8 GiBのdriver/workspace予約内で確保し、
`gpu_bytes`とpeak VRAMに含める。KVや計算workspaceを他streamと共有しない。

### 1層先読み

`--expert-prefetch`で有効。現在層のFFN正規化済み活性を入力として次層のrouterを
独立streamで実行する（次層のattention/mHCを通す前の近似）。
現在層のMoEと重ねて予測IDを取得し、次層のhost/diskミスを先に転送する。
次層で実際のrouterを必ず実行し、そのID・重みで計算する。近似IDで出力を決めない。
追加router D2Hは最大41回/トークン。予測が外れると余分なディスク読込や待ちを増やすため、
**効果の実測前には既定有効にしない**。命中件数と全転送量で採否を判断する。

既定経路は全ヒット層で追加のexpert転送待ちを行わない。
ただし**実ルートIDのD2H待ち42回/トークンは残る**。
疎attentionの11回とheadの1回を含む定常の計算側D2H待ちは54回。
保守D2Hはworker側へ移す。全ヒット層のルート待ちを除くにはGPU側のresident lookupと
ホストへのmiss判定が必要で、miss層を再実行する仕組みは今回入れていない。

## 計測の前後

既存の`/mnt/NVME4TB/hayate/spill/gpuq19/glm-prof.err`を新しい集計器でも再集計した。
127 decode steps、45層、NVFP4 plain、warmup2、頻度ファイル、eager。
第6段階のログを読み取った数値で、このworktreeのGPU再測定ではない。

| 項目 | 第6段階 | 第7段階 |
|---|---:|---|
| transfer | 129.704 ms/token | 未測定 |
| maintenance | 83.287 ms/token | 未測定（ホスト投入時間とworker時間を分離） |
| attention | 25.878 ms/token | 未測定 |
| mHC | 16.208 ms/token | 未測定 |
| routing | 2.209 ms/token | 未測定 |
| gate_up + down + reduce | 7.237 ms/token | 未測定 |
| D2H copies / 同期待ち | 71.520 / 60.520 per token | 未測定 |
| D2H待ちwall（他と重なる） | 153.120 ms/token | 未測定 |
| forward wall（profile有効） | 266.789 ms/token | 未測定 |
| profileなしdecode / ITL p50 / p95（統合担当提供） | 3.81 tok/s / 258 ms / 399 ms | 未測定 |
| 品質（統合担当提供、dense FP8構成） | KLD 0.041 / top-1 91.1% | 未測定 |

plainとdense FP8を同一構成の比較として混ぜない。
既存`glm-bench-fp8dense.err`は別構成で4.646 tok/s / 215.135 ms / 297.259 msを記録している。
新規比較は同じHYT・頻度・prompt・warmup・token数で行う。

`--profile`のmaintenance行に以下を追加した。

- `host_h2d`, `disk_h2d`, `h2d_bytes`: 実・先読みを含む転送投入件数とpadded画像bytes。
- `disk_bytes`: 完了したディスク読込のpayload bytes。4096-byte整列の読込paddingは含まない。
- `read_wait_ms`: ホストがreader結果を回収する待ち。`reader_ms`: 並列readerのI/O+packing+H2D合計wall。
- `predicted`, `prefetch_hits`: ミス画像の先読み投入数と実ルートで使った数。
- `maintenance_h2d`, `maintenance_d2h`, `maintenance_disk`, `maintenance_bytes`: 交換投入件数とH2D+D2H bytes。
- `maintenance_worker_ms`: 完了したRAM交換workerのD2H+H2D待ちwall。
- `maintenance_wait_ms`: 使用する画像が交換中だったためデコード側で待ったwall。

worker完了のbytes/時間は通知を回収したトークンへ帰属する。
保守D2Hを計算側の同期回数に加算しない。worker時間は並列・CUDA区間と重なるので加算しない。
集計JSONの既存`ms_per_token`に件数・bytesのキーも含む（旧ログは0）。
未完了の保守を最後にdrainし、`expert_final_drain_ms`を別途報告する。
これはITLへ加算しない。速度判定はprofileなしの測定で行う。

## 統合担当の実機コマンド

リポジトリルートから実行。頻度は第6段階で使った同じファイルを指定する。
既存ファイルを上書きしない新しい出力ディレクトリを使う。

```sh
nix develop .#tools --offline
export PYTHONPATH=python-tools
export GLM53_HYT=/mnt/NVME4TB/hayate/hyt/glm53-nvfp4
export GLM53_FREQUENCIES=/path/to/the/stage6/frequencies.bin
export GLM7_OUT=/mnt/NVME4TB/hayate/spill/glm53-stage7
mkdir -p "$GLM7_OUT"
cargo build --release --features cuda --offline
cargo test --workspace --release --features cuda glm53_async_expert_transfers -- --ignored --nocapture --test-threads=1
cargo test --workspace --release --features cuda glm53_parallel_disk_transfers -- --ignored --nocapture --test-threads=1
cargo test --workspace --release --features cuda glm53_decode_stage6_graph -- --ignored --nocapture --test-threads=1

# bash。最初は全てeagerで条件をそろえ、採用構成だけGraphも測る。
for mode in batch frozen prefetch; do
  GLM7_FLAGS=(--maintenance-interval 32)
  case "$mode" in
    frozen) GLM7_FLAGS=(--maintenance-interval 0) ;;
    prefetch) GLM7_FLAGS+=(--expert-prefetch) ;;
  esac
  for repeat in 1 2 3; do
    target/release/hayate run glm53 --layers 45 --context 4096 \
      --hyt-dir "$GLM53_HYT" --prompt Hello --max-tokens 128 \
      --host-gb 64 --allow-disk --prefill-chunk 128 --warmup 2 \
      --frequencies "$GLM53_FREQUENCIES" --no-graphs "${GLM7_FLAGS[@]}" \
      >"$GLM7_OUT/$mode-$repeat.out" 2>"$GLM7_OUT/$mode-$repeat.bench"
  done
  target/release/hayate run glm53 --layers 45 --context 4096 \
    --hyt-dir "$GLM53_HYT" --prompt Hello --max-tokens 128 \
    --host-gb 64 --allow-disk --prefill-chunk 128 --warmup 2 \
    --frequencies "$GLM53_FREQUENCIES" --profile "${GLM7_FLAGS[@]}" \
    >"$GLM7_OUT/$mode-profile.out" 2>"$GLM7_OUT/$mode.profile"
  python3 -m hayate_tools.glm53_profile "$GLM7_OUT/$mode.profile" \
    >"$GLM7_OUT/$mode-breakdown.json"
done
# 第6段階の同条件profileを同じ集計器で再集計。
python3 -m hayate_tools.glm53_profile /mnt/NVME4TB/hayate/spill/gpuq19/glm-prof.err \
  >"$GLM7_OUT/stage6-breakdown.json"
# 元ログはeagerなので、まず上のeagerと比較。dense FP8はHYTを差し替えて別途同じ測定を行う。

python3 -m hayate_tools.glm53_eval --corpus-out "$GLM7_OUT/corpus" --tokens 512
GLM7_IDS="$GLM7_OUT/corpus/input_ids.i32.bin"
target/release/hayate run glm53 --layers 45 --context 512 \
  --hyt-dir /mnt/NVME4TB/hayate/hyt/glm53-fp8 --host-gb 64 --allow-disk \
  --prefill-chunk 128 --decode-reference --maintenance-interval 0 \
  --eval-ids "$GLM7_IDS" --eval-prefix 64 --out "$GLM7_OUT/fp8-reference"
for mode in batch prefetch; do
  GLM7_FLAGS=()
  if [ "$mode" = prefetch ]; then GLM7_FLAGS=(--expert-prefetch); fi
  target/release/hayate run glm53 --layers 45 --context 512 \
    --hyt-dir "$GLM53_HYT" --host-gb 64 --allow-disk --prefill-chunk 128 \
    --maintenance-interval 32 "${GLM7_FLAGS[@]}" \
    --eval-ids "$GLM7_IDS" --eval-prefix 64 --out "$GLM7_OUT/$mode-eval"
  python3 -m hayate_tools.glm53_eval --reference "$GLM7_OUT/fp8-reference" \
    --actual "$GLM7_OUT/$mode-eval" --out "$GLM7_OUT/$mode-quality.json" --check
done
# 同一token列を入力したbatch/prefetchのlogitsはbytes一致を要求する。
cmp "$GLM7_OUT/batch-eval/eval_logits.bf16.bin" "$GLM7_OUT/prefetch-eval/eval_logits.bf16.bin"
```

新しい先読みは重みの選択以外に使わないため、同じHYTのgreedy列・teacher forcing logitsの一致を確認する。
KLD <=0.15 / top-1 >=82%の既存ゲートを維持し、dense FP8構成では提供値0.041 / 91.1%の再現も確認する。
8 tok/s / ITL <=125 msだけでなく、実decode_steps、disk reads、先読みの無駄、GPU/host hit、
peak VRAM、final drain、prefill/load時間を記録する。profileのD2H待ちとCUDA時間は加算しない。

## このworktreeの検証

PATHにcargo/nvcc/python3がなかったため、既存`/tmp/qwen38-stage4-env.sh`のNix toolchainを使用した。
外部crateを追加せず、cargoは全てoffline。

- `cargo fmt --all`: 成功。担当外の既存mod順だけ復元。
- `cargo clippy --workspace --all-targets --offline -- -D warnings`: 成功、警告ゼロ。
- 同`--features cuda`: 成功、警告ゼロ。
- `cargo test --workspace --offline`: 220 passed / 17 ignored。
- 同`--features cuda`: 239 passed / 127 ignored。sm_120aの既存CUDAカーネルとFFIのリンクを確認。
- `PYTHONPATH=python-tools python3 -m unittest tests.test_glm53_decode tests.test_glm53_eval tests.test_glm53_nvfp4 -v`:
  20件、18 passed / CUDA2 skipped。
- 新しいCPU試験は、O_DIRECTのNVFP4/FP8/F32 packingを1100 payload bytesで確認、
  末尾短読込、truncation、alignment、dirty padding、先読み外れのscratch予約順、保守間隔とヒステリシス、worker失敗の伝播・持続CUDAエラーの保持を検証。
- 既存dense CPU 9ケースの最大相対L2は2.78416453e-08。この値は転送速度やモデル品質の測定ではない。
- 新集計器で既存第6段階profile 127トークンを再集計: 上の前後表の値を確認。
- `git diff --check`: 成功。

既存Graph回帰テストがeager後もGraphを無効にしたままだった点を修正し、
Graphを実行してからeagerのgreedy/logitsと比較する順にした。先読みも有効にして検証する。
途中のCUDA clippyのwhile-let指摘とworker停止処理編集中の構文エラーは修正し、全checkを再実行した。

## 確認できなかったこと・他担当への必要な変更

GPUでの並列コピーと所有権交換、checkpointでのgreedy/Graph/先読みbit一致、KLD/top-1、
変更後の内訳・tok/s・ITL、peak VRAM、NVMe/PCIeの実効帯域は実機確認が必要。
ignored GPUテストを成功済みとは扱わない。

担当外の必須コード変更はない。統合担当は上の実機テスト・品質・速度を実行する。
`models/src/lib.rs`の既存mod順は統合時にfmtへ合わせる必要がある。
ルートIDの42回のD2H待ちを除く処理は残作業。実機で今回のtransfer/maintenance改善を測った後、
GPU resident lookupとmissフラグの扱いを判断する。

参照: docs/ARCHITECTURE.md（§1、§4.3、§4.4、§5、§10）、docs/MODELS.md、
既存GLMコード、`crates/models/src/qwen38/resident_load.rs`（読取りのみ）。
Web参照URLなし。H3重み・参照実装は今回使用していない。


## 差し戻し対応（2026-10-15）

目的は第6段階4.65 tok/sの回復、次に6 tok/s。GPUを使えないため、達成は未確認。
統合担当からの同一HYT `glm53-stage6-fp8-dense`、45層、128生成tokens、warmup2、
同じ頻度ファイルでの結果を基準とする。

| 構成 | tok/s | ITL p50/p95 ms | GPU hit | GPU+host hit | disk reads |
|---|---:|---:|---:|---:|---:|
| 第6段階（同期・毎token） | 4.65 | 215 / 297 | 94.3% | 未提供 | 648 |
| 当初第7段階 Graph・間隔32 | 2.88 | 339 / 458 | 82.4% | 96.2% | 1,648 |
| 当初第7段階 eager・間隔32 | 2.90 | 339 / 463 | 82.6% | 96.2% | 1,652 |
| 当初第7段階 frozen | 2.31 | 430 / 560 | 75.8% | 94.8% | 2,234 |
| 当初第7段階 prefetch | 2.22 | 445 / 615 | 82.4% | 96.2% | 2,408 |
| 差し戻し後（毎token・準備後に公開） | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |

| profile内訳 ms/token | 第6段階（提供値） | 当初第7段階（提供値） | 差し戻し後 |
|---|---:|---:|---|
| transfer | 130 | 301 | 未測定 |
| maintenance（decode側） | 83 | 0.006 | 未測定 |
| read_wait（transferと重複） | 未提供 | 138.5 | 未測定 |
| reader（並列worker時間の和） | 未提供 | 195 | 未測定 |
| attention / mHC / routing | 未提供 | 21.2 / 16.3 / 2.3 | 未測定 |
| gate_up / down | 未提供 | 3.7 / 1.9 | 未測定 |

当初第7段階はH2D 46 ms / 836 MB、disk H2D 12.7 ms / 181 MB/token。
181 MB / read_wait 138.5 ms = 約1.31 GB/sはdecode側の実効値で、物理NVMe帯域ではない。
保守間隔の拡大でミスが約22→59 experts/tokenになったのが差し戻し理由。

### 現行の保守

- 既定 `--maintenance-interval 1`。1/2/4を比較可能。0は配置固定。
- GLMだけ直近32区切りの正確な使用回数を保持し、最も頻繁な非residentを
  GPU側の最も少ないresidentと交換する。residentが窓内で一度使われていても候補にする。
  同数なら交換しない。頻度ファイルは初期配置に使い、追加の1.25倍条件は廃止。
  使用統計は保守が進行中でも毎区切り更新し、古い32区切り前の回数を引く。
  decodeは1token/区切り、prefillは既存と同じ1chunk/区切り。
- 専用Eコア20のworkerは、事前確保した別GPU画像へH2Dする。
  ホスト昇格時は元GPUを別pinned画像へD2Hし、元ホストから別GPUへH2Dする。
  元のGPU/ホスト画像は読み取り専用のArcで保持し、decodeはそのまま使える。
  ディスク昇格時も元GPUは変更しない。workerはpacking/検証/転送を行い同期後に結果を返す。
- workerの完了通知を**head D2Hで全層の利用を終えたトークン境界だけ**で回収し、
  GPUの対応とhost tierを公開する。使用中の画像を書き換えず、decode側の保守待ちは不要。
  swap計画の走査・ソートと公開はdecode側で行う。保守が未完了なら新しいbatchを投入しない。
- 1batchは最大8交換、H2D+D2H合計256,000,000 bytes以内。
  上限の計算ではdisk交換も2画像分と数える（保守的な上限）。
  NVFP4 plainは8件で226,504,704 bytes。8GPU画像＋8pinned画像を再利用し、
  メモリ確保をworkerのhot pathに置かない。追加GPUは113,252,352 bytes。
  同じdriver/workspace予約内で確保するため、residentスロット数を減らさない。
- エラー時は旧tierを保持して結果を返し、既存failure状態で次のGPU投入を止める。
  Arcがworkerの読み取り元の寿命を保証し、reset/終了時はdrainする。
  `maintenance_wait_ms`は通常のdecodeで0になる。終了drainは従来通り別報告。
- 共通ExpertCacheの既存finish_interval/ヒステリシス/同期applyの契約は変更しない。
  Qwen/Gemmaは今回の直近窓・準備後公開APIを使わない。

### ディスクとRAM

実ルートの全ミスは元から一層分をまとめて投入していたため、再実装しない。
exact top-kは8であり、1層の要求queueも最大8。`--expert-readers 1..16`を追加し、
各readerは独立O_DIRECT/pinned staging/CUDA streamを持つ。
16readerはEコア16..23へ2本ずつ、8は1本ずつ配置する。
8/16で次層の実ルートまで予測することはない。近似prefetchは既定無効を維持。

**読み取り専用CPU計測**（GPU/H2Dなし、release、2026-10-15）:
実HYT layer3の128 experts、payload 1,811,940,864 bytes/run。
各構成warmup2 + 反復3、事前に各readerのstagingを確保。
O_DIRECT読み込み・画像packing・有限値検証を含み、H2DとGPU計算は含まない。

| readers | payload GB/s（3反復） | 中央値 GB/s | wall中央値 s |
|---|---|---:|---:|
| 4 | 4.124 / 4.116 / 4.131 | 4.124 | 0.439406 |
| 8 | 4.094 / 4.153 / 4.096 | 4.096 | 0.442367 |
| 16 | 4.142 / 4.057 / 4.023 | 4.057 | 0.446597 |

5 GB/sには届いていない。この計測では8/16の改善がないため、既定reader数は4を維持。
H2Dを含む実decodeの改善可否は下の4/8/16比較で決める。
単一層の連続128画像の結果から、少数のランダムな層ミスの速度を推定しない。
I/OなしのCPU速度や他ジョブとのSSD競合は測っていない。

`--profile`に`reader_io_ms`（pread）、`reader_pack_ms`（packingと検証）、
`reader_upload_ms`（H2D投入と待ち）、`reader_bytes`（demand/prefetch readerのpayload）を追加。
時間は並列workerの和で、重なるためtoken wallへ加算しない。
既存`disk_bytes`は保守読込も含む。集計器の`disk_bandwidth`は、reader bytes / read_waitと
reader bytes / reader_ioの比を出す。いずれも物理NVMeの総帯域とは扱わない。
物理帯域の判断には上のCPUベンチのwallと、GPU版profileの各区間を併用する。

`--host-gb`の既定は未測定のまま増やさず64を維持。
Linux MemTotalから24 GiBを引いた値をpinned expert予算の上限として拒否する。
128 GB（10進）のRAMなら上限約102.23 GB、88 GBは約40 GBを残す。
24 GiBにはOS・KV・resident側ホスト重み・staging・ページキャッシュを含む。
これは他の巨大モデルとの同時常駐を許可する仕組みではない。
88 GB採用時はhost peak/RSSとMemAvailableを測り、既定の採否を決める。

### 実機コマンド（差し戻し後）

GPUなしで実行したCPU I/Oコマンド:

```sh
GLM53_HYT=/mnt/NVME4TB/hayate/hyt/glm53-stage6-fp8-dense \
  cargo test -p hayate-models --release --offline glm53_direct_io_bandwidth \
  -- --ignored --nocapture
```

統合担当は同じ頻度ファイルを設定して次を実行する。bash、出力先は新規ディレクトリ。
間隔/reader/RAMの変更を個別に比較する。速度はprofileなし、warmup2/反復3。

```sh
export PYTHONPATH=python-tools
export GLM53_HYT=/mnt/NVME4TB/hayate/hyt/glm53-stage6-fp8-dense
export GLM53_FREQUENCIES=/path/to/stage6/frequencies.bin
export GLM53_STAGE7R_OUT=/mnt/NVME4TB/hayate/spill/glm53-stage7-revision
mkdir -p "$GLM53_STAGE7R_OUT"
cargo build --release --features cuda --offline
cargo test -p hayate-models --release --features cuda glm53_async_expert_transfers -- --ignored --nocapture
cargo test -p hayate-models --release --features cuda glm53_parallel_disk_transfers -- --ignored --nocapture
cargo test -p hayate-models --release --features cuda glm53_decode_stage6_graph -- --ignored --nocapture

for config in 1:4:64 2:4:64 4:4:64 1:8:64 1:16:64 1:4:88 32:4:64 0:4:64; do
  IFS=: read -r interval readers host <<< "$config"
  tag="i${interval}-r${readers}-h${host}"
  flags=(--layers 45 --context 4096 --hyt-dir "$GLM53_HYT" --prompt Hello
         --max-tokens 128 --host-gb "$host" --allow-disk --prefill-chunk 128
         --warmup 2 --frequencies "$GLM53_FREQUENCIES"
         --maintenance-interval "$interval" --expert-readers "$readers")
  for repeat in 1 2 3; do
    target/release/hayate run glm53 "${flags[@]}" \
      >"$GLM53_STAGE7R_OUT/$tag-$repeat.out" 2>"$GLM53_STAGE7R_OUT/$tag-$repeat.bench"
  done
  target/release/hayate run glm53 "${flags[@]}" --profile \
    >"$GLM53_STAGE7R_OUT/$tag-profile.out" 2>"$GLM53_STAGE7R_OUT/$tag.profile"
  python3 -m hayate_tools.glm53_profile "$GLM53_STAGE7R_OUT/$tag.profile" \
    >"$GLM53_STAGE7R_OUT/$tag-breakdown.json"
done
# 採用構成は同じflags + --no-graphsでも3回比較。
# MemAvailable/peak RSSも --host-gb 88の実行中に記録する。
```

品質は速度と別実行。既存45層FP8 referenceを用い、KLD<=0.15/top-1>=82%を維持。
同じHYTのfrozenとasyncは全BF16 logitsのbytes一致を要求する。
既存dense FP8構成のKLD 0.041/top-1 91.1%の再現も確認する。

```sh
export GLM53_FP8_REFERENCE=/path/to/45-layer/fp8-reference
python3 -m hayate_tools.glm53_eval --corpus-out "$GLM53_STAGE7R_OUT/corpus" --tokens 512
for interval in 0 1 4; do
  target/release/hayate run glm53 --layers 45 --context 512 --hyt-dir "$GLM53_HYT" \
    --host-gb 88 --allow-disk --prefill-chunk 128 --maintenance-interval "$interval" \
    --eval-ids "$GLM53_STAGE7R_OUT/corpus/input_ids.i32.bin" --eval-prefix 64 \
    --out "$GLM53_STAGE7R_OUT/eval-i$interval"
  python3 -m hayate_tools.glm53_eval --reference "$GLM53_FP8_REFERENCE" \
    --actual "$GLM53_STAGE7R_OUT/eval-i$interval" \
    --out "$GLM53_STAGE7R_OUT/quality-i$interval.json" --check
done
cmp "$GLM53_STAGE7R_OUT/eval-i0/eval_logits.bf16.bin" "$GLM53_STAGE7R_OUT/eval-i1/eval_logits.bf16.bin"
cmp "$GLM53_STAGE7R_OUT/eval-i0/eval_logits.bf16.bin" "$GLM53_STAGE7R_OUT/eval-i4/eval_logits.bf16.bin"
```

`GLM53_FP8_REFERENCE`は同じsource/hash/IDs/context/prefix/prefill-chunkのreferenceディレクトリ。
未作成なら本文のFP8 reference生成手順で先に作る。

### 検証・変更ファイル・残件

`/tmp/qwen38-stage4-env.sh`の既存Nix toolchainを使用、外部依存追加・ネットワーク・commitなし。

- `cargo fmt --all`: 成功。担当外`models/src/lib.rs`の既存mod順差分を復元。
- `cargo clippy --workspace --all-targets --offline -- -D warnings`: 成功、警告0。
- 同 `--features cuda`: 成功、警告0。
- `cargo test --workspace --offline`: 223 passed / 18 ignored、doc tests成功。
- 同 `--features cuda`: 242 passed / 128 ignored、doc tests成功。
- `PYTHONPATH=python-tools python3 -m unittest tests.test_glm53_decode tests.test_glm53_eval tests.test_glm53_nvfp4 -v`:
  20件中18 passed / CUDA2 skipped。dense CPU9ケースの最大相対L2=2.78416453e-08。
- 上記実HYT CPU I/O benchmark: 1 passed、4/8/16各warmup2+反復3、値は表の通り。
- `git diff --check`: 成功。

CPU試験で最近の窓の期限切れ、保守中の統計更新、使われた低頻度residentからの昇格、
準備後公開時の対応検証・二重公開拒否、byte上限、RAM上限を確認。
GPU ignore試験は昇格準備中の旧GPU/host利用、完了後のbytes、元tierの画像数を検証する。
演算・量子化・ルーティング式・expertの縮約順は変更していない。

変更ファイル（担当範囲内のみ）:
`crates/cli/src/glm53.rs`、`crates/models/src/expert_cache.rs`、
`crates/models/src/glm53/{README.md,STAGE7.md,executor.rs,experts.rs,transfer.rs}`、
`python-tools/hayate_tools/glm53_profile.py`、`python-tools/tests/test_glm53_decode.py`。

未確認: GPUでのpreparedコピーとGraph、45層品質、変更後tok/s/ITL/hit/read_wait/H2D/peak、
host88の実測、変更後の物理NVMe帯域。94%hit、4.65/6 tok/s、5 GB/sの達成は主張しない。
他担当への必要なコード変更なし。統合担当は上記GPUテスト・速度・品質・host88比較を実行し、
表の「差し戻し後」を実測で埋める。担当外`models/src/lib.rs`のmod順はfmt対応が必要。
参照: 指定設計文書とローカルGLM/runtimeのみ。Web参照URLなし、third_party変更なし。
