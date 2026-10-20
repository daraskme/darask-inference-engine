第10段階の差し戻し後の既定は E23 + park、窓64、最大1交換、頻度差4、token末尾で保守完了。4構成の実機比較手順は [STAGE10.md](STAGE10.md#差し戻し対応保守の帯域競合) を参照。変更後のGPU速度・品質は未測定。
第7段階の差し戻し対応（毎tokenの非同期保守・直近32区切り・4/8/16 readerの実測と実機手順）は [STAGE7.md](STAGE7.md#差し戻し対応2026-10-15) を参照。
第6段階の記録は [STAGE6.md](STAGE6.md)。以下は第5段階時点の記録。

# GLM-5.3-Flash 第5段階: teacher forcing による NVFP4 品質判定

**実機の品質ゲートは未確認。** 判定は45層の固定コーパスに対する
FP8基準の KLD <= 0.15 かつ top-1一致率 >= 82%。自然対数、温度1。
第3・4段階の短い greedy / logits L2 は回帰診断として残すが、量子化の採否には使わない。
重みだけの測定では回転が悪化し、MSEが改善したため、まず回転なし+MSEを評価する。

## 実重み測定

元 FP8 × F32 block scale を F32 で復元し、途中で BF16 に丸めない。
層3 / 6 / 10、expert0 / 17 / 143 の gate/up/down、計27行列を全要素で測った。
入力側の正規化 Sylvester Hadamard は直交するため、回転後の量子化 L2 は元の基底と等しい。
活性化の分布を含む logits の品質はこの表から判断できない。

| ブロック | RTN 平均相対 L2 | MSE 平均相対 L2 |
|---|---:|---:|
| なし | 9.20175% | **8.15051%** |
| 32 | 9.51323% | 8.60018% |
| 64 | 9.51224% | 8.59872% |
| 128 | 9.51243% | 8.59898% |
| 256 | 9.51261% | 8.59913% |

CPU8 threads、Torch 2.13.0+cu130。RTN の比較25.0 s、MSE の比較130.1 s。
測定全件・変換容量・CPU pilot は
`python-tools/tests/glm53_nvfp4_measurements.json` に記録した。
再測定は `nix develop .#tools` 内で:

```sh
export PYTHONPATH=python-tools
python3 -m hayate_tools.glm53_nvfp4 --out /tmp/glm53-rtn.json
python3 -m hayate_tools.glm53_nvfp4 --mse --out /tmp/glm53-mse.json
```

## 変換と実行

- `--hadamard-block 0|32|64|128|256`。既定0。NVFP4 routed expert のみ。
  `W' = W H^T` と `x' = H x` を対応させる。CUDA の蝶演算は F32、出力は BF16。
  decode は全選択 expert の gate/up に同じ回転済み入力を共有する。
  grouped prefill は token の入力を一度回転してから gather する。
  down は既存の clipping / SiLU / BF16 product を作ってから別途回転する。
- `--mse`。各16要素ブロックで RTN と最大値比
  0.75 / 0.8 / 0.85 / 0.9 / 0.95 / 1.05 / 1.1 / 1.15 / 1.2 を比較する。
  E4M3 に格納したスケールと E2M1 RTN の復元 MSE が最小の候補を採用する。
  RTN を候補に残すので重みのブロック MSE は増やさない。
  行列ごとの FP32 スケールは既存と同じ最大値式で、各 expert の各射影が独立して持つ。
- `--down-fp8`。gate/up は NVFP4、down は元の128×128 block-FP8 をそのまま保存する。
  down の入力には回転を掛けない。既存 FP8 GEMV / restore を使う。
  これは品質候補で、45層では容量増加により disk miss が増える可能性がある。
- 回転は各層の HYT 内の F32 `[1]` エントリ `mlp.experts.hadamard_block` に保存する。
  原子 rename で重みと同時に公開し、metadata 不在は回転0として扱う。
  不正ブロック、型、gate/up の形式は拒否する。HYT version / QuantSpec は変えない。
  down の形式も HYT から読むため、実行時に別の量子化フラグは必要ない。
- 1行列ずつ変換する。既存 RTN / FP8 の既定と旧 HYT は引き続き使える。
  Layer oracle、prefill、decode、GPU/pinned RAM/NVMe の画像ロード・昇格を同じ配置に揃えた。
  CPU Q4 計算や先回りしたカーネル最適化は追加していない。

NVFP4画像は各行列のpayload後を256 bytesで整列する。
行列 stride は4,718,848 bytes、3行列で14,156,544 bytes。
混合画像は gate/up の後に元FP8 down + F32 block scales を置き、17,828,352 bytes。
全12,096 expert のruntime画像は混合化で **44.414 GB 増える**。
HYT は各テンソルを独立した4096-byte整列のentryとして保存する。

45層の実ヘッダからの出力容量は、表・整列込みで全NVFP4 **186.608 GB**、
down FP8 **231.001 GB**（+44.394 GB）。回転64+MSEの実ヘッダ計画は186,607,804,584 bytes（+22,696 bytes）。
回転なし+MSEの候補を新しい出力先へ変換する:

```sh
python3 -m hayate_tools.glm53_convert --layers 45 --expert-quant nvfp4 --mse \
  --device cuda --out /mnt/NVME4TB/hayate/hyt/glm53-nvfp4-mse --estimate
python3 -m hayate_tools.glm53_convert --layers 45 --expert-quant nvfp4 --mse \
  --device cuda --out /mnt/NVME4TB/hayate/hyt/glm53-nvfp4-mse
# 回転候補: 上記に --hadamard-block 64 を追加し、別の出力先を使う。
# down候補: 上記に --down-fp8 を追加し、別の出力先を使う。
```

変換は各層の秒数と総秒数を記録する。GPU変換時間は未計測。
CPU8 threads の layer3/expert0 の3行列、読み込み・復元・書き込み・fsync込み、
warmup1 / 反復3 の中央値は RTN 0.370 s、MSE 2.948 s、回転64+MSE 3.325 s、
MSE+down FP8 2.028 s。全expertへの単純外挿はそれぞれ1.24 / 9.90 / 11.17 / 6.81時間。
ヘッダ計画、全入力SHA-256、resident/globalsのコピーは外挿に含まない。
これは1 expert の CPU pilot で、GPU時間の予測ではない。
8 / 16層ならまず `--layers 16` で候補を変換できる。同じHYTを8層にも使える。

## 第5段階の変換・容量・所要時間

以下は `nix develop .#tools --offline` 内、リポジトリルートで実行する。
`export PYTHONPATH=python-tools` を設定する。入力は既定の元FP8 safetensors、
出力先は新規ディレクトリ。既存FP8の `.hyt` は保持する。

| 候補 | 追加フラグ | 45層のディスク容量(bytes) |
|---|---|---:|
| NVFP4素 | なし | 186,607,781,888 |
| NVFP4 MSE | `--mse` | 186,607,781,888 |
| MSE + downのみFP8 | `--mse --down-fp8` | 231,001,327,616 |
| MSE + 回転64 | `--mse --hadamard-block 64` | 186,607,804,584 |

全4候補を同時に残すと790.825 GB。元331.4 GB・既存FP8・評価ファイルは別途必要。
容量は `--estimate` でヘッダだけから再計算できる。以下の変換コマンドに
`--estimate` を追加すれば、変換・入力全体のSHA256・出力作成は行わない。

```sh
export PYTHONPATH=python-tools
H=/mnt/NVME4TB/hayate/hyt
python3 -m hayate_tools.glm53_convert --layers 45 --expert-quant nvfp4 --device cuda \
  --out "$H/glm53-nvfp4-plain"
python3 -m hayate_tools.glm53_convert --layers 45 --expert-quant nvfp4 --device cuda \
  --mse --out "$H/glm53-nvfp4-mse"
python3 -m hayate_tools.glm53_convert --layers 45 --expert-quant nvfp4 --device cuda \
  --mse --down-fp8 --out "$H/glm53-nvfp4-down-fp8"
python3 -m hayate_tools.glm53_convert --layers 45 --expert-quant nvfp4 --device cuda \
  --mse --hadamard-block 64 --out "$H/glm53-nvfp4-rot64"
```

GPUの所要時間は未計測なので、実機の `--pilot` から見積もる。
元重みのlayer3/expert0を1回warmup、3回変換し、中央値×12,096 expertsを
`estimated_expert_conversion_s` に出す。読み込み・F32復元・H2D・量子化・D2H・fsync込み。
全入力SHA256、resident/globalのコピー、ヘッダ計画は含まない。
一回の小さなwarmなpilotの外挿なので、NVMeの持続書き込み速度や各expertの分布差に注意する。
全変換では各層と総時間を別に記録する。GPUカーネル時間だけの測定ではない。

```sh
S=/mnt/NVME4TB/hayate/spill/glm53-eval
mkdir -p "$S/pilot"
TMPDIR="$S/pilot" python3 -m hayate_tools.glm53_convert --layers 45 \
  --expert-quant nvfp4 --device cuda --pilot > "$S/pilot-plain.json"
TMPDIR="$S/pilot" python3 -m hayate_tools.glm53_convert --layers 45 \
  --expert-quant nvfp4 --device cuda --pilot --mse > "$S/pilot-mse.json"
TMPDIR="$S/pilot" python3 -m hayate_tools.glm53_convert --layers 45 \
  --expert-quant nvfp4 --device cuda --pilot --mse --down-fp8 > "$S/pilot-down-fp8.json"
TMPDIR="$S/pilot" python3 -m hayate_tools.glm53_convert --layers 45 \
  --expert-quant nvfp4 --device cuda --pilot --mse --hadamard-block 64 > "$S/pilot-rot64.json"
```

CPU2 threadsで新pilot経路も確認した。素の中央値0.6844 s/expert、
全expertの外挿8,278 s（2.30時間）。これは `/tmp` でのCPU計測でGPU時間の予測ではない。
上の第4段階CPU8 threadsの値とも条件が異なる。

## 固定コーパスと品質ゲート

`python-tools/hayate_tools/glm53_eval_corpus.txt` は自作の英語技術文・Pythonコード・日本語。
GLMの既存Rust BPEで528 tokensになる。全IDを `glm53_eval_corpus.ids.json` に固定し、
既定で先頭512 tokens、短縮時は `--tokens 256` を使う。text/tokenizer/IDsのSHA256を記録する。
今回のコーパスをMSE探索・頻度収集に使わない。ローカルtokenizerとの照合はCPUで可能:

```sh
cargo test -p hayate-models --offline glm53_eval_corpus_tokenizer -- --ignored --nocapture
python3 -m hayate_tools.glm53_eval --corpus-out "$S/corpus" --tokens 512
cargo build --release --features cuda --offline
```

`--eval-ids FILE --eval-prefix N --out DIR` はQwen3.8と同じI32 little-endian ID列を読む。
最初のN tokensを既存prefillで通し、その後は固定IDを1つずつstepに渡す。
**row jは ids[N+j] を消費した直後の次トークン分布**（Qwen3.8と同じ行順）。
生成ID・EOSによる打ち切りは使わない。評価モードはtokenizer/chat templateをロードしない。

保存は全語彙の**BF16 logits**。実行器のhead出力が元々BF16なので追加丸めなし。
F16より指数範囲が広く、F32保存の半分。top-kと残り質量への近似は行わない。
FP8とNVFP4の全語彙分布から厳密に KLD(P_FP8 || P_candidate) をCPU F64 log-softmaxで計算する。
FP8も保存精度・実行器が同じ。154,880語彙、448行で138,772,480 bytes/構成（138.8 MB）。
`eval_logits.bf16.bin`、`eval_ids.i32.bin`、最後に完了マーカーの `eval.json`
（prefix / rows / vocab / source_hash / layers / prefill_chunk / dtype）を書く。出力ディレクトリの再利用は拒否する。
NaN/Inf、異なるsource hash・層数・ID列・prefix・prefill chunk・語彙、ファイルサイズの不整合は拒否する。
比較器は同じmanifestの全語彙F32保存も読める。

KLD / top-1は448位置。平均NLLは実際の次IDを使うため、次IDのない最終行を除く447位置。
`mean_nll_delta = NLL_candidate - NLL_FP8`、正なら悪化。単位はnats/token。
`--check` は KLD <= 0.15 かつ top-1 >= 0.82 以外で終了コード1。
8/16層で動作確認できるが、品質の採否は45層で決める。
全候補でlayers/context/prefix/prefill-chunkと入力IDsを揃える。

```sh
IDS="$S/corpus/input_ids.i32.bin"
target/release/hayate run glm53 --layers 45 --context 512 \
  --hyt-dir "$H/glm53-fp8" --allow-disk --host-gb 64 --prefill-chunk 128 \
  --eval-ids "$IDS" --eval-prefix 64 --out "$S/fp8"
for V in plain mse down-fp8 rot64; do
  target/release/hayate run glm53 --layers 45 --context 512 \
    --hyt-dir "$H/glm53-nvfp4-$V" --allow-disk --host-gb 64 --prefill-chunk 128 \
    --eval-ids "$IDS" --eval-prefix 64 --out "$S/$V"
  # --check付きで候補ごとにゲート判定。失敗時もJSONは保存される。
  python3 -m hayate_tools.glm53_eval --reference "$S/fp8" --actual "$S/$V" \
    --out "$S/$V-report.json" --check
done
```

速度測定とは別実行。1 tok/sを全入力に仮置きするなら512 tokensで約8.5分/構成+ロード。
これは既知のFP8 prefillからの粗い作業時間の目安で、teacher-forced stepの速度やdisk missで変わる。

## 不合格時の層感度測定

まずMSE・down FP8の実測を比べる。down FP8が通れば、44.394 GBのディスク増と
定常速度を測って採否を決める。それでも通らなければ、NVFP4のMoE層を1層ずつFP8へ戻して
同じコーパスを通し、`baseline KLD - restored KLD` が大きい順に並べる。
KLD改善、top-1改善、追加ディスク容量、実行時間、完全なコマンドを `sensitivity.json` に保存する。

```sh
python3 -m hayate_tools.glm53_sensitivity --reference "$S/fp8" --actual "$S/mse" \
  --fp8-hyt "$H/glm53-fp8" --candidate-hyt "$H/glm53-nvfp4-mse" \
  --hayate target/release/hayate --out "$S/sensitivity" --layers 45
# まず数層だけ測るなら --restore-layers 3 20 44 を追加。
# 改善量の大きい層番号を選んで、組み合わせを必ず再評価する（以下は例示番号）。
python3 -m hayate_tools.glm53_sensitivity --reference "$S/fp8" --actual "$S/mse" \
  --fp8-hyt "$H/glm53-fp8" --candidate-hyt "$H/glm53-nvfp4-mse" \
  --hayate target/release/hayate --out "$S/mixed-selected" --layers 45 \
  --restore-layers 3 20 --combine
```

元の重みは変更せず、各層ディレクトリへのsymlinkだけで混合HYTを作る。
同じsource hashから変換したFP8/NVFP4であることをRustローダーが検証する。
層ごとの形式と回転metadataに従って既存FP8/NVFP4計算を選ぶ。
現在の共通ExpertCacheは等サイズslotなので、混合時は最大のFP8 imageに揃えてNVFP4の末尾を0埋めする。
均一形式の既存容量は変わらないが、混合はNVFP4のslot数を減らすため、**層感度診断用**。
そのtok/sを高速な混合精度実装の性能とみなさない。品質が通る最小の層集合を決めた後、
expert単位へ絞るには該当層内のexpert別感度の追加測定と可変形式配置が必要。
今回は使わないexpert単位の量子化・可変サイズキャッシュは追加していない。
改善の加算は保証されないため、順位上位をまとめた実行でゲートを再確認する。

全MoE42層のsweepは43回分の評価（元候補含む）。1 tok/s仮置きで約6時間+各回ロード。
小さい評価コーパスや `--restore-layers` で候補を絞ってから全走査できる。

## 45層の定常速度測定

品質に通った候補だけ測る。頻度収集は評価と別prompt、速度測定はwarmup2 / 反復3、
生成128 tokens、prefill chunk128で固定する。logits保存を速度測定に付けない。

```sh
GLM_HYT=/mnt/NVME4TB/hayate/hyt/glm53-nvfp4-mse
target/release/hayate run glm53 --layers 45 --hyt-dir "$GLM_HYT" \
  --prompt 'Explain how to sort an array.' --max-tokens 128 --host-gb 64 --allow-disk \
  --warmup 2 --save-frequencies /tmp/glm53-frequencies.bin 2>/tmp/glm53-frequencies.log
for R in 1 2 3; do
  target/release/hayate run glm53 --layers 45 --hyt-dir "$GLM_HYT" \
    --prompt Hello --max-tokens 128 --host-gb 64 --allow-disk --prefill-chunk 128 \
    --warmup 2 --frequencies /tmp/glm53-frequencies.bin 2>"/tmp/glm53-steady-$R.log"
done
```

warmup後のcache配置を保持し、時間・hit・readカウンタをリセットして計測する。
`decode_gpu_hit_rate` / `decode_gpu_host_hit_rate` / `decode_disk_reads` をdecode区間だけで出す。
頻度ファイルは層数と同じ長さが必要。FP8/各候補の比較では同じ頻度ファイルを使用する。
TTFT、ITL p50/p95、tok/s、実decode_steps、GPU/RAM/disk容量、ページキャッシュ状態も残す。
`disk_reads` は expert image の読み込み回数で、物理NVMe I/O回数ではない。

## 検証と未確認事項

実行結果（`nix develop .#tools` の既存ツール、すべてoffline）:

- `cargo fmt --all`: 成功。担当外 `models/src/lib.rs` の既存mod順変更だけ復元。
- `cargo clippy --workspace --all-targets --offline -- -D warnings`: 成功、警告0。
- 同 `--features cuda`: 成功、警告0。
- `cargo test --workspace --offline`: 211成功、17 ignore。
- 同 `--features cuda`: 227成功、112 ignore。
- `PYTHONPATH=python-tools python3 -m unittest tests.test_glm53_eval tests.test_glm53_nvfp4 -v`:
  13成功、GPU2 skip（15件）。
- `cargo test -p hayate-models --offline glm53_eval_corpus_tokenizer -- --ignored --nocapture`:
  1成功。実tokenizerと固定528 IDsが完全一致、約2.25秒。
- `python3 -m hayate_tools.glm53_eval --corpus-out /tmp/glm53-stage5-frozen-corpus`: 成功、512 IDs出力。
- 45層の各候補のヘッダ計画: 上表のbytesを確認。CPU2 threadsの実expert pilotも実行（上記）。

サンドボックスのPATHにはcargo/python3/nvccがなかったため、Nix tools環境を取得して使用した。
初回のRust test起動は環境の異なるglibcが混在してabortしたため、Nix glibcを
`LD_LIBRARY_PATH`の先頭にして解消した。リポジトリのビルド設定は変更していない。

CPUでは指標の独立したスカラー確率式、NLLの次ID対応、chunk境界、top-1ゲート、
異なるsource/contextや壊れたファイルの拒否、固定コーパス、層差し替え・組み合わせ、
NVFP4 imageのFP8サイズslotへの0埋めを確認した。
GPUの行順・保存値をforced stepと照合するテストは `#[ignore]`。

```sh
PYTHONPATH=python-tools python3 -m unittest tests.test_glm53_eval tests.test_glm53_nvfp4 -v
GLM53_HYT=/mnt/NVME4TB/hayate/hyt/glm53-fp8 \
  cargo test -p hayate-models --features cuda glm53_eval_matches_forced_steps -- --ignored --nocapture
```

GPU実行、45層のKLD/top-1/NLL、層感度順位、GPU変換時間、定常速度は未確認。
4-bitの行列誤差からモデル品質を推定しない。品質ゲート通過は主張しない。
参照は `docs/ARCHITECTURE.md` / `docs/MODELS.md` とローカル既存GLM実装、
Qwen3.8 worktreeの `qwen38_nvfp4_eval.py` / CLI / runtime（読み取りのみ）。
Web参照URLなし。外部依存追加・git commitなし。qwen38 / third_partyは変更していない。
他担当への必須コード変更なし。統合担当は上のGPU変換・評価・速度計測を実機で行う必要がある。

変更ファイル:

- `crates/cli/src/glm53.rs`
- `crates/models/src/glm53/README.md`
- `crates/models/src/glm53/eval.rs`
- `crates/models/src/glm53/executor.rs`
- `crates/models/src/glm53/experts.rs`
- `crates/models/src/glm53/gpu.rs`
- `crates/models/src/glm53/gpu_tests.rs`
- `crates/models/src/glm53/mod.rs`
- `crates/models/src/glm53/tokenizer.rs`
- `python-tools/hayate_tools/glm53_convert.py`
- `python-tools/hayate_tools/glm53_eval.py`
- `python-tools/hayate_tools/glm53_eval_corpus.txt`
- `python-tools/hayate_tools/glm53_eval_corpus.ids.json`
- `python-tools/hayate_tools/glm53_sensitivity.py`
- `python-tools/tests/test_glm53_eval.py`
