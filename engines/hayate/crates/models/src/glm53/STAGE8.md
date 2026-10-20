# 第8段階: pinned RAM 上のミスを CPU で計算

実装と CPU 検証を実施。GPU を使えないため、15 tok/s、変更後 ITL、
CPU/GPU 相対 L2 < 1e-3、45層 KLD 0.041 / top-1 91.1% の維持は未確認。

## 変更ファイル

- `crates/cli/src/glm53.rs`
- `crates/kernels/src/glm53.rs`
- `crates/kernels/src/cpu/glm53.rs`（新規）
- `crates/kernels/cu/glm53/decode.cu`
- `crates/models/src/glm53/{mod.rs,executor.rs,experts.rs,gpu.rs}`
- `crates/models/src/glm53/{cpu_experts.rs,STAGE8.md}`（新規）

qwen38、共通 CPU pool/runtime、offload、expert_cache、変換器、third_party、
依存 crate は変更しない。CPU モジュールは担当範囲の kernels::glm53::cpu から公開する。
コミットなし。

## 実装と設計判断

- `--host-gb` 既定100 GB、ディスクは既定で拒否。既存の MemTotal − 24 GiB 上限を維持し、
  OS/KV/staging 用の余白を残す。GPU 常駐分との合計で全エキスパートを収容する。
  足りない場合は必要な RAM の正確なバイト数と安全上限を表示し、ロード前にエラーにする。
  全 NVFP4 は 12,096 × 14,156,544 = 171,237,556,224 bytes。
  層ごとに FP8 を残した構成では既存通り最大画像サイズの均一 slot を使うため、容量は増える。
- CPU は pinned RAM の元 NVFP4 をそのまま読む。E2M1 を AVX2 shuffle で展開し、
  タイル配置の E4M3 scale × FP32 matrix scale を乗算、FMA で F32 内積を計算する。
  gate/up 出力の BF16、SwiGLU の clamp と中間 BF16、down 出力の BF16 を保持する。
  source block-FP8 / down FP8 も既存の BF16 重み丸めで計算する。
  Hadamard 入力は GPU と同じ BF16 バッファを D2H し、必要な down 入力の回転は CPU で行う。
- P コア 0,2,…,14 に8本を固定する常駐 pool。AtomicPtr の release/acquire で
  行分割 GEMV を投入し、spin で完了を待つ。起動・確保は層の hot path に置かない。
  API は全 worker を回収してから借用を返し、重みと活性化の寿命を保つ。
  非線形 FP4 を int4 に再量子化しないため、VNNI ではなく AVX2/FMA を使う。
  CPU を int8 化する近似も今回は追加しない。
- `--pcie-frac 0..1`（既定0）は GPU ミスのうち転送する割合。
  端数を層・トークン間で蓄積するため、各層のミスが1件でも比率を指定できる。
  GPU hit は比率に含めない。明示 `--allow-disk` 時の disk miss は従来通り転送する。
  `--expert-prefetch` は比率1だけで使える（CPU に回す画像の無駄な転送を避ける）。
- ルーター/入力回転後に ID と 8 KiB の活性化を回収する。
  転送対象だけを従来の batch_images で解決し、GPU hit と shared expert を起動する。
  CPU はそれと並行して RAM のミスを計算する。CPU slot は GPU の gate/up/down を省略し、
  完了後に各8 KiBの出力だけを同じ GPU stream へ H2D する。
  最後は既存 reduce が expert ID 昇順・同じ BF16 丸めで合算する。
- CUDA Graph は gate/up → down まで。CPU 出力の H2D と reduce/expand を後続に投入する。
  CPU 出力 pinned buffer は層ごとに所有し、次回の router 同期で前回の使用を退役する。
  Layer の Drop も stream を同期する。GPU pointer table は各層の実ルートで更新する。
- 第7段階の非同期昇格、最大8交換・256 MB/回、直近32区切りの統計、準備完了後の
  トークン境界での公開は変更しない。CPU は Arc の不変 RAM 画像を読み、昇格と競合しない。
  prefill と decode-reference は従来の GPU 計算を維持する。

RAM がなお不足する実機構成では、まず表示された必要量まで `--host-gb` を増やす。
安全上限を超える場合の案は、元 FP8 からホスト用だけ Q4Bf16(group128、4.125bit)等へ
オフライン変換すること。NVFP4(4.5bit)比で約8.3%小さく、100GB相当で約8GB削減できる。
ただし今回のローダ/CPU経路はその別形式を扱わない。CPU/GPUの形式間で昇格するには
変換器と配置/転送処理の対応、品質評価が必要で、推論中の再量子化は今回追加しない。

## 計測方法と結果

CPU ベンチは実 HYT layer3 の32エキスパート（453,009,408 bytes > LLC）を事前ロード。
2 sweep の warmup + 5 sweep を固定し、gate/up → SwiGLU → down と pool 同期を含める。
SSD 読込・GPU・H2D は計測から除外。GB/s は各画像を1回読んだとする有効バイト数/wall。
CPUモデルは lscpu で i9-12900KS を確認。外部の並列ジョブは制御していない。

確定した CPU 計測値・チェック結果は末尾に記録する。

1トークンの前後比較（前は統合担当から提供、後はGPUが必要）:

| 項目 | 第7段階 host88/interval4/denseFP8 | 第8段階 |
|---|---:|---|
| tok/s | 4.74 | 未測定 |
| ITL p50 / p95 ms | 202 / 345 | 未測定 |
| GPU hit / GPU+host hit | 91.1% / 99.6% | 未測定 |
| disk reads / 128生成tokens | 166 | ディスク拒否で0を要求、実行未確認 |
| PCIe 転送を含む ITL | 約200 ms | CPU へ置換する経路を追加、未測定 |
| attention / mHC / route / GPU gate_up / down / reduce | 今回の提供値なし | --profile で計測 |
| CPU expert wall / bytes / GB/s | CPU計算なし | --profile と通常出力で計測 |

`--profile` の maintenance 行に `cpu_experts`, `cpu_bytes`, `cpu_ms` を追加。
通常の生成出力にも decode の CPU 件数・バイト・時間・GB/s・ms/token を出す。
CPU ms は GPU と重なるため GPU イベント時間に足さない。
`reduce_ms` は GPU down 終了から reduce 終了までなので、CPU 完了待ち/H2D の隙間を含む。
既存 glm53_profile.py は CPU 欄の集計をまだ行わない（担当外）。生ログに値は残る。

## 実機コマンド

既存と同じ HYT・頻度ファイル・プロンプト・source hash で比較する。
速度は profile なし、warmup2、各比率3回。ディスクを許可するオプションを付けない。
以下は bash。

```sh
export GLM53_HYT=/mnt/NVME4TB/hayate/hyt/glm53-stage6-fp8-dense
export GLM53_FREQUENCIES=/path/to/stage6/frequencies.bin
export GLM53_STAGE8_OUT=/mnt/NVME4TB/hayate/spill/glm53-stage8
export PYTHONPATH=python-tools
mkdir -p "$GLM53_STAGE8_OUT"
cargo build --release --features cuda --offline
cargo test -p hayate-kernels --release --features cuda --offline glm53_cpu_nvfp4_gemv_matches_gpu -- --ignored --nocapture
cargo test -p hayate-models --release --features cuda --offline glm53_cpu_expert_matches_gpu -- --ignored --nocapture
cargo test -p hayate-models --release --offline glm53_cpu_expert_bandwidth -- --ignored --nocapture

for frac in 1 0.75 0.5 0.25 0; do
  flags=(--layers 45 --context 4096 --hyt-dir "$GLM53_HYT" --prompt Hello
         --max-tokens 128 --host-gb 100 --prefill-chunk 128 --warmup 2
         --frequencies "$GLM53_FREQUENCIES" --maintenance-interval 4 --pcie-frac "$frac")
  for repeat in 1 2 3; do
    target/release/hayate run glm53 "${flags[@]}" \
      >"$GLM53_STAGE8_OUT/f$frac-$repeat.out" 2>"$GLM53_STAGE8_OUT/f$frac-$repeat.bench"
  done
  target/release/hayate run glm53 "${flags[@]}" --profile \
    >"$GLM53_STAGE8_OUT/f$frac-profile.out" 2>"$GLM53_STAGE8_OUT/f$frac.profile"
done
# 採用比率について interval 1/2/4、--no-graphs も同条件で比較。
# 実行中の peak RSS/MemAvailable と disk_bytes=0 を確認する。
```

品質は速度と別に実行。現行 dense FP8 の KLD 0.041 / top-1 91.1% の維持を判定する。
同じ CPU/GPU でも F32 内積の加算順が変わるので logits bytes 一致は要求しない。
既存の --check のゲート(KLD<=0.15/top-1>=82%)だけでは「現行品質維持」の判定には不足する。

```sh
export GLM53_FP8_REFERENCE=/path/to/45-layer/fp8-reference
python3 -m hayate_tools.glm53_eval --corpus-out "$GLM53_STAGE8_OUT/corpus" --tokens 512
for frac in 1 0.5 0; do
  target/release/hayate run glm53 --layers 45 --context 512 --hyt-dir "$GLM53_HYT" \
    --host-gb 100 --prefill-chunk 128 --maintenance-interval 4 --pcie-frac "$frac" \
    --frequencies "$GLM53_FREQUENCIES" \
    --eval-ids "$GLM53_STAGE8_OUT/corpus/input_ids.i32.bin" --eval-prefix 64 \
    --out "$GLM53_STAGE8_OUT/eval-f$frac"
  python3 -m hayate_tools.glm53_eval --reference "$GLM53_FP8_REFERENCE" \
    --actual "$GLM53_STAGE8_OUT/eval-f$frac" \
    --out "$GLM53_STAGE8_OUT/quality-f$frac.json" --check
done
```

## 他担当への必要な変更・未確認

- `python-tools/hayate_tools/glm53_profile.py` に CPU件数/bytes/ms の集計を追加する。
  GPUと重なる CPU ms を token wall へ二重加算しない。
- 担当外 `crates/models/src/lib.rs` の既存 mod 宣言順が fmt と異なるため、統合側で順序整理が必要。
  今回の fmt で生じる担当外差分は戻す。
- GPU実行、実際のpinned RAM/VRAM配置とpeak、45層品質、GPU overlap、変更後tok/s/ITLは未確認。
- RAM不足時の別形式への変換は提案のみ。
- Web参照URLなし。指定設計文書とローカルGLM/runtime/量子化仕様のみ参照し、外部実装をコピーしていない。

## 確定した検証結果（2026-10-15）

既存の `/tmp/qwen38-stage4-env.sh` から Nix の cargo/rustc/nvcc を PATH に設定。
依存追加・ネットワーク・GPU実行・commit は行っていない。

| 実行コマンド | 結果 |
|---|---|
| `cargo fmt --all` | 成功。担当外lib.rsのmod順差分だけ復元 |
| `cargo clippy --workspace --all-targets --offline -- -D warnings` | 成功、警告0 |
| 同 `--features cuda` | 成功、警告0、CUDAコンパイル成功 |
| `cargo test --workspace --offline` | 230 passed / 0 failed / 19 ignored、doc tests成功 |
| 同 `--features cuda` | 249 passed / 0 failed / 131 ignored、doc tests成功 |
| 上記 `glm53_cpu_expert_bandwidth` release コマンド | 1 passed、2 warmup + 5測定 |
| `git diff --check` | 成功 |

新規の非ignore CPU試験は7件。SIMD vs F64の相対L2 < 1e-5、独立した期待値による
scale tile境界、BF16 ties-to-even、常駐poolの繰り返し実行と行分割、比率の端数累積と
不正値拒否、SwiGLU clamp/Hadamard、実寸定数重みのNVFP4およびdown FP8のFFNを確認した。
新規GPU試験2件はignore。GEMVの同一入力比較、実HYT expertのCPU/GPU比較は実機で行う。

最終CPUベンチは上記のビルド・全体検証が完了してから別実行した。
読み込む32画像は全NVFP4、合計453,009,408 bytes/sweep。
main PCのPコア8本、外部ジョブとの競合は制御していない。

| 測定 | wall s / 32 experts | 有効GB/s | ms/expert |
|---|---:|---:|---:|
| 1 | 0.040223 | 11.262317 | 1.256983 |
| 2 | 0.037434 | 12.101419 | 1.169825 |
| 3 | 0.041322 | 10.962877 | 1.291317 |
| 4 | 0.043929 | 10.312373 | 1.372773 |
| 5 | 0.048529 | 9.334856 | 1.516525 |
| 中央値 | 0.041322 | **10.962877** | **1.291317** |

これはCPU FFN全体の有効帯域であり、純粋なRAM帯域、H2D帯域、tok/sではない。
1トークンの実際のCPU件数、attentionとの重なり、GPU合流の待ち時間は未計測。
15 tok/s達成や品質維持の判断は、上の実機コマンドの結果を待つ。
