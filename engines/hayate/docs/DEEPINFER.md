# DeepInfer の調査と取り込み

参照: [deepseek-ai/DeepInfer](https://github.com/deepseek-ai/DeepInfer)、コミット `774ce244b87cc0eb8d4b94ab9f5de2ab22049f01`。Apache-2.0 の出典とライセンスを `NOTICE` と `licenses/DeepInfer-LICENSE` に残した。

2026-10-21 JST: 同じ作業ツリーの変更と保存済み実機ログを、量子化調査側のチャットで引き継いだ。
DeepInfer 調査・sampler・二重バッファローダーの完了状態は維持する。
GSQ / RCO と EXL3 / Gyro の統合案、および実装済み・設計段階の区別は
[量子化調査の統合状況](GLM53_QUANT_RESEARCH.md) を参照する。
ユーザー指示により GPU 使用は一時停止中。明示的な再開指示まで CPU の調査・設計を進める。

## 取り込んだ考え方

DeepInfer の [fused_sample.py](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/layers/tilelang_ops/cuda/fused_sample.py) は、語彙のフィルタリングに基数ヒストグラムを使い、全語彙の比較ソートを避ける。

Hayate の GLM serve は BF16 ロジットをホストへ読み、CPU で top-p を処理する。そのため TileLang カーネルの移植ではなく、**正の f32 確率を安定なバイト基数ソートで並べる Rust 実装**を独立に書いた。全語彙の比較ソートを 4 パス以下の線形走査に置き換える。同じ値だけのバイトは再配置を省略し、作業領域は再利用する。

候補が 2048 個以上で、top-k による部分選択をしていない場合に使う。小さな候補集合と部分選択後の top-k は従来の比較順序を使う。この分岐は数値精度・ハードウェアのフォールバックではなく、集合サイズに合ったアルゴリズムの選択。

同じ確率のトークンを ID 昇順に保ち、top-p の累積和・乱数・抽選の順序を変えない。温度、min-p、top-k、繰り返しペナルティを含むケースで比較ソート版と候補列・選択トークンの完全一致を検証する。

## サンプラー単体の測定

ローカル CPU、release ビルド、語彙 154,880、temperature 1.0、top-p 0.95、top-k なし。ウォームアップ 8 回の後、64 回のサンプルを 7 組測り、1 回あたりの中央値を比較した。実測ファイルは以前の GLM FP8 評価が保存した BF16 の 1 行。

| ロジット | 比較ソート | 基数ソート | 倍率 |
|---|---:|---:|---:|
| 合成分布 | 2.974 ms | 1.554 ms | 1.91 倍 |
| GLM FP8 の保存ロジット | 3.937 ms | 2.155 ms | 1.83 倍 |

保存ロジット: `/mnt/NVME4TB/darask-inference-engine/local/hayate/spill/glm53-dealignai/fp8-ref/prefix_logits.bf16.bin`。

```sh
cargo test -p hayate-llm --release radix_sample_benchmark -- --ignored --nocapture
HAYATE_BENCH_LOGITS=/path/to/one-row.bf16.bin \
  cargo test -p hayate-llm --release radix_sample_benchmark -- --ignored --nocapture
```

これはサンプラーだけの測定で、推論全体の tok/s 改善率ではない。モデルの量子化・カーネル・エキスパート配置はこの最適化では変えない。

## 今回移植しなかった部分

- SM100 向けの GPU カーネル: 手元は SM120 で、同じ Blackwell でもカーネルとリソース条件が異なる。
- [信頼度に基づく MTP 検証量の配分](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/layers/tilelang_ops/cuda/schedule_mtp_varlen.py): 複数系列とドラフトの信頼度を使う方式。Hayate の serve は現在 MTP なしの単一 GPU ワーカーで、同じ入力情報を持たない。
- SWA bounded replay: DeepSeek-V4.1 の窓と巻き戻し条件に基づく。GLM の再帰状態と疎アテンションに同じ打ち切りを適用できる根拠はない。
- 大規模 EP / EPLB、分離サービング: 1 枚の GPU と CPU オフロードを使う現環境には対応しない。

非同期実行、メモリの事前確保、CUDA Graph、CPU/GPU の重ね合わせは Hayate に既にある。DeepInfer を依存関係に追加する必要はない。

## 専用化後の検証

- `cargo test --workspace --offline`: 179 件成功、実機・計時用の 12 件は通常実行の対象外。
- CUDA release ビルドと CUDA 有効時の全テストのリンク成功。
- 実機の小規模 CUDA テスト 5 件成功: dense decode と Graph 再生、LoRA、grouped GEMM、NVFP4 GEMV、LOWBIT/NVFP4 のビット一致。共有 GPU ロックを取得して実行。
- GLM・量子化の Python テスト: 64 件成功、実機用の 3 件は対象外。GPU 共有プロトコルの Python テスト: 13 件成功。
- GLM テンプレート: transformers 5.17.0 と 868 ケースすべてバイト一致。
- 保存済み GLM ロジットを使う比較で、32 回の抽選と全候補列が比較ソート版と完全一致。
- CLI / server / llm の Clippy と差分の空白検査成功。

既存の GPU ジョブを継続したまま検証したため、45 層のモデル全体の通し生成・tok/s は再測定していない。

## 追加調査: 量子化と未マージ PR（2026-10-20 UTC）

main の参照コミットは上記と同じ。GitHub API から取得できた PR 95 件のタイトル・説明を選別し、
下表の #52 / #51 / #54 / #111 / #107 / #92 は差分まで読んだ。
これらは取得時点で open、レビュー提出は 0 件。提案者の測定値は Hayate の実測と区別する。
取得した状態・head SHA・説明は `spill/deepinfer-quant-research/pr-snapshot.json` に保存した
（リポジトリ外、ローカルの検証資料）。

| PR | 参考になる点 | Hayate への判断 |
|---|---|---|
| [#52: pinned 二重バッファ](https://github.com/deepseek-ai/DeepInfer/pull/52) | 次のディスク読み込みと前の H2D 転送を重ねる。提案者は H20・warm cache の単体試験で 2.06 倍と報告 | **今回実装**。Hayate は既に pinned + 4 ワーカーなので、改善対象は同期転送と全テンソル分の一時領域 |
| [#92: SM90 と圧縮状態の MoE](https://github.com/deepseek-ai/DeepInfer/pull/92) | FP4 のまま保持する W4A8 と、FP8 展開方式を比較。後者は提案者の実装で約 1.9 倍の expert メモリ。hybrid は約 2.9 倍 | 保存 bpw と常駐 bpw を別々に計測する設計を参考にする。Hayate の EXL3/LOWBIT は既に直接復号して計算するため、Hopper カーネルの移植は不要 |
| [#107: EPLB の統計帰属](https://github.com/deepseek-ai/DeepInfer/pull/107) | 配置変更前のスロット統計を新しい expert と取り違える不具合。記録時に論理 expert ID へ集約 | Hayate の `ExpertCache` は `ExpertId` に紐づく不変の添字で counts/recent を保持し、slot の入替えと分離済み。同じ不具合は確認されない |
| [#54: decode の固定 CPU 費用](https://github.com/deepseek-ai/DeepInfer/pull/54) | profiler が無効なときの処理省略、イベント再利用、空の出力の早期終了 | Python/pybind の費用が中心。Rust の Hayate で数値を流用しない。イベントは生成と使用完了の対応を保って再利用する |
| [#111: host overhead](https://github.com/deepseek-ai/DeepInfer/pull/111) | 大量待機列の毎ステップ走査を省略、まとめてキャンセル、出力をまとめて受け渡す | 10万件待機・1024系列などの条件。現在の単一ワーカーに同じ効果は期待できない。ログ用集計を必要時だけ行う考え方は有用 |
| [#51: CUDA VMM](https://github.com/deepseek-ai/DeepInfer/pull/51) | 仮想アドレスを固定したまま物理ページを差し替え、Graph のポインタを保つ土台 | 将来の KV 退避・伸縮に有望。ただし PR 自体は割り当ての置換までで、sleep/wake は未実装。既存 CUDA IPC との互換性もないため今回は保留 |

注意点:

- #92 の humming 方式にも小さい scale を再丸めする条件がある。FP4 常駐だけを根拠に無損失とは扱わない。
  proposal の `moe.py` は exponent span、対象行列数、重み誤差を明示している。
- [#91](https://github.com/deepseek-ai/DeepInfer/pull/91) / [#101](https://github.com/deepseek-ai/DeepInfer/pull/101) は
  MXFP4 → FP8 の互換経路。容量削減を目的とする Hayate の優先候補にはしない。
- [#87](https://github.com/deepseek-ai/DeepInfer/pull/87) の streaming prefill は文書だけの RFC。
  実装済みの最適化として数えない。
- [#34](https://github.com/deepseek-ai/DeepInfer/pull/34) の温度 0 と近接 logits の問題は、Hayate では
  `Sampler::sample` が温度 0 を直接 argmax にするため該当しない。
- [#80](https://github.com/deepseek-ai/DeepInfer/pull/80) は非有限値をゼロにして比較する危険を指摘。
  Hayate の品質比較は既に NaN/Inf を拒否する。量子化試験でもこの条件を保つ。

### 今回の HYT ローダー変更

`crates/models/src/glm53/load.rs` を、各ワーカー **8 MiB × 2** の循環ステージングへ変更。
同時に 4 ワーカーでも pinned 一時領域は最大 **64 MiB**。従来はワーカーごとの最大テンソル全体を確保した。
`HytFile::read_tensor_range` はエントリ内の範囲だけを positional read する。

- H2D 中に次のバッファを読み込み・検証する。バッファ再利用前に対応する CUDA event を待つ。
- BF16/F32/FP8 の有限性、per-tensor scale の正値・有限性、NVFP4 の padded scale を従来どおり検査する。
  payload と scale を別区間にして、FP32 scale がチャンク境界で切れないようにした。
- 全転送の完了後だけ GPU バッファを返す。途中の I/O・検証・転送エラーや panic でも、
  ステージングを破棄する前に専用ストリームを同期する。
- 重みの再量子化はしない。GPU に置くバイト列は同じ。
  これはロード時間とホスト一時メモリの改善で、モデル容量や decode tok/s の改善ではない。

### ローダーの実機検証

RTX PRO 6000 Blackwell / i9-12900KS、CUDA release ビルド、共有 GPU ロック内で測定。
旧版と新版とも最大 4 ワーカーで、比較順を交互にし、ウォームアップ後の 5 回の中央値を使った。
OS のファイルキャッシュが温まった条件。GPU 確保・ファイル読み込み・検証・転送完了待ち・
pinned 領域の解放を含み、返した GPU バッファの解放時間を除く。

| 対象 | 転送バイト数 | 従来 | 二重バッファ | 倍率 | pinned 確保量: 従来 → 新版 |
|---|---:|---:|---:|---:|---:|
| sc260 `globals.hyt` の全 3 tensor | 1,903,173,636 | 1.693 秒 | 0.719 秒 | 2.36 倍 | 1.772 GiB → 32.016 MiB |
| sc260 層 0 の常駐 26 tensor | 290,439,944 | 140.91 ms | 129.69 ms | 1.087 倍 | 112.750 MiB → 49.500 MiB |
| sc260 層 3 の常駐 27 tensor | 154,033,032 | 119.14 ms | 100.93 ms | 1.180 倍 | 78.750 MiB → 45.500 MiB |

この共通重みではロード時間を 57.5%、pinned 確保量を 98.2% 削減。
pinned の数値は実装が確保するバッファサイズの合計であり、プロセス全体の peak RAM ではない。
全モデルの起動時間や生成速度の改善率は、この結果から推定しない。

- CPU テスト: `hayate-core` 41 件、`hayate-models` 62 件成功。
  範囲読み込みの境界・オーバーフロー拒否と、チャンク検査の従来検査との一致を含む。
- CUDA テスト: 2 件成功。BF16 / F32 / per-tensor FP8 / NVFP4 のチャンク境界、
  複数ワーカー、空入力、転送先不足、NaN scale、open 後のファイル切り詰めを検証。
- `globals.hyt` の 1,903,173,636 byte 全てを GPU から読み戻し、元ファイルと完全一致。
  層 0 / 3 の上表の常駐重みも全バイト一致。各層の比較ベンチマークも成功。
- CUDA/test 構成の Clippy は終了コード 0。既存 `executor.rs:1264` の `clone_on_copy` 警告が
  1 件あり、`-D warnings` ではそこで失敗する。今回の変更箇所に警告は出なかった。

実測ログ: `/mnt/NVME4TB/darask-inference-engine/local/hayate/spill/deepinfer-quant-research/loader-gpu.log`。
層のログは同じディレクトリの `loader-layer0-bench.log` と `loader-layer3-bench.log`。
再実行は `HAYATE_UPLOAD_BENCH_HYT` に globals または層の HYT を指定し、共有ロック下で
`cargo test --offline --release -p hayate-models --features cuda glm53_streamed_upload -- --ignored --nocapture --test-threads=1`。
層の試験では production と同じく `mlp.experts.*` を除き、常駐重みだけを比較する。

### 量子化の調査を踏まえた次の優先順位

| 手法・一次資料 | 得られるアイデア | Hayate での条件 |
|---|---|---|
| [GSQ](https://github.com/IST-DASLab/GSQ) + [RCO](https://github.com/IST-DASLab/RCO)、[HF の公開モデル](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF) | scalar の離散コード・scale 最適化と、容量制約下での精度配分を分ける | 最優先の研究候補。既存 LOWBIT カーネルで計算できる grid に制約して学習する。HF の別モデルの結果を GLM に外挿しない |
| [EXL3](https://github.com/turboderp-org/exllamav3) | QTIP 系 trellis と self-calibration | 既存 sc260 を品質基準にする。現 loader は同一層内の gate/up、down それぞれで bitrate 一様という制約がある。expert ごとの自由な配分には runtime 拡張が必要 |
| [HQQ](https://github.com/dropbox/hqq) | calibration-free の scale/zero 最適化 | 簡易比較の候補。ただし affine zero-point や grouping をそのまま現 LOWBIT 形式へ流せない |
| [HIGGS](https://huggingface.co/docs/transformers/main/quantization/higgs) / [FLUTE](https://github.com/HanGuo97/flute) | Hadamard で外れ値をならし、非一様 grid を使う | Hayate には回転と非一様 grid の実験が既にある。新規手法として二重計上せず、同じ校正データで比較 |
| [SINQ](https://huggingface.co/docs/transformers/main/quantization/sinq) | Sinkhorn 正規化による calibration-free 量子化 | scale の適用位置と追加メタデータを含めた runtime 費用の評価が必要 |

既存の `.worktrees/glm53q` には GPTQ/Hessian-weighted の LOWBIT 変換・配分と held-out 評価がある。
まずその校正資産を再利用して GSQ 型の精度改善を試す方が、新しい形式を一から増やすより比較しやすい。
RCO の task-loss 最適化と、重み MSE を使う簡易 knapsack は別物として記録する。

最初の量子化試作は、次の範囲に絞るのが妥当と判断した（未実装・未評価）:

1. 既存の層 3 / 24 の校正・held-out activation で、GPTQ 初期値から離散コードと scale を
   Gumbel-Softmax で調整する。目的関数は重み MSE だけでなく、その層の出力再構成誤差とする。
2. 学習中から Hayate の固定 LOWBIT grid、FP8 group scale + FP32 tensor scale の復号値を使う。
   GSQ 標準の整数 grid をそのまま HYT に保存しても、現カーネルと互換になるわけではない。
   まず同じ grid / group / 容量の既存 GPTQ と比較し、変換方式だけの効果を切り分ける。
3. その後、層ごとの gate/up と down を配分単位にする。group size を固定し、2/3-bit の候補を
   同じ teacher の loss で選ぶ。scale・alignment・dense 重みの実バイト数を予算に含め、
   最終 HYT サイズで再確認する。bpw だけの制約を実容量の保証として扱わない。
4. sc260 EXL3 と同じ評価セットで比較し、同品質で容量を下げられるかを判定する。
   GPU miss 時の CPU 復号時間も計る。GPU だけ速くても CPU オフロードを悪化させる候補は採用しない。

DeepInfer の FP4/FP8 KV も確認した:
[cache_common.py](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/layers/tilelang_ops/cache_common.py)、
[store_kv_cache.py](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/layers/tilelang_ops/cuda/store_kv_cache.py)。
Hayate の 45 層中 11 層の latent KV は 512 次元 BF16。131,072 token で合計 1.375 GiB。
同じ要素数を FP8 + 32要素ごとの1-byte scale とする理論値は約 0.709 GiB、FP4 + 16要素ごとの1-byte scale は約 0.387 GiB。
削減はそれぞれ約 0.666 / 0.988 GiB にとどまる（他のキャッシュ、作業領域、追加の scale や整列は含まない）。
DeepSeek の KV 分布・精度評価は GLM に適用できないため、まず expert の品質/容量/CPU miss 費用を改善し、その後に KV を扱う。

採用判定には、同一ソースの固定 teacher-forced KLD・top-1・NLL、日本語/コード/長文/ツールの held-out 品質、
HYT 実バイト数・VRAM/RAM・PCIe 転送量、short/code1k/long32 の TTFT と decode を使う。
校正時の重み誤差だけで「性能維持」とは判定しない。
