# GLM-5.3 / Flash の量子化調査と採用

調査日: 2026-10-20 UTC。公開元のモデルカード・実装・実験報告を確認した。
以下の公開性能は各作者の報告であり、Hayate での再現値ではない。

## 作業の統合状況（2026-10-21 JST）

ユーザー指定により、Hayate の作業をこの量子化調査を行っているチャットへ統一する。
引き継ぎ元の [DEEPINFER.md](DEEPINFER.md)、同じ作業ツリーのソース、保存済み実測ログを照合した。
対象は `/mnt/NVME4TB/darask-inference-engine/engines/hayate` の `main`、照合時の HEAD は
`66ae72f5a8754412aa17e6e751dfbc36b0f4faa3`。既存の未コミット変更・削除・未追跡ファイルを保持する。

**GPU はユーザー指示で一時停止中。明示的な再開指示まで GPU ジョブを開始しない。**
再開後も [GPU_SHARING.md](GPU_SHARING.md) の共有ロックを使う。CPU での調査・設計・小規模検証は継続可能。

| 項目 | 引き継いだ状態 | 確認できた範囲 |
|---|---|---|
| DeepInfer main / PR 調査 | 完了 | 詳細・固定 SHA・採否は `DEEPINFER.md`。PR snapshot を保持 |
| CPU radix sampler | 実装・既存検証完了 | `crates/llm/src/sample.rs` に実装あり。候補列・抽選一致と単体計時は引き継ぎ資料の記録 |
| HYT 二重バッファローダー | 実装・実機検証完了 | `load.rs`、`HytFile::read_tensor_range`、GPU tests と保存ログを照合。globals 約 2.36 倍、1,903,173,636 bytes 全一致。モデル全体の起動・生成速度の値ではない |
| NVFP4 `mse-ge1` | 実装・CPU 重み評価完了 | 27 行列で RTN 比 SSE 約 20.8% 減、追加容量 0。FP8 原本の復元重みを使用。モデル品質・GPU 計時は未評価 |
| 既存 EXL3 sc260 | importer・CPU/CUDA 推論を実装済み | 比較基準として保持。独自 EXL3 encoder は未実装 |
| GSQ 型 LOWBIT 改善 + RCO 型配分 | 設計段階 | `.worktrees/glm53q` の既存 GPTQ / 校正・held-out 評価コードは参照資産。GSQ/RCO を実装済みと数えない |
| EXL3 / Gyro の独自量子化器 | 設計段階 | Kogarashi の TQ2_T encoder と検証記録を参照。Hayate 側の encoder・TQ2_T runtime はまだ作成していない |

ローダー実測は `/mnt/NVME4TB/darask-inference-engine/local/hayate/spill/deepinfer-quant-research/loader-gpu.log` と
`loader-layer0-bench.log` / `loader-layer3-bench.log`、NVFP4 実測は
`python-tools/tests/glm53_nvfp4_scale_search_measurements.json` を参照する。
この統合作業ではログ・コードを照合し、GPU 試験やモデル全体の品質試験は再実行していない。
別時点・別範囲のテスト件数は合算しない。引き継ぎ資料にある既存 Clippy 警告も未解消の記録として保持する。

以下の各調査時点の優先候補を、末尾の「GSQ・RCO・EXL3・Gyro を組み合わせる設計」に統合する。
完了済みのローダーを作り直す工程は置かず、校正・符号化・配分・推論を分けて改善する。

## 結論と対象

容量を最優先する現行の `glm53x3-sc260`（EXL3 self-calibrated 2.60 bpw）を品質比較の対象として残し、
今回は **既存 NVFP4 経路に、容量を増やさず重み誤差と過度なクリッピングを抑えるスケール探索**を追加した。
`--scale-search mse-ge1` で明示的に使う。モデル全体の品質ゲートを通した既定値変更ではない。

GLM-5.3 と Flash は同一アーキテクチャではない。
[GLM-5.3 の公式 config](https://huggingface.co/zai-org/GLM-5.3/blob/main/config.json) は
78 層・hidden 6144・256 experts・RoPE ありの DSA、
[Flash](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/config.json) は
45 層・hidden 4096・288 experts・KDA と NoPE sparse MLA の混合。
現 Hayate の実行対象は Flash。GLM-5.3 の量子化手法は参考にするが、重みや Hopper カーネルをそのまま読み込む実装にはしない。

## 公開実装の比較

| 出典 | 工夫・確認できたこと | Hayate への判断 |
|---|---|---|
| [turboderp / GLM-5.3-Flash-exl3](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3) / [ExLlamaV3](https://github.com/turboderp-org/exllamav3) | 通常 2.05/3.05/4.05 bpw、self-calibrated 2.10/2.60/3.10/4.10 bpw を公開 | EXL3 直接実行は導入済み。まず現行 sc260 と同じ teacher/corpus で比較する。再量子化で EXL3 の圧縮効果を失わない |
| [Murai の NVFP4 レシピ](https://github.com/Murai-Labs/GLM-5.3-Flash-NVFP4-Murai/blob/main/docs/RECIPE.md) | 重み MSE 改善後に行動評価が悪化し、スケール縮小を禁止する実験で一部回復。重み誤差と行動品質を別々に評価 | **今回採用した考え方**。既存 HYT のまま `mse-ge1` を独立実装し、クリッピング量も計測。原実装の完全再現とはしない |
| [LibertAI NVFP4](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4) | routed experts のみ NVFP4、attention/shared/router/MTP などは高精度に保持 | Hayate の既存 expert-only 変換と整合。各行列固有の global scale を保存し、gate/up のスケールを混同しない |
| [PhalaCloud GLM-5.3-W4AFP8](https://huggingface.co/PhalaCloud/GLM-5.3-W4AFP8) | BF16 master から AWQ、coding-agent traces で校正。expert INT4 group128 + FP8 activation、Hopper の grouped GEMM | ソースを合わせ、用途に合う校正と未使用セッションでの評価を採用。SM90 の速度を SM120 に流用しない |
| [GPUStack GLM-5.3-W4A8](https://huggingface.co/gpustack/GLM-5.3-W4A8/blob/main/EVAL_DETAILS.md) | FP8 から再量子化し、non-expert は元バイトを保持。703.7 → 372.3 GiB。GPQA の小差をノイズとして扱い、NLL と MTP も評価 | 同じ親モデル・同じ engine での比較方針を参考にする。上記 AWQ の報告とは条件が異なる |
| [RTX PRO 6000 / EXL3 cache](https://github.com/SSHdotCodes/glm-5.3-flash-pro6000) | EXL3 3.05 bpw、固定 expert + GPU LFU cache + pinned RAM、MTP。先読みは demand miss と PCIe を競合して無効化 | Hayate も expert cache / CPU 分担を実装済み。PCIe 5.0・別 CPU の報告値を転載せず、CPU miss と H2D miss を分けて測る。prefetch の無条件導入はしない |
| [2× DGX Spark / NVFP4 / DFlash2](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark) | MTP、長文、SM121 の sparse MLA・スケール・indexer 等の修正を報告 | 参照すべきバグ分類。Hayate の SM120・独自 HYT に同じパッチを適用する根拠はない。MTP は受理率だけでなく総 latency を確認する |
| [REAP144-Coder](https://github.com/lapadatbogdan/GLM-5.3-Flash-REAP144-Coder) | router 重み × expert 出力ノルムの累積で 288 → 144 experts。Python agent tasks では良好だが、作者が算術・他言語・一般質問の劣化を明記 | **汎用既定モデルには採用しない**。頻度だけで expert を削除しない。日本語・一般推論も使う Hayate には領域外検証が必要 |
| [GSQ](https://github.com/IST-DASLab/GSQ) | GPTQ 初期化後に離散丸め先と scale を校正入力で学習し、scalar 低ビット形式を維持 | 次の容量削減候補。既存 `.worktrees/glm53q` の LOWBIT/GPTQ 実験を土台にできるが、現 main に GSQ 実装や学習済み重みは追加していない |
| [RCO](https://github.com/IST-DASLab/RCO) | 容量制約を保ちながらモデル損失で混合精度・pruning を探索 | 次の配分候補。単純な重み誤差順位や knapsack を RCO と呼ばない。Hayate の層内フォーマット制約とキャッシュ常駐バイトを目的関数に含める必要がある |

## 今回の変更

`glm53_nvfp4.quantize_expert` と `glm53_convert` に `--scale-search rtn|mse|mse-ge1` を追加。
従来の既定 RTN と `--mse` は維持する。既存 RTN/MSE は変更前とのバイト一致も確認する。

`mse-ge1` の倍率候補は `1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.5, 1.75`。
E4M3 に丸めた **保存後の scale** と E2M1 復元値で各 16 要素ブロックの誤差を比較し、最良候補を保存する。
RTN を最初の候補に含め、同点では先行候補を残す。既存 MSE より探索上限も広げているため、比較差を
「クリッピング抑制だけの効果」と解釈しない。推論時の探索、追加メタデータ、新カーネルは不要。
`weights.json` / pilot 出力には方式と倍率候補を記録する。

倍率を 1 以上にしても FP8 scale の丸め・underflow によるクリッピングは残り得る。
保証するのは **RTN より小さい保存 scale を選ばないこと**。
「クリッピングが完全にゼロ」「回答精度が保証される」とは呼ばない。

新しい `glm53_quant_study` は同じ実重みで複数方式・Hadamard block を比較する。
1 行列ずつ処理し、復元 F32 の SHA-256、ソース config の SHA-256、保存バイト数、F64 の重み誤差、
クリッピング量を JSON に記録する。`deep_clipped_blocks` は表現可能範囲を超える量がそのブロックの
最大絶対値の 10% より大きいブロック数。回転時のクリッピングは回転後の基底で測る。
結果には `quality_gate: null` / `quality_status: not_evaluated` を明記する。

## 容量

ローカル dealignai FP8 の実ヘッダから、45 層・MTP なし・dense は元の形式で算定。
HYT ヘッダ・alignment を含む論理ファイル長。VRAM/RAM 常駐量とは別。

| 形式 | HYT bytes | 十進 GB |
|---|---:|---:|
| FP8 | 319,788,591,104 | 319.79 |
| NVFP4 RTN | 186,607,781,888 | 186.61 |
| NVFP4 mse-ge1 | 186,607,781,888 | 186.61 |
| 現行 EXL3 sc260（実ファイル長） | 106,130,415,632 | 106.13 |

今回の探索は既存 NVFP4 に対して **追加 0 bytes**。NVFP4 の FP8 比約 41.6% 削減は既存形式の効果。
現行 EXL3 より小さくなる変更ではない。EXL3 は親モデル・dense の量子化条件も異なるため、容量表だけで品質を比較しない。

## 実重みでの測定

ローカル `GLM-5.3-Flash-dealignai-FP8` の layer 3/23/44、expert 0/143/287 の
gate/up/down、計 27 行列（約 2.26 億 weights）。全 expert を代表する無作為抽出ではない。
CPU i9-12900KS・4 threads・PyTorch 2.13.0、合計 174.6 秒。
相対 L2 は各行列の二乗誤差と重みエネルギーを合算してから平方根を取った値。

| 方式 | 相対 L2（回転なし） | 深いクリップのブロック数 | 相対 L2（Hadamard 128） |
|---|---:|---:|---:|
| RTN | 0.0919933 | 0 | 0.0951468 |
| 従来 MSE | 0.0819480 | 922,171 | 0.0860115 |
| mse-ge1 | **0.0818689** | **0** | **0.0851918** |

回転なしの RTN 比で相対 L2 は **11.0% 減**、二乗誤差は **20.8% 減**。
27/27 行列で RTN より二乗誤差が小さかった。各方式の payload 合計は同じ **127,402,092 bytes**。
Hadamard 128 でも RTN 比の二乗誤差は 19.8% 減、追加容量なし。
このサンプルでは回転なしの方が小さい誤差だが、モデル全体の品質順位とは解釈しない。

[測定要約と入力ハッシュ](../python-tools/tests/glm53_nvfp4_scale_search_measurements.json) を保存した。
行列別の詳細は `/mnt/NVME4TB/darask-inference-engine/local/hayate/spill/glm53-quant-research/nvfp4-recipes-cpu.json`。
入力は FP8 の F32 復元値であり、公式 BF16 master や現行 EXL3 との品質比較ではない。

検証:

- GLM Python suite: 60 件中 58 件成功、CUDA 専用 2 件は未実施。
- 共通量子化 suite: 11 件中 10 件成功、CUDA 専用 1 件は未実施。
- 追加方式の保存 payload を既存 C++/CUDA 共通の CPU 復号器で読み、Python 復号値と一致。
- 回転・down FP8 保持・HYT 書き出し・非有限入力拒否・比較レポートを確認。
- 変更前実装との比較で、10 seeds × RTN/MSE の 20 payload がバイト一致。
- GPU 共有ロック取得は既存ジョブの使用中に 30 秒で timeout。GPU の直接試験と全モデルの
  KLD/NLL・生成品質・tok/s は今回未実施。チェックポイント全体の変換や既定モデルの差し替えも行っていない。

## 再現

リポジトリ直下で `nix develop .#tools --impure` に入り、実在するソースを明示する。
GPU の場合は [GPU_SHARING.md](GPU_SHARING.md) の `gpu_slot` を使う。

```sh
export PYTHONPATH=python-tools
python3 -m hayate_tools.glm53_quant_study \
  --source /mnt/NVME4TB/darask-inference-engine/local/hayate/src/GLM-5.3-Flash-dealignai-FP8 \
  --layers 3 23 44 --experts 0 143 287 --hadamard-blocks 0 128 \
  --device cpu --threads 4 --out /tmp/glm53-recipes.json

# まず全変換なしで容量を確認する。
python3 -m hayate_tools.glm53_convert \
  --source /mnt/NVME4TB/darask-inference-engine/local/hayate/src/GLM-5.3-Flash-dealignai-FP8 \
  --layers 45 --expert-quant nvfp4 --scale-search mse-ge1 --estimate

python3 -m unittest discover -s python-tools/tests -p 'test_glm53*.py'
```

全変換時は `--estimate` を外し、`--out` で新しい出力先を指定する。
`--pilot` で変換時間を先に測れる。`--hadamard-block` と `--down-fp8` も従来どおり併用できる。
モデル全体の採否は同一ソースの teacher-forced KLD/top-1/NLL と、日本語・コード・長文・ツール利用の
held-out 評価で決める。推論の TTFT・decode tok/s・RAM/VRAM と、思考が上限に達して回答を出せない比率も記録する。
重み MSE だけを根拠に既定チェックポイントを差し替えない。

## 追加共有: EXL3 の量子化器と Kogarashi の TQ2_T

ユーザー提供の [carpedm20/exllamav3 の README](https://github.com/carpedm20/exllamav3) と
[`fury/doc/exl3.md`](https://github.com/carpedm20/exllamav3/blob/fury/doc/exl3.md) を確認した。
README は Hessian の逐次計算、融合 Viterbi kernel、ビットレート指定の変換、明示的な中断再開を説明している。
形式資料では QTIP 系の procedural codebook と循環 trellis、出力層を含む実容量、
tokenization/scoring を統一した PPL 比較を重視する。
これは **独自 EXL3 エンコーダを持たない Hayate の次の参照先として有用**。
文書の確認だけでは `fury` 固有の改良、上流との差分、GLM-5.3-Flash 対応を確認したことにはならない。
README の他モデル向け変換時間・VRAM 数値は GLM の見積もりに流用しない。

Hayate の `glm53_exl3.py` は既存 EXL3 を HYT に取り込み、routed experts の trellis をそのまま保存する。
現在の GLM runtime は mul1 codebook、K=2/2.5/3/3.5/4 に対応し、層内の gate/up と down はそれぞれ
全 expert で同じ bitrate が必要。HYT の汎用 EXL3 parser が受け入れる範囲と GLM kernel の対応範囲は別。
容量削減の優先候補は、既存 EXL3 runtime を使える符号化・校正・層単位の配分改善とする。

Kogarashi からは Gyro TQ2_T と互換の独立 CPU 研究エンコーダが共有された。
以下のローカル資料を読み、後段の訂正を優先した:

- `/mnt/NVME4TB/darask-inference-engine/engines/kogarashi/kogarashi/quant_lab/tq2.py`
- `/mnt/NVME4TB/darask-inference-engine/engines/kogarashi/research/opus-rereview.md`
- `/mnt/NVME4TB/darask-inference-engine/engines/kogarashi/research/astra-final-review.md`
- `/mnt/NVME4TB/darask-inference-engine/engines/kogarashi/research/quant-integration-validation.json`

コード・テスト・再レビューの7ファイルと、検証記録が参照する5成果物を SHA-256 で照合し、すべて一致。
照合記録は `/mnt/NVME4TB/darask-inference-engine/local/hayate/spill/glm53-quant-research/kogarashi-evidence-audit.json`。
この作業では encoder 試験、249 GB のモデル全体 hash、GPU 試験を再実行していない。

| 項目 | Hayate の EXL3 | Kogarashi の TQ2_T |
|---|---|---|
| 保存形式 | 256 weights の trellis tile + 行列の suh/svh FP16 | 128 weights ごとに FP16 scale 2 bytes + path 32 bytes、2.125 bpw |
| 現時点の利用範囲 | GLM Flash の importer、CPU/CUDA 推論 | 固定 Gyro C decoder と互換の CPU 研究 encoder |
| 探索・評価 | 独自 encoder は未実装、13種の上流復号 fixture あり | 固定 scale/境界の F64 前進 DP、複数境界・scale、保存 FP16 と隣接値の再採点 |
| 相互利用 | 閉路の検証方法・保存精度での再採点を参考にできる | EXL3 とバイト互換ではなく、128状態への縮約も TQ2_T 固有 |

共有された Ornith BF16 72ブロックの SSE 改善（元基底67.31%、Hadamard基底66.94%）は
公開 placeholder に対する結果であり、EXL3/NVFP4 や作者の非公開 encoder に対する優位性ではない。
4境界探索の固定 scale コストが全128境界より11.21%大きい18ブロックの記録も保持する。
4境界を exact と呼ばず、全境界でも scale 全体やモデル品質の大域最適性は主張しない。

GLM へ応用する条件:

1. 同じ BF16/FP16 原本の少量サンプルで比較する。以前の dealignai FP8 復元実験を BF16 教師と扱わない。
2. importance は符号化後の基底の対角計量。元基底の imatrix 対角を回転後に流用せず、全ゼロ重要度を拒否する。
3. 重み側 `W S H` と入力側 `H S x` の両方を適用し、全利用経路とメタデータを検証する。
   Gyro の `prism.hadamard.*` を Hayate の既存 metadata や EXL3 の suh/svh と同一視しない。
4. CPU FP16 LUT と CUDA int8 LUT / q8_1 activation の数値差を別途測る。
5. TQ2_T の HYT tag・CPU/CUDA decode/prefill・校正・全層 coverage・品質ゲートは未実装。
   GGUF の単一3D tensor内で expert ごとに dtype を変える前提を置かない。
6. 外部コードの移植時は該当ファイルのライセンス・出典を確認する。
   今回は資料と契約の整理だけを行い、Kogarashi/Gyro/QTIP のコードを Hayate へ複製していない。

Kogarashi の共有は比較候補と評価資産として受け入れる。大規模変換・追加モデル取得・公開の指示とは扱わない。

## GSQ・RCO・EXL3・Gyro を組み合わせる設計

状態: **設計のみ、未実装・未評価**。ユーザーの「全部組み合わせて、いいとこどり」に対する統合案。
共通の校正・評価の上で、保存形式に合う最適化器を使い、最後にモデル全体の容量と品質で配分する。
個々の手法の改善率を足したり掛けたりした見積もりは行わない。

### 役割と実装の境界

| 要素 | 採り入れるもの | 適用先・境界 |
|---|---|---|
| [GSQ](https://github.com/IST-DASLab/GSQ) | 校正入力に対する出力再構成を目的に、離散割当と group scale を調整 | まず既存 LOWBIT の固定 grid に制約する。最終 FP8 group scale / FP32 tensor scale の保存・復号まで評価する |
| [EXL3](https://github.com/carpedm20/exllamav3/blob/fury/exllamav3/modules/quant/exl3_lib/quantize.py) | Hessian による誤差補償、回転・scale 正規化、trellis 符号化 | EXL3 encoder の候補。入力と出力の変換を一貫させ、既存 decoder の pack 契約を保つ |
| Kogarashi / Gyro TQ2_T | 閉路探索、複数 scale・境界候補、保存 FP16 と隣接値の再採点、CPU oracle | TQ2_T の候補と検証資産。閉路検証や保存後再採点の考え方は EXL3 にも使えるが、LUT・状態縮約・バイト列は別 |
| [RCO](https://github.com/IST-DASLab/RCO) | モデル損失で候補の配分を学習し、容量制約を保つ | 各 backend が生成・検証した候補を選択する外側の工程。局所 MSE の順位付けや knapsack 単体を RCO と呼ばない |

EXL3 の `fury` の `quantize.py` を追加で取得し、`block_ldl` / `ldlq` の誤差補償と
`regularize` 相当の入力・出力 scale / Hadamard の処理を確認した。取得ファイルの SHA-256 は
`88e9b9ee554fd68fa40ed804865f843ff9559052a6a6281ca4ee6af97eda56d3`。
可動 branch の単一ファイル確認であり、全 repository の固定・GLM 対応・実行検証ではない。
外部 encoder コードは Hayate へ移植していない。

GSQ の公開方式は scalar grid 上の独立した割当を対象とする。これを trellis の各 weight に直接適用すると、
隣り合う状態と閉路の制約を壊し得る。trellis 側では、校正損失を使う scale / 正規化の更新と、
有効な閉路を返す Viterbi 探索を交互に行う案を評価する。これは **GSQ から着想した別の拡張**であり、
GSQ の忠実な実装や既知の品質向上として扱わない。微分可能な経路探索は後段の研究候補に留める。

EXL3 と TQ2_T は代替の保存形式。同じ重みを GSQ → EXL3 → TQ2_T と順に再量子化しない。
各候補を同じ親モデルから生成して比較する。TQ2_T 互換出力を持つ CPU 研究 encoder と、
既存 EXL3 runtime に合う独自 encoder の両案を保持する。

### 共通の校正・評価契約

1. 親 checkpoint、tensor、shape、基底、校正 / 検証 / 最終評価の分割、復元値 SHA を記録する。
   過去の FP8 復元値と BF16/FP16 原本を区別する。既存 capture の親モデルとの一致は利用前に照合する。
2. 回転後の activation から統計を取る。元基底の対角重要度を回転後へそのまま渡さない。
   MoE は論理 expert ID、routing weight、gate/up と down の異なる入力分布を保持する。
3. 各候補には pack → 実 decoder → 出力誤差の評価を行う。連続 scale や未丸めの重みの損失で採用しない。
   有効な初期候補を残し、比較した同じ有限精度の目的関数で悪化する更新を捨てる。
   この非悪化は校正データ上の条件であり、未知の入力や生成品質の保証ではない。
4. 検証用データで候補を選び、最終評価は探索に使わない。小規模比較で重み誤差と routed-output 誤差、
   モデル比較で固定 teacher-forced KLD / NLL / top-1 と日本語・コード・ツール・長文を測る。
5. metadata / scale / alignment を含む保存 bytes と、展開・cache を含む常駐 bytes を別に記録する。
   GPU 再開後に prefill / decode、CPU offload 時の復号、PCIe、TTFT、RAM / VRAM を比較する。

### 配分を runtime に合わせる

現 EXL3 runtime は各層の全 routed expert について gate/up の bitrate と down の bitrate をそれぞれ一様に要求し、
同じ層で EXL3 と他の expert 形式を混在できない。初期の統合探索は **層ごとに形式 family を選び、
その中で gate/up と down の精度候補を選ぶ**単位に制限する。
LOWBIT の一部経路が細かい配分を許しても、この初期比較では共通の単位を使う。
TQ2_T は HYT tag・CPU/CUDA decode/prefill が未実装なので、現時点で実行可能候補には含めない。

RCO 公開実装の標準コストは parameter count × 共通 bitwidth 候補。
形式ごとに scale や alignment が異なる本設計の **group × option の実バイト数**は、そのまま同じコスト式にならない。
導入時にはコスト表・勾配・予算への射影・離散化を対応させて検証する必要がある。
ヘッダや高精度で保持する tensor などの固定費も予算から差し引く。
指定上限にぴったり一致する離散配分が存在するとは限らないため、上限以下の実現可能な予算を選び、
最終 HYT の実サイズで確認する。byte 単位 DP の計算量も見積もり、bpw の一致だけで厳密予算達成と呼ばない。
VRAM や latency を同時に厳密制約化できるとは仮定せず、まず保存容量の制約と実行時の採否基準を分ける。

### 次の作業と比較順

まず既存 LOWBIT 校正・評価コードを現 main と照合し、上記の共通入出力・測定契約を用意する。
その上で LOWBIT は GPTQ → GSQ 型調整、trellis は重み誤差のみ → activation 校正 → 保存精度再採点、
という比較を同じデータで行う。Kogarashi の oracle と既存 EXL3 fixture をそれぞれの形式の検証に使う。
有効な候補が揃ってからモデル損失を使う配分工程を接続する。

現 sc260 を比較基準に保持し、改善は「同品質で小容量」または「同容量で高品質」を実測して判断する。
この設計記録の追加時点では、新しい量子化器のコード・新規モデル・品質改善値は存在しない。
