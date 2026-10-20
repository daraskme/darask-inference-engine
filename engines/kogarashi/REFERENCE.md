# Kogarashi

Ornith-1.5-122BをRTX PRO 6000 Blackwell 96 GB / DDR4 128 GBで動かすための実験プロジェクトです。
現在のテキスト推論バックエンドは、コミットを固定したAgentionAI版llama.cppです。SM120a向けにビルドしています。
独自のTQ2_T CPU量子化試作を4段階のレビュー後に統合しました。公開デコーダと互換の34バイトブロックを生成します。
全モデル変換とCUDAエンコーダは未実装で、公開されていないGyro作者のエンコーダーを再現したものではありません。

## 起動

NixOSで、リポジトリのルートから実行します。

```sh
nix develop ../.. --command python -m kogarashi doctor
nix develop ../.. --command python -m kogarashi serve
```

取得・SHA256検証済みの `models/Ornith-1.5-122B-Q4_K_M.gguf` を使用します。
APIは `http://127.0.0.1:8080/v1`、モデル名は `kogarashi-ornith` です。
既定は8Kコンテキスト、GPUへの全層配置、Q8 KV、Flash Attention、8スレッド、DirectIOロードです。
`--dry-run` で実行する引数を確認できます。別のGPU処理と同時にモデルをロードするとVRAMが不足します。

```sh
nix develop ../.. --command python -m kogarashi serve --mtp on --draft-max 1
```

日本語greedyの不一致はCUDAの短いバッチで計算順序をそろえて修正しました。
公式Q4_K_M、Q8 KV、ドラフト長1では、日本語3課題（約2Kトークンの入力を含む）・コード・JSONの
各512トークン生成を3反復し、通常生成と全トークンIDが一致しました。MTPの実際の採用も確認しています。
通常生成側の計算順序も変わるため、修正前の出力との一致は保証しません。
MTPは既定offです。任意の長文、チャット、ツール呼び出し、別の量子化モデルに対する同等性は未検証です。
修正と検証範囲は `research/mtp-fix-validation.json` を参照してください。

## 実測

以下はMTPのCUDA修正前の測定です。公式Q4_K_M、RTX PRO 6000 Blackwell、i9-12900KS、CUDA、3反復。

| 条件 | 結果 |
|---|---:|
| llama-bench、pp512、MTPなし | 平均2150 tok/s |
| llama-bench、tg128、MTPなし | 平均142 tok/s |
| 短い日本語・コード・JSON、256トークン、MTPなし | 課題別平均137–138 tok/s |
| 同じ課題、MTPドラフト長1 | 課題別平均161–177 tok/s |

MTPの比較は連続した別実行で、交互実行による統計的な比較ではありません。ドラフト長3は70–150 tok/sとばらつきました。
モデルのロード時間、長文性能、品質の同等性を上の数値から推定しないでください。
原データは `artifacts/official-q4-bench.json`、`artifacts/mtp-comparison.json`、
`artifacts/mtp-draft1-results.json`、比較条件は `research/mtp-initial-comparison.json` にあります。

## ビルドと検証

```sh
bash scripts/fetch-sources.sh
nix develop ../.. --command bash scripts/build.sh
nix develop ../.. --command python -m unittest discover -s tests -v
```

ビルドはCUDA SM120aを対象にします。ソース取得は固定リビジョンと異なる既存チェックアウトを上書きしません。
`scripts/build.sh` は `patches/` の局所修正を確認して適用します。適用済みなら再適用せず、競合時は停止します。
推論テストは大きなVRAMを使うため、通常のCPUテストとは分かれています。
実機ではTQ2_T/TQK6/TQK7/TQP5/TQP6/TQP7のCUDA行列演算1,162件が通過し、未対応198件はスキップ扱いです。
GATED_DELTA_NETは36件通過しました。上流の許容誤差によるCPU参照との演算単位の比較であり、
全モデルの同値性や新しいCUDAエンコーダーの検証ではありません。条件とログhashは `research/backend-validation.json` にあります。

```sh
nix develop ../.. --command python scripts/benchmark_mtp.py \
  --draft-max 1 --load-mode dio --output artifacts/my-mtp-run.json
```

`--require-match --n-predict 512 --prompts japanese japanese_technical japanese_long code json` を加えると、
MTPの採用と全トークンID一致を検査します。単に文章が似ているだけでは合格しません。

ベンチマークは新しい実行ディレクトリに記録し、全測定が揃うまで成功扱いにしません。
既存の出力ファイルは上書きしません。モデルはローカルで実行し、評価データを外部へ送信しません。

## 量子化とレビュー

BF16原本249,362,980,832バイトを取得し、全体のSHA-256を公式LFS値と照合済みです。
証拠は `research/official-bf16-verification.json` にあります。Q4の再量子化は行っていません。
初期TQ2_T CPU試作に対する4段階レビューは完了しました。非有限参照値とCLIのoverflowを修正し、17 CPUテストが通過しました。

1. [Opus 5.5設計](research/opus-design.md)
2. [Astra敵対的検証](research/astra-review-1.md)
3. [Opus 5.5再検証](research/opus-rereview.md)
4. [Astra最終検証](research/astra-final-review.md)

後段の訂正を優先します。最終承認はCPU研究MVPの範囲です。CUDAの同期、枝刈りの誤差保証、全変換、モデル品質は別の検証が必要です。
Opusは実際の `claude-opus-5-5` を使用し、回答原本・モデルID・検証結果を `research/` に保存しています。
互換なバイト形式、モデル演算の正しさ、作者版と同等の品質は別々に検証します。
Hereticによる重み変更は未実施です。

```sh
nix develop ../.. --command python -m kogarashi.quant_lab \
  models/Ornith-1.5-122B-BF16.gguf --per-tensor 8 --basis both \
  --output artifacts/quant-lab-new-run.json
```

これは指定した少量の重みを圧縮して比較する実験です。モデルGGUFは書き出しません。
重要度は変換後の基底で指定し、全ゼロ重要度はエラーにします。無効な上流参照は理由付きで記録し、比較可能なペアの数を集計します。
NumPyはNix環境に含まれます。Pythonパッケージとして使う場合は `pip install -e '.[quant-lab]'` で追加できますが、固定バックエンドのビルドも必要です。
初期実験の手法と限界は [量子化実験記録](research/quant-lab.md) を参照してください。

完成原本の72ブロックで、公開placeholderに対する重み再構成の二乗誤差は元基底で67.31%、
Hadamard基底で66.94%減りました。元試作と入力SHA・出力バイトは全件一致しました。
モデルの回答品質、作者のGyroとの同等性、全層への一般化は未検証です。
別の18ブロックでは、保存スケールを固定した4境界探索の誤差合計が全128境界探索より11.21%大きく、
探索を減らすことによる損失も確認しています。
条件と原データへの参照は `research/quant-integration-validation.json` にあります。

### 追加のCPUスケール探索とGSQ/RCO統合案

EXL3の試行スケール探索を参考にした `kogarashi/quant_lab/scale_search.py` を追加しました。
既存エンコーダーの出力を保持し、対数間隔の初期スケールと局所精査から得た候補をCデコーダーで採点します。
保存したFP16スケールを含む34バイト形式は同じです。EXL3のコードブック・回転・LDLQは導入していません。
同じ重要度による重み再構成誤差が小さい場合だけ採用し、同点なら既存出力を返します。

既存72ブロックで入力ハッシュと既存エンコーダーの全出力バイトが一致しました。
別に事前固定した72ブロックでは、31ブロックが改善し、誤差合計が0.603%減りました。
144ブロックすべてで誤差悪化0、ベースラインと各試行の返却候補でC/Python復号のbit差0です。
holdoutのCPU変換時間は既存比3.68倍でしたが、単回の逐次比較であり推論速度の測定ではありません。
事前の採用閾値「誤差合計1%以上改善、変換時間5倍以内」には届かず、既定エンコーダーは変更していません。
原データは [CPU比較結果](artifacts/scale-search-v1/results.json)、事前条件は同ディレクトリの `manifest.json` にあります。
実験時のrunnerは同ディレクトリの `benchmark-source.py` に保存しました。レビュー後に件数・重複・標本分離の
検査を強化し、保存済みmanifestにも適用しています。量子化本体は変更せず、実験の再実行はしていません。
モデル全体のSHAは再計算せず、既存の検証済みsize/mtimeとcontinuityの標本SHAを確認しました。
このholdoutを見て探索方法を調整する場合は、新しいholdoutが必要です。

```sh
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  nix develop ../.. --command python scripts/benchmark_scale_search.py \
  --output artifacts/scale-search-new
```

[統合設計](research/combined-quant-design.json)では、GSQを既存スカラー形式の候補生成、
RCOを全体の損失と容量制限による候補選択に分けます。GSQ/RCO本体と全モデル変換は未実装です。
RCOの期待コストと、実際に保存・配置する整数バイト数の上限制約を別々に検査する必要があります。
GGUFの一つの3D expertテンソルは一つの型を持つため、expertごとに異なる保存型を選ぶにはruntime変更が必要です。
同じ型のままexpert別に探索量を変えることは可能です。EXL3のpayloadを既存Gyroデコーダーへ直接渡すことはできません。
重み誤差の改善を回答品質の改善とは扱いません。
複合設計と追加CPU試作のレビュー記録は、[Opus設計](research/combined-quant-opus-design.txt)、
[Astra検証](research/combined-quant-astra-review.txt)、[Opus再レビュー](research/combined-quant-opus-rereview.txt)、
[Astra最終監査](research/combined-quant-astra-final.txt)です。このCPU研究範囲の4段階レビューは完了しました。
設計txtは初期案の履歴で、訂正した `combined-quant-design.json` と後段レビューを優先します。

作業の引き継ぎは [統一した状態記録](research/session-unified-2026-10-21.json) を正とします。
次のセッション向けの本文は [現状と引き継ぎ](KOGARASHI_SESSION_CONTEXT.txt) にまとめています。
2026-10-21のユーザー指示により、本チャットからのGPU実行・照会は明示的な再開指示まで停止中です。

## 参照ソースとPR

- [AgentionAI llama.cpp](https://gitlab.com/agentionai-group/llama.cpp): MIT、`9aa51ce4fdd3435a91531ffe16a17414ccc6dc6f`
- [Strata](https://github.com/Niko1221/Strata): MIT、`cbfd5b926b794b356195d3cb600191b0c3567540`
- [DeepInfer](https://github.com/deepseek-ai/DeepInfer): Apache-2.0、`774ce244b87cc0eb8d4b94ab9f5de2ab22049f01`
- [Heretic](https://github.com/p-e-w/heretic): AGPL-3.0、任意の外部ツールとして検討中
- [QTIP](https://github.com/Cornell-RelaxML/qtip): GPL-3.0、`e90c6688c8dfae326a3a81b5eb032db7c6680ec0`。アルゴリズムの検証用参照で、コードをKogarashiへ複製していません。
- [ExLlamaV3 fury](https://github.com/carpedm20/exllamav3/tree/fury): MIT、`4700920a5d283a561406709ed0e27d9dc3c137f7`。実CUDA量子化・保存復元の部分行列検証は `research/exl3-validation.json`。
- [GSQ](https://github.com/IST-DASLab/GSQ): Apache-2.0、`03fc16484c369e3127225615d5e03e8d3a6043e3`。現在はREADMEとライセンスの調査段階。
- [RCO](https://github.com/IST-DASLab/RCO): Apache-2.0、`9a1e09c07d468109cbe60a1b87d5036034a79d10`。公開コードの設計検証段階。固定ソースとライセンスは `research/gsq-rco-sources/`。

PR差分の保存、採用判断、未検証の点は `research/reuse-decisions.json` を参照してください。
DeepInferの93件とStrataの直近60件のPR概要から候補を選び、11件の差分・適用条件を記録しています。
DeepInfer [#115](https://github.com/deepseek-ai/DeepInfer/pull/115) の結果分離と
[#102](https://github.com/deepseek-ai/DeepInfer/pull/102) の受理率検査は独立実装として取り込みました。
[#52](https://github.com/deepseek-ai/DeepInfer/pull/52) の転送バッファ案は既存バックエンドの経路を使い、
Strata [#3483](https://github.com/Niko1221/Strata/pull/3483) のストリームごとの完了待ちは将来のloader変更時の検証条件にしています。
DeepInferのSM100・DeepSeek専用演算をそのままSM120・Ornithに適用してはいません。
現在取り込んだものは評価の独立した実装と、既存バックエンドの非同期転送経路を使う設定です。
各ソースのライセンスは `upstream/` の対応するチェックアウトに保存されています。
