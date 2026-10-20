# チェックポイントの形式（2026-10-21）

Tsumuji が読める Qwen3.8-Flash-Next の量子化版と、その中身。どれも `tsumuji_engine::open` に
ディレクトリを渡すだけで、形式はテンソル名と dtype から判別する（`crates/tsumuji-engine/src/weights.rs`）。

| 版 | 置き場 | routed experts | dense | PLE の表 | MTP |
|---|---|---|---|---|---|
| RadixArk NVFP4（既定） | NVMe `hayate/src/Qwen3.8-Flash-Next-NVFP4` | NVFP4（丸め） | BF16 → 読み込み時に MXINT8 | FP8（1行 160 B） | BF16 → 読み込み時に NVFP4 / MXINT8 |
| nvidia FP4-Local（QAD） | SSD16TB `hayate/src/nvidia-Qwen3.8-Flash-Next-FP4-Local` | NVFP4、スケールは CSF 圧縮 | MXFP8（学習済み、そのまま使う） | NVFP4（学習済み、1行 80 + 10 B） | experts NVFP4、ほかは BF16 |
| RedHatAI NVFP4 | SSD16TB `hayate/src/RedHatAI-Qwen3.8-Flash-Next-NVFP4` | NVFP4（丸め、compressed-tensors） | BF16 | BF16（102 GB） | BF16 |

## NVFP4 の復元

値 = e2m1 コード × e4m3 ブロックスケール（16 要素ごと）× 大域スケール。コードは 1 バイトに 2 つで、下位 4 ビットが先。

- RadixArk、nvidia（ModelOpt）：大域スケールは `weight_scale_2`（そのまま掛ける）。
- RedHatAI（compressed-tensors）：`weight_global_scale` で割る。BF16 原本との照合で確認した（掛けると誤差が 1e8 倍になる）。

## nvidia の CSF（スケールの可逆圧縮）

`weight_scale.nvfp4_csf_fixed`（U8 [rows/16][16 + 16 · ns/2]）と `weight_scale.nvfp4_csf_exceptions`（U32 [m]）の2つ。
公開の仕様が無いので、BF16 原本との照合で逆算した（`csf_decode`）。

- 16 行ごとに、先頭 16 バイトが各行の基準値（e4m3 のバイト）。続いて各行の 4 ビット値が ns/2 バイト（下位が先）。
- スケールのバイト = 基準値 + 4 ビット値。
- 例外の各要素は `バイト << 24 | 通し番号`（行優先の [rows][ns] での位置）で、その位置のバイトを直接与える（4 ビットに収まらない値）。

照合：4 層 × 3 expert × 3 射影で、BF16 原本との相対二乗誤差が 0.0092（例外を無視すると 0.017、基準のずらし方を変えると 0.015 以上）。

## 量子化の誤差（BF16 原本との相対二乗誤差、experts）

| 版 | 誤差 |
|---|---:|
| RadixArk | 0.00897 |
| RedHatAI | 0.00898 |
| nvidia QAD | 0.00918 |

RadixArk と RedHatAI はほぼ同じ丸めなので、RedHatAI に替える意味は無い。nvidia は重みそのものではなく出力を元のモデルに合わせる学習（QAD）なので、重みの誤差は同程度でも出力の質は上がる見込み。KL で比べる（docs/m3.md）。

## 品質の実測：BF16 原本の FP32 参照との KL（2026-10-21、Tsumuji で各版を読み込み）

| 版 | ja | en | code | long | 平均 |
|---|---:|---:|---:|---:|---:|
| RadixArk（既定：experts NVFP4、PLE FP8、dense MXINT8） | 0.0918 | 0.0598 | 0.0850 | 0.0623 | 0.0747 |
| nvidia FP4-Local そのまま（experts QAD、PLE NVFP4、dense MXFP8） | 0.1045 | 0.0621 | 0.1121 | 0.0714 | 0.0875 |
| 組み合わせ：nvidia の experts ＋ RadixArk の残り（`TSJ_EXPERTS`） | 0.0889 | 0.0562 | 0.0906 | 0.0593 | 0.0738 |

- nvidia の PLE の表は 4 ビット（元との相対二乗誤差 0.009、RadixArk の FP8 は 0.0007）、dense は MXFP8（MXINT8 より KL が大きいと分かっている形式）。QAD はそれを学習で補っているが、元のモデルへの忠実さでは RadixArk に及ばない（正答率のような指標は別）。
- QAD の experts だけを使っても、RadixArk の丸めと差は 1% 程度。experts の量子化を変えない限り KL はこれ以上下がらない。次の手は計画 M6 の自前の量子化（GPTQ 型、効く層だけ高精度）。
- 既定は RadixArk のまま。nvidia 版は SSD16TB に置いたまま（使わないので NVMe に移さない）。RedHatAI 版も同じ。

## orcarouter Qwen3.8-Flash-Next-Uncensored-NVFP4（2026-10-21 追加）

置き場：NVMe `hayate/src/orcarouter-Qwen3.8-Flash-Next-Uncensored-NVFP4`（Hugging Face のゲート付き。ユーザーのトークンで取得）。

| 部分 | 形式 | Tsumuji での扱い |
|---|---|---|
| routed experts | NVFP4（compressed-tensors：`weight_packed`、`weight_scale`、`weight_global_scale`） | そのまま。大域スケールは割る（RedHatAI と同じ） |
| attention（q/k/v/o）、GDN の in_proj_qkv / in_proj_z / out_proj、shared expert | FP8 e4m3 ＋ 行ごとの BF16 スケール | 読み込み時に BF16 に戻して MXINT8 に変換（`fp8_to_bf16`） |
| GDN の in_proj_a / b、indexer、router、GR、lm_head、埋め込み、MTP | BF16 | 既定どおり（MXINT8 / NVFP4 に変換） |
| PLE の n-gram 表 | BF16（1 ファイル 102 GB） | 元のモデルとバイト単位で同一（2 シャード × 64 行で確認）。ダウンロードせず、`tsumuji.json` の `ple_from` で RadixArk 版の FP8 の表を読む |

- `config.json` の層の種類は `qwen_sparse_attention`（新しい transformers の表記）。Tsumuji は `linear_attention` 以外を QSA とみなすので、そのまま読める。
- 埋め込み（embed_tokens）は元のモデルと異なる（この版で変更されている）。
