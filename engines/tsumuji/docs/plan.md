# Tsumuji 計画書：Qwen3.8-Flash-Next 専用の Rust + CUDA 推論エンジン

> 2026-10-20 にクラウドのセッションで作った計画書を、そのまま保存したものです（クラウド環境からは HF・arXiv に入れなかったため、GitHub 上の技術レポート、Transformers、SGLang などの実装コードをもとに書かれています）。
> このマシンで一次情報（実チェックポイント、Transformers 5.17 の `qwen4_exp`）と照合した結果、**いくつか修正があります。修正点と M0 の実測値は [m0.md](m0.md) にまとめてあり、食い違う場合は m0.md が優先です。**

一次情報（技術レポート、Transformers の `qwen4_exp` 実装、実際の `config.json`、SGLang の実装）で設計の前提を確認しました。そのうえで、RTX PRO 6000 1枚と DDR4 128GB の構成で SGLang を上回るための計画と実装案をまとめます。数値のうち「推定」と書いたものは、最初の2週間（M0）の実測で確定させます。

---

## 0. 前提

### 0.1 モデルの構造（実 config ベース）

| 項目 | 値 | 設計への影響 |
|---|---|---|
| 規模 | 本体125B（うち routed experts ≈121B）、1トークンで使うのは ≈6.6B。別に n-gram 表（PLE）51.2B と MTP ≈4B | experts を4bitにしないと96GBに入らない |
| 層構成 | 48層。GDN（線形アテンション）36層と QSA（疎アテンション）12層で、4層に1回が QSA。hidden 2560 | |
| GDN | QK 16ヘッド / V 48ヘッド、head_dim 128、conv 4、状態は FP32、出力ゲートは **sigmoid** | 状態は1層 3MiB、1シーケンス 108MiB |
| QSA | Q 24 / KV 2ヘッド（GQA 12）、head_dim 256、RoPE は先頭64次元のみ（θ=1e7）、出力ゲートあり。Indexer は Q4/K1ヘッド×128次元で、キーを4トークンずつ平均して圧縮し、上位512ブロックと末尾の未完成ブロックを選ぶ | KV は BF16 で 24KiB/トークン。1クエリが見るのは最大2051トークン |
| MoE | 512 experts、top-10 + shared 1（shared は sigmoid ゲート付き）、intermediate 640、`norm_topk_prob=true` | |
| Gated Residual（GR） | 残差を4本（10240幅）に広げる。各サブ層の前で、4本を別々に RMSNorm → 低ランク(320)の sigmoid ゲートで読み、本ごとのスカラーで書き戻す。最終 norm はない | pre-norm の代わり。小さい GEMV が97個ある |
| PLE（n-gram 埋め込み） | layer 2 だけ。2-gram と 3-gram をそれぞれ8個のハッシュで引き、16行×160次元を取る。EOS でリセット。dilated depthwise conv（k=4, d=3）付き | 1トークンで読むのは16行（FP8なら2.5KB）だけなので、表はホスト RAM に置く |
| MTP | 1層（QSA + MoE + GR）。4本の残差を入力に使う | 技術レポートでは4ステップ投機で平均受理長4.07 |
| 語彙 | 248,320。embedding と lm_head は別の重み | lm_head が1トークンの読み出しバイトの約1割を占める |

### 0.2 倒すべき数字（同じ RTX PRO 6000 1枚での実測）

| エンジン | 構成 | 1ユーザー decode | prefill |
|---|---|---|---|
| llama.cpp | Q5_K_M、48層中20層の experts を RAM | 約45 tok/s | 約380 tok/s |
| llama.cpp（MTP の PR 版） | UD-Q4_K_XL | 約170 tok/s（個人の報告） | – |
| vLLM 0.31 | NVFP4 + MTP k=3 | 127（chat）/ 195（code） | 約14K tok/s |
| **SGLang（Pennyroyal fork）** | NVFP4 + online MXFP8 + FP8 KV + MTP3 + FR-Spec | **179–188 / 281–287** | **約14K tok/s** |

Diffusers と ComfyUI は拡散モデル（画像生成）用で、このモデル（自己回帰の VLM）は動かせません。実際の比較相手は llama.cpp と vLLM、そして同じ構成で最も速い実測が出ている SGLang です。

---

## 1. 速度目標と根拠

decode の速さは、1トークンごとに読むバイト数を、メモリ帯域（1.79TB/s）で割った値でほぼ決まります。バッチ1・投機なしのときの内訳は次のとおりです。

| 部位 | パラメータ数 | 形式 | 1トークンの読み出し |
|---|---:|---|---:|
| routed experts（10/512 × 48層） | 2.36B | NVFP4（0.5625 B/param） | 1.33 GB |
| GDN の射影（36層） | 2.08B | MXFP8 | 2.15 GB |
| QSA の射影（12層） | 0.62B | MXFP8 | 0.64 GB |
| GR の低ランク射影（97個） | 0.64B | MXFP8 | 0.66 GB |
| lm_head | 0.64B | MXFP8 | 0.66 GB |
| shared expert / router / PLE 射影 | 0.33B | MXFP8 / BF16 | 0.40 GB |
| GDN 状態の読み書き、KV、indexer | – | FP32 / BF16 | 0.29 GB |
| **合計** | | | **約6.1 GB** |

- 理論値は 3.4 ms/トークン、つまり投機なしで約290 tok/s です。
- SGLang は MTP 込みで約185 tok/s なので、帯域の理論値に対しておよそ6割しか使えていない計算です（推定）。原因は、48層×約10個の依存したカーネルの間に生じる空白、W4A4 の活性量子化処理、ホストの介在だと見ています。

| 指標 | SGLang 実測 | Tsumuji 目標（M0 の実測で再設定） |
|---|---:|---:|
| decode、投機なし | – | 240 tok/s 以上（理論値の85%） |
| decode、MTP、chat | 185 | 300 以上 |
| decode、MTP、code | 285 | 420 以上 |
| prefill（8–32K トークン） | 14K tok/s | 30K tok/s 以上 |
| 4ユーザー合計 | 481 | 700 以上 |
| 品質 | W4A4（活性も4bit） | W4A16（活性は量子化しない）。KL ダイバージェンスを常時計測 |

---

## 2. 判断基準の具体化（高速 >= 質 >>> VRAM 節約 >>> シンプル）

- **高速**：このモデル専用にします。形状はすべてコンパイル時定数にして、汎用性は捨てます。
- **質**：「読むバイトは減らすが、演算の精度は落とさない」が方針です。
  - 重みだけを量子化し、復元（dequant）は誤差なしで行います。
  - decode では活性を量子化しません。
  - 投機デコードでは本体モデルが検証するので、draft 側の精度を下げても出力の品質は変わりません。
- **VRAM**：余った VRAM は品質と速度に回します（KV を BF16 で持つ、prefix cache、感度の高い層の experts を FP8 にする）。
- **シンプル**：最後に考えます。megakernel、独自のファイル形式、独自の FFI も採用します。

---

## 3. メモリ配置

### VRAM（95.6 GiB）

| 内容 | 形式 | サイズ |
|---|---|---:|
| routed experts（48層） | NVFP4（オフラインで MMA 向けに並べ替え済み） | 63.3 GiB |
| MTP 層 | experts は NVFP4、それ以外は FP8 | 約1.4 GiB |
| dense 部（GDN、attn、shared、GR、PLE 射影、lm_head） | MXFP8（router のみ BF16） | 約4.2 GiB |
| token embedding | BF16 | 1.2 GiB |
| vision encoder（遅延ロード可） | BF16 | 約0.8 GiB |
| KV（256K トークン分） | **BF16** | 6.0 GiB |
| GDN 状態（16スロット） | FP32 | 1.7 GiB |
| prefill 作業領域、CUDA context・graph | | 約6.5 GiB |
| **合計** | | **約85 GiB（約10 GiB の余り）** |

KV を BF16 にしても速度はほとんど落ちません。疎アテンションなので、decode で読む KV は1トークンあたり最大2051行、約50MB で、全体の1%未満です。

### ホスト RAM（128GB）

| 内容 | 形式 | サイズ |
|---|---|---:|
| PLE 表 | **FP8 E4M3**（BF16 だと102GB で入らない） | 47.7 GiB（pinned、2MB hugepage） |
| 転送用ステージングバッファ | | 1 GiB |
| prefix cache のホスト側（GDN 状態のスナップショット、KV） | | 24–32 GiB |
| OS ほか | | 残り |

OS は Linux を推奨します（WDDM のオーバーヘッドがなく、pinned メモリの上限もないため）。設定は `iommu=pt`、memlock 無制限、hugepage の確保です。

---

## 4. 精度の設計

| テンソル | 保存形式 | decode の演算 | prefill の演算 |
|---|---|---|---|
| routed experts | NVFP4（e2m1 + 16要素ごとの e4m3 スケール + テンソル単位の fp32） | **W4A16**：BF16 に誤差なしで復元して BF16 の MMA | W4A16（既定）/ W4A4（速度優先モード） |
| dense 射影 | MXFP8（e4m3 + 32要素ごとの ue8m0 スケール） | **W8A16** | W8A8（MXFP8） |
| router、shared_expert_gate、GDN の a/b 射影 | BF16 | BF16 | BF16 |
| GDN 状態 | FP32（config の指定どおり） | | |
| KV / indexer の圧縮キー | BF16（長文脈では FP8 も選べる） | | |
| 残差ストリーム | BF16（prefill のみ FP8 も選べる。論文でほぼ劣化なし） | | |
| PLE 表 | FP8 E4M3 + スケール | | |
| MTP の draft head | NVFP4、または頻出語だけに絞った FR-Spec | 出力品質には影響しない（本体が検証する） | |

**NVFP4 から BF16 への復元が誤差なしになる理由**：e2m1 の有効桁は2bit、e4m3 は4bit なので、積の有効桁は最大6bit で、BF16 の8bit に必ず収まります。そのため decode での誤差源は「重みを量子化した誤差」だけになり、同じバイト数の W4A4（SGLang、vLLM）より品質が上になります。

---

## 5. decode 経路（最優先）

### 5.1 1層ぶんのカーネル列（GDN 層の例）

| # | カーネル | 内容（融合するもの） |
|---|---|---|
| 1 | `gr_read_a` | 4本の残差それぞれの RMSNorm → down(10240→320) と inject(→4) を split-K して部分和を出す |
| 2 | `gr_read_b` | 部分和を固定順で合算 → silu(x/4) → up(320→10240) → sigmoid → ゲートを掛けて平均 → x(2560) |
| 3 | `gdn_in` | qkv、z、a、b を1本の連結 GEMV（2560→16480）で計算。後処理で conv4 + SiLU、q,k の L2 正規化、α、β |
| 4 | `gdn_rec` | delta rule の再帰（5.4 の遅延コミット方式） |
| 5 | `gdn_out` | gated RMSNorm(y)·σ(z) → out_proj。後処理で GR の書き戻し（R_i += s_i·o） |
| 6–7 | `gr_read_a/b` | MLP 側の読み出し |
| 8 | `router` | 2560→512 の BF16 GEMV（shared ゲート用の1行も連結）→ softmax → top-10 → 再正規化。**決まった直後に、選ばれた experts の L2 プリフェッチを発行** |
| 9 | `moe_fused` | routed と shared の gate_up → 活性化 → down を1カーネルで処理し、部分和を出す |
| 10 | `moe_combine` | 部分和を固定順で合算 → shared ゲート → GR の書き戻し |

### 5.2 カーネル間の空白を消す（ここで SGLang に差をつける）

1. **PDL（Programmatic Dependent Launch）**：重みは前のカーネルの出力に依存しません。そこで各カーネルは、`griddepcontrol.wait` より前に自分の重みを SMEM や L2 に読み込み始めます。これで、前のカーネルの終わり際とメモリ読み出しが重なります。
2. **CUDA Graph**：バッチサイズ（1〜8）と投機の段数 k ごとに、1ステップ全体をキャプチャしておきます。
3. **デバイス側ループ**：graph の while 条件ノードで、最大32ステップを GPU だけで回します。生成トークンは mapped pinned メモリのリングバッファに書き、ホストはそれを poll するだけにします。トークンごとのホスト同期はゼロになります。
4. **megakernel（M6）**：PDL を入れても理論値の85%に届かない場合に導入します。1ステップを1つの persistent kernel にして、依存関係をグローバルカウンタで管理します。待っている間も次の op の重みを L2（約128MB を想定、M0 で実測）に流し続けます。移行を機械的にするため、各 op は最初から `__device__` 関数として書き、通常カーネルと megakernel の両方から呼べる形にしておきます。

### 5.3 GEMV の設計

- **swap-AB**：重みを MMA の M（16行）、トークンを N（8列）に割り当てます。N≤8 までは無駄なく処理でき、k≤7 の検証までそのまま対応できます。
- **バッチ不変**：split-K の分け方は重みごとに固定し、トークン数で変えません。float の atomics も使いません。こうすると、投機の有無で出力がビット単位で一致します。
- **重みの事前並べ替え**：Marlin と同じ考え方で、オフラインで MMA のフラグメント順に並べておきます。これで SMEM を経由せず、16B のベクトルロードで直接レジスタに読めます。

### 5.4 MTP 投機デコード

- **GDN 状態の遅延コミット**：検証パスでは状態を書きません。受理された数は GPU 上に残しておき、次のステップの冒頭で受理分の更新をやり直してから書き込みます。状態の読み書きは1ステップ1回で済み、ロールバックも不要です（SGLang は1リクエストに5スロット使っています）。conv 状態と PLE の文脈も同じ方式で扱います。
- **QSA の圧縮ブロック**：k̄ は確定したトークンだけで作ります。投機トークンでブロックを確定させません。
- **draft の安さ**：draft は精度を下げても出力品質に影響しないので、NVFP4 の head を使います（FR-Spec を使うかどうかは受理長を見て決めます）。index は draft のステップ間で使い回します。
- **段数 k の自動調整**：MoE では、検証のコストが「同時に読む experts の種類数」に比例して増えます。受理率とこのコストから k を毎ステップ選びます。
- **コード編集向けの併用**：MTP に加えて、プロンプト内の文字列をコピーして draft にする方式（suffix automaton）を混ぜます。既存コードを写す編集では受理長が大きく伸びます。

---

## 6. prefill 経路

- **チャンク分割**：8–16K トークンずつ処理します。lm_head は最後のトークンだけ計算します。
- **dense**：MXFP8 の W8A8 GEMM（CUTLASS / CuTe の SM120 block-scaled MMA を使い、`sm_120a` 向けにビルド）。GR の読み書きは GEMM の前処理と後処理に融合します。
- **MoE**：トークンを expert ごとに並べ替えてから grouped GEMM にかけます。既定は W4A16（FP4 を SMEM 上で BF16 に復元）、速度優先モードは W4A4 です。どちらにするかは、M0 で測る MMA の実効性能と KL で決めます。どちらでも SGLang の2倍以上が見込めます（推定）。
- **GDN**：chunk 版の gated delta rule です（FlashQLA の TileLang アルゴリズムを CUDA C++ に移植）。
- **QSA**：
  - 先頭の約2051トークンは **dense と完全に一致する**ので、indexer を飛ばして通常の FlashAttention 型カーネルで処理します。
  - それ以降は、チャンクごとにスコアを VRAM に一度書き出し、radix select で top-512 を選び、gather して疎アテンションを計算します。
  - VRAM に余裕があるので、v1 はスコアを書き出す方式で作り、v2 で融合します。

---

## 7. PLE（n-gram 表をホストに置く設計）

- ハッシュは GPU で計算します。前の2トークンは GPU 上に保持します（EOS が来たら EOS,EOS にリセット）。
- 表の行は **GPU から pinned ホストメモリを直接読みます**（zero-copy）。CPU を介さず、CUDA Graph にもそのまま入ります。
- ステップの最初に別ブランチで読み始め、layer 0 の計算（約70µs）の間に終えます。読み出し時間は計算に完全に隠れます。prefill の 16K チャンクでも約42MB なので、同じく layer 0 の裏で終わります。
- PCIe Gen4 と Gen5 のどちらでも足ります。DDR4 の帯域は問題になりません。

---

## 8. Rust 側の構成

```
tsumuji/
  crates/
    tsumuji-cuda/     # Driver API の薄いラッパー（cudarc::driver::sys）: PDL, Graph, 条件ノード, HostRegister
    tsumuji-kernels/  # .cu を build.rs で nvcc -arch=sm_120a → cubin に。include_bytes! で埋め込み
    tsumuji-pack/     # safetensors → .tsj（GPU 上の配置そのままのレイアウト）。NVFP4/MXFP8 量子化、PLE の FP8 化
    tsumuji-model/    # Qwen4Exp の実行計画（層ごとのカーネル列、バッファ割り当て）
    tsumuji-engine/   # メモリ計画, KV ページ, GDN スロット, prefix cache, スケジューラ, 投機制御, サンプラ
    tsumuji-text/     # tokenizers crate + minijinja（chat template）+ qwen3 の reasoning/tool 用パーサ
    tsumuji-server/   # axum による OpenAI/Anthropic 互換 API（SSE）
    tsumuji-ref/      # 層単位で BF16 を流す参照実装（品質の基準）
    tsumuji-bench/    # マイクロベンチと E2E ベンチ
  kernels/{common,gemv,gemm,gdn,qsa,gr,ple,moe,sample,vision}/
```

- **エンジンスレッド**：専用の OS スレッドを1本 CPU コアに固定し、busy-poll で回します。tokio はサーバ側だけで使います。
- **メモリ割り当て**：起動時にすべて確保し、ホットパスではアロケーションしません。
- **ロード**：O_DIRECT で読み、ステージングバッファ2本を交互に使って DMA します。NVMe の速度で決まり、約70GB で十数秒の見込みです。

---

## 9. 具体的なコード案

（元の計画書のコード案 (a)〜(f) は、実装に取り込んだうえで次のファイルに置き換えました。）

| 案 | 実装 |
|---|---|
| (a) 形状はすべて定数 | `kernels/include/q38.h`、`crates/tsumuji-model/src/shape.rs`（両者の一致をテストで確認） |
| (b) PLE ハッシュの定数 | `crates/tsumuji-model/src/ple.rs`（transformers の出力とビット一致をテストで確認） |
| (c) PLE の gather | M1 で `kernels/ple/` に実装（M0 では `kernels/bench/zerocopy.cu` で読み出し性能だけ測定） |
| (d) PDL 付きの起動 | `crates/tsumuji-cuda/src/lib.rs` の `launch`（cudarc を使わず自前の FFI） |
| (e) GDN decode | M1（遅延コミットは [m0.md](m0.md) の修正点を反映） |
| (f) MoE の融合 decode | M1–M2 |

---

## 10. 正しさ・品質・速度の検証

1. **演算ごとの golden テスト**：Transformers の `qwen4_exp` を縮小した乱数 config（4層、hidden 256）で動かし、全演算の中間テンソルを fixture として保存します。GR、PLE（EOS リセットを含む）、GDN、QSA の選択インデックス、MoE が対象です。CUDA カーネルの単体テストはこれと照合します。
2. **実モデルの参照**：`tsumuji-ref` は BF16 の重みを1層ずつ NVMe から GPU に流して実行します。1層は約5GB なので載ります。これで評価用プロンプト全体の参照 logits を作り、各トークンの KL、top-1 一致率、PPL を毎回のコミットで測ります。
3. **タスク評価**：GSM8K、HumanEval+、MMLU-Pro の一部、RULER（64K/128K、QSA の検証用）、ツール呼び出しの2段階テストを使います。
4. **速度**：同じマシンで llama.cpp、vLLM、SGLang を、同じプロンプト（毎回ユニーク）で測ります。makieali のベンチスクリプトと同じ条件で、chat、code、TTFT（1K/8K/32K/128K）、4ユーザーを比較します。

## 11. マイルストーン（1人で専念した場合で約5–6か月）

| 段階 | 期間 | 完了条件 |
|---|---|---|
| M0 計測と下調べ | 1–2週 | §13 の結果と golden fixture |
| M1 正しく動く | 4–5週 | バッチ1・投機なしで参照との KL が予算内 |
| M2 decode 高速化 | 3–4週 | PDL + Graph + デバイス側ループで、投機なし 240 tok/s 以上 |
| M3 MTP | 2–3週 | chat 300 / code 420 tok/s 以上 |
| M4 prefill | 4–6週 | 32K トークンで 30K tok/s 以上、RULER が参照と同等 |
| M5 サーバ化 | 2–3週 | API、chat template、ツール呼び出し、prefix cache、複数ユーザー |
| M6 仕上げ | 継続 | megakernel、自前の GPTQ 型 NVFP4 量子化、感度の高い層の experts を FP8 に、vision、YaRN |

## 12. 実装上の落とし穴（既存実装が実際にはまった点を含む）

- GDN の出力ゲートは **sigmoid** です。Qwen3-Next 用の SiLU カーネルを流用すると、36層すべてが誤りになります。
- RMSNorm は `(1+w)` 形式です。例外は GDN の gated norm だけです。
- 最終 norm はありません。`hyper_connection_mixer` が readout を兼ねます。
- `config.json` に `norm_topk_prob` が書かれていないので、既定値の true を明示的に使います。
- GQA が12グループです。汎用のアテンションカーネルが対応している形の外です。
- PLE の conv は dilation=3 です。通常の causal conv1d カーネルは使えません。
- PLE 表は BF16 だと102GB です。128GB RAM では FP8 にするしかありません。
- indexer のキー圧縮は「平均 → RMSNorm → ブロック先頭の位置で RoPE」の順です。
- MTP の入力 norm は10240幅全体にかける RMSNorm で、4本別々ではありません。`fc_hidden` は4本に同じ重みを使います。

## 13. リスクと対策

| リスク | 対策 |
|---|---|
| SM120 用のツールがまだ発展途上（tcgen05/TMEM がなく、mma.sync と TMA が中心） | CuTe の SM120 atom を使い、足りない部分は PTX を直接書く。M0 で実効性能を実測する |
| 受理長が SGLang 並み（約2.1）にとどまる | k の自動調整と、prompt-lookup 型 draft の併用 |
| ホストメモリの zero-copy 読み出しが IOMMU や TLB で遅い | `iommu=pt`、hugepage、駄目なら CPU で gather して DMA する方式に切り替え |
| NVFP4 の品質 | M1 は RadixArk の NVFP4 をそのまま使う（W4A16 で使えば同じ重みでも品質が上がる）。M6 で自前の GPTQ 型量子化 |
| ディスク容量 | RadixArk の NVFP4 が186GB、変換後が約115GB。参照用の BF16 は360GB（あれば理想） |

## 14. 最初の2週間（M0）でやること

1. Ubuntu、オープンカーネルモジュール版ドライバ、CUDA 13.x、Rust stable をそろえ、`sm_120a` 向けにビルドできることを確認します。
2. `tsumuji-bench` で次を実測します。
   - (1) 単純なストリーミング読み出しで、帯域の何%まで出るか
   - (2) graph 内で PDL を使ったとき、カーネル間の遅延がどれだけ縮むか
   - (3) 2560×1280 の W4A16 GEMV で出る帯域
   - (4) 48GB の pinned 領域から 160B の行をランダムに zero-copy で読んだときのレイテンシと帯域
   - (5) L2 の実効サイズと、共有メモリの上限
   - (6) FP8、FP4、BF16 の MMA 実効性能（prefill を W4A16 と W4A4 のどちらにするか決めるため）
3. RadixArk の NVFP4 チェックポイントとトークナイザを取得し、`tsumuji-pack inspect` でテンソル一覧を config と突き合わせます。
4. 縮小 config の golden fixture を生成し、カーネルの単体テストの土台を作ります。

---

**元の計画書が参照した資料**

- QwenLM/Qwen3.8-Flash-Next（README と tech_report.pdf）・arXiv 2608.30320
- Transformers `qwen4_exp` 実装
- Qwen/Qwen3.8-Flash-Next config.json
- SGLang（qwen4_exp、qwen4_exp_mtp、QSA カーネル）
- makieali/qwen38-flash-next-1x-rtx-pro-6000
- NVIDIA Developer Forums: 1× RTX PRO 6000 で 171 tok/s
- Unsloth GGUF の MTP に関する議論、pegainfer ロードマップ #1103、MaxText PR #5729、llama.cpp b10660 リリースノート、LMSYS day-0 ブログ、FlashQLA
