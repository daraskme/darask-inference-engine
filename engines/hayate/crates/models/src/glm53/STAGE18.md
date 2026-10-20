# 第18段階: 実行速度トラック(phase18/glm53r)

基点は main(ef13e96)に第12〜13b段階(607a420)を merge したもの。

## 変更点

### CPU エキスパート(decode のミス計算)

- `crates/kernels/src/cpu/glm53.rs` を書き直した。
  - 形式は `Format`(`Nvfp4` / `Fp8Block`)で選ぶ。新しい形式(量子化トラックの LOWBIT Q2/Q3)は
    「1 行 × 最大 4 活性行の内積を返す行カーネル」を 1 つ足せば、プールと層単位の呼び出しはそのまま使える。
  - NVFP4 の行カーネル: 64 列(32 bytes、スケール 4 個)ずつ読む。E2M1 → int8 は `vpshufb`、
    E4M3 スケールは 256 要素の F32 表、FP32 のグローバル倍率(と ×0.5)は行末で 1 回。
    活性は 16 列ごとに偶数列 8 個・奇数列 8 個へ並べ替え(`deinterleave`)、下位/上位ニブルとそのまま対応させる。
    活性は F32 のまま(int8 量子化なし)。
  - プールは各ワーカーに **連続した行範囲** を渡す(以前は 8 行おきの飛び飛びで、ハードウェアの先読みが効かなかった)。
    1 回のジョブは複数行列(`Segment`)をまたいで行を均等に分ける。
  - 待機は「一定時間スピンしてから park」(既定 2 ms)。decode 中は層の間隔より長いので、
    ワーカーは起きたまま次の層を受け取る(futex の起床待ちがなくなる)。`--cpu-spin` は従来どおり常時スピン。
- `cpu_experts.rs` に `forward_layer` を追加。1 層のミス全部を **2 回の配布** で計算する:
  gate/up/SwiGLU を融合した 1 回(行 r の gate と up を同じワーカーが計算して即 SwiGLU)と、down の 1 回。
  単体の `forward` も同じ経路で、結果はエキスパートごとに 1 回ずつ呼んだときとビット一致(テストあり)。
- 同期経路(`compute_cpu`)と非同期経路(ドアベル)の両方が `forward_layer` を使う。

実重み(layer3 の 32 エキスパート、453 MB、DRAM から)での CPU 帯域
(`cargo test --release -p hayate-models --lib glm53_cpu_expert_bandwidth -- --ignored --nocapture`):

| | GB/s |
|---|---:|
| 旧カーネル(1 エキスパートずつ、3 回配布) | 14.1〜14.6 |
| 新カーネル、1 エキスパートずつ | 28.0〜29.7 |
| 新カーネル、3 エキスパート/層 | 27.2〜30.7 |
| 新カーネル、8 エキスパート/層 | 29.9〜32.3 |

(他のエージェントのビルドと学習が同時に走っていた時間帯。P コアが埋まると数 GB/s まで落ちる。下記「注意」)

### 伸縮する VRAM 予算(elastic budget)

- 起動時: `--vram-gib N`(`Options::vram_budget`、バイト)で GLM が GPU 上に持つ総量を指定する。
  エキスパートの slot 数 = (N − 常駐(重み・状態・作業領域) − 2 GiB の余白 − 転送用画像 (17 + swaps) 枚) / 画像サイズ。
  指定しなければ従来どおり(機械の計画値から 8 GiB 残す)。
- 実行中: **`Executor::set_vram_budget(bytes) -> Result<Resize>`**。
  - トークンの境界(`step` / `prefill` / `spec_step` の外)で呼ぶ。非同期 decode の途中なら自分で同期して退役させる。
  - 縮小: 順位(直近 64 トークンの経路回数 + 起動時頻度 × 64)の低い GPU エキスパートから外す。
    RAM のコピーがなければ pinned RAM に D2H(2 本のコピー stream)、RAM の上限
    (`--host-gb`、無指定なら MemTotal − 24 GiB)を超える分はディスク段へ落とす(`--allow-disk` が必要。無いと何も変えずに
    `OutOfMemory` を返す)。外したエキスパートの VRAM は **返る前に cudaFree 済み**。
  - 拡大: 順位の高い非常駐エキスパートを RAM(なければディスク)から GPU へ載せ、載せたものの RAM コピーは解放する
    (起動時と同じ「RAM は GPU にないものだけ」の不変条件)。
  - `Resize { slots_before, slots_after, gpu_delta, d2h_bytes, h2d_bytes, disk_reads, seconds }`。
    `gpu_delta` > 0 が解放したバイト数(縮小)、< 0 が新たに確保したバイト数(拡大)。
  - 常駐部(重み、KV/状態、作業領域)は変えない。`Executor::vram_bytes()` が今の計画上の GPU 使用量。
  - prefill の一時領域が使う「空き VRAM」も、予算があるときは予算の残りまでに制限する。
- 非同期 decode は、RAM にもないエキスパート(ディスク段)を HYT から読んで CPU で計算するようになった(遅いが落ちない)。
- 試験用 CLI: `--resize-at TOKEN:GIB`(複数可)。生成の途中で `set_vram_budget` を呼び、`glm_vram_budget` 行を出す。

serve からの使い方(想定): GPU 台帳が重いジョブのために `need` を空けたいとき、
`set_vram_budget(現在値 − 不足分)` を GLM のワーカースレッドでリクエストの合間に呼び、返った `gpu_delta` を台帳に反映する。
ジョブが終わったら元の予算に戻す。

## 注意

- **P コアの競合**: CPU エキスパートは P コア 0,2,…,14 に固定したワーカーで計算する。同じ時間に他のプロセス
  (学習のデータローダー、ビルドなど)が P コアを使うと、帯域は 30 GB/s から数 GB/s まで落ち、decode も比例して遅くなる。
  GLM と同居させる重いジョブは E コア(16〜23)へ `taskset` するのがよい。

### セッション状態(prefix 再利用)と prefill の中断

serve 担当向けの API(`hayate_models::glm53`):

- `Executor::export_session() -> Result<Session>` / `Executor::import_session(&Session)`。
  `Session { position, layers: Vec<LayerSession>, mtp: Option<LayerSession> }`。
  `LayerSession { position, buffers: Vec<(name, bytes)>, latent, tail_keys, tail_gates }`。
  バッファ名は固定(`gpu::SESSION_BUFFERS`):
  - KDA 層(index % 4 != 3): `state`(64 × 128 × 128 F32 = 4 MiB)、`hq` / `hk` / `hv`(短い畳み込みの直近 3 行、各 48 KiB)。
  - sparse 層(index % 4 == 3、MTP 層 45 も同じ形): `pool_keys` / `pool_gates`(未完成プールの末尾 4 × 128 BF16)、
    `pools`(完成プール、position.div_ceil(4) × 128 BF16)、`latent`(pinned RAM の潜在 KV、position × 512 BF16)。
  - 128K で 1 セッション ≈ 1.6 GB(大半は sparse 11 層の latent)。KDA 34 層の固定部は約 140 MB。
  `import_session` は内部で reset してから全層を書き戻し、`position` を合わせる。続けて新しい部分だけを `prefill` すればよい。
  ロジットと MTP の draft 用の隠れ状態は含まない(次の `prefill` が作り直す)。
- `Executor::cancel_handle() -> Arc<AtomicBool>` と `Executor::prefill_cancellable(ids) -> Result<PrefillOutcome>`。
  `PrefillOutcome::Done(next)` か `Cancelled { position }`。フラグは短い入力の逐次経路ではトークンごと、
  バッチ経路では層ごとに見る。チャンクの途中で止めたときは、そのチャンクの開始時に取った固定部のスナップショット
  (上の `state` 等、約 140 MB、D2H)を書き戻し、`position` はチャンク境界に戻る。latent と pools はその先を後で上書きする。
  遅延は「今の層 1 つ分」。現状 8K チャンクの sparse 層は 1 層 4 秒かかるので、吸収型 sparse attention(下の計画)が入るまでは
  最悪数秒。最初のチャンクで層 0 がキャッシュ借用と重なって走った場合だけ、そのチャンクの終わりまで止まらない。
- GPU テスト(ignore): `glm53_session_roundtrip_and_cancel`(export → reset → import の続きがビット一致、1.5 秒後の中断と再開)。

## 実測(RTX PRO 6000、`glm53-mtp-stage11`、固定 frequencies、128 tokens、warmup 2、保守あり)

プロンプト "Write a Rust function that sorts a list of integers and explain its complexity."(raw, ignore-eos)。
`/mnt/NVME4TB/hayate/spill/glm53r/s3/*.log`。

| 構成 | 予算 | decode tok/s | ITL p50 / p95 ms | CPU ミス GB/s | CPU ms/token | GPU hit |
|---|---|---:|---:|---:|---:|---:|
| main(sync)※ | 計画値 ≈ 88 GiB | 3.36 | 270 / 585 | 3.4 | 249 | 0.82 |
| 607a420 async※ | 同 | 7.83 | 125 / 175 | 13.0 | 89 | 0.76 |
| 本段階 async(静的分割)※ | 同 | 5.69 | 161 / 313 | 9.0 | 129 | 0.76 |
| **本段階 async(行チャンク取り合い + 2 ms spin)** | 同 | **14.28** | **68 / 92** | **33.8** | **34** | 0.76 |
| 本段階 async + MTP n=2 | 同 | 13.36 | 74 / 99 | 32.3 | 40 | 0.74 |

※ 他のプロセス(CPU で動く iopaint 11 コア分、CUDA ビルド)と同時。負荷平均 36〜49。同じ条件の再測定は保留中。
過去の main の静かな時の値は約 10 tok/s(README)。

- MTP n=2: 受理率 81%、1 ステップ 157 ms で 2.6 トークン(60 ms/token)だが、検証の 3 行で CPU ミスの和集合が増え、
  適応判定が最初の 8 トークン後に MTP を止めた。MTP 予約で本体の GPU slot も 4 GB 減る。既定はオフのまま。
- elastic(`--resize-at 40:60 --resize-at 90:86`): 88 → 60 GiB で 29.9 GB を 7.7 秒で解放
  (D2H 17.0 GB、RAM 上限を超えた 18.9 GB はディスク段へ)、60 → 86 GiB の拡大は 10.4 秒(ディスク読み 934 個)。
  縮小後のデバイス使用量 68.5 GB → 拡大後 97.5 GB。D2H が 2.2 GB/s と遅いのは、退避先を 1 枚ずつ cudaMallocHost で
  確保しているため(次の改善点: pinned の予備プール)。

## prefill の現状と次の計画(未実装)

第12c段階の Gen5 実測(`spill/glm53-prefill-stage12c-gen5`)の 8,192 トークン all-experts プロファイルでは、合計 70.6 秒のうち:

- sparse attention 11 層: 46.5 秒(1 層 4.2 秒)。`prefill.cu` の `attention` は (行, ヘッド) ごとに 1 ブロックで、
  選んだ最大 2,051 トークンの K/V(ヘッドごとに展開済み、512 要素 × 64 ヘッド)をスカラーで読む。`select` は top-512 を
  1 スレッドの繰り返し argmax で選ぶ。
- KDA 34 層: attention 2.9 秒、routed MoE 4.0 秒(ホストのエキスパート 80 GB の H2D)。
- キャッシュ借用 10.7 秒 + 返却 4.7 秒(14.8 GB の D2H/H2D)。

GLM-5.3 の sparse 層には RoPE がない(`kv_b_proj` 512 → 64 × (K 256 + V 256)、スコア ÷ 16)。そのため
**吸収型(MLA absorbed)** にできる: `q_abs[h] = W_K[h]ᵀ q[h]`(512 次元)、スコア = `q_abs · latent`、
出力 = `W_V[h] (Σ p · latent)`。選択はトークン単位で 64 ヘッド共通なので、1 クエリ行あたり
`Q_abs [64 × 512] · L_selᵀ [512 × 2048]` と `P [64 × 2048] · L_sel [2048 × 512]` の Tensor Core GEMM になる
(1 行 268 MFLOP、8K で 1 層 2.2 TFLOP ≈ 数十 ms)。decode でも選択 2,048 行の K/V 射影(1 層 69 GFLOP/トークン)と
sparse 層ごとの `kv`/`k`/`v` 作業領域(1 層 270 MB、11 層で 3 GB)が不要になる。

順序:
1. 吸収型 sparse attention(prefill と decode の両方。decode を同じ式にして prefill-vs-step のゲートを揃える)。
2. `select` を Tensor Core のスコア計算 + 並列 top-k に。
3. エキスパートの先読みバンクを 288 枚 × 2 から小さいグループの二重バッファにして、キャッシュ借用をなくす
   (借用/返却の 15 秒が消え、短い入力でもバッチ経路が使える。今は 512 トークン未満を逐次 decode で処理)。
4. MTP 有効時もバッチ prefill を使う(今は MTP を読むと legacy の 128 チャンクになる)。

## prefill と step の一致(未解決、切り分け済み)

`glm53_prefill_matches_tokenwise_prefix_logits`(all=true、257 トークンを 128/128/1 で prefill、その後 4 step)を
新しい診断つまみで再実行した(`s3/synth-*.log`)。

| 経路 | row0 | row1 | row2 | row3 |
|---|---:|---:|---:|---:|
| 既定(ミスは CPU) | 0.0109 | 0.0079 | 0.0012 | **0.2910**(top-1 不一致) |
| `GLM53_PMATCH_PCIE=1`(両経路とも全エキスパート GPU) | 0.0110 | 0.0076 | 0.0014 | **0.2891**(top-1 不一致) |

CPU と GPU の丸め差ではない(全 GPU でも同じ)。row3 は prefill の最後(位置 256、1 行チャンク)で始まった
索引プール 64(位置 256〜259)を decode が位置 259 で完成させる最初の step。row0〜2 の 0.008〜0.011 は
GEMM/GEMV の差の範囲。次の切り分け: prefill 長を 256/258/260 にずらして悪い行がプール境界に付いて動くか、
チャンク 64 で同じか、`pool_tail` → decode の `pool_keys`/`pool_gates` と、その位置の latent・pools を両経路でダンプして比較する。

## セッション API の実測

`glm53_session_roundtrip_and_cancel`(`s3/session.log`): export → reset → import の後の 6 step が通常経路と
**ビット一致**。1.5 秒後の中断は `Cancelled { position: 0 }`、フラグから戻るまで 2.9 秒
(最初のチャンクの前のキャッシュ借用 D2H が中断できないため。借用をなくす計画 3 で解消する)。再開後の top-1 は 6/6 一致。

## マージゲート(e7f7069 = ブランチ `phase18/glm53r-gate`)

固定コーパス 512 tokens(prefix 64、448 行、teacher forcing)、FP8 基準(`spill/glm53-eval/fp8`)。
`s3/eval-{sync,async}-report.json`:

| 経路 | KLD | top-1 | max KLD | NLL 差 |
|---|---:|---:|---:|---:|
| sync | 0.0429 | 90.85% | 0.491 | +0.0024 |
| async | 0.0429 | 90.85% | 0.491 | +0.0024 |

sync と async の logits は完全一致。ゲート(KLD ≤ 0.05、top-1 ≥ 90%)を満たす(第6段階 0.041 / 91.1%、main README 0.043 / 90.8%)。

静かな時間帯(負荷平均 約 7)の再測定(`s3/*-r2.log`, `new-sync88.log`, `new-async60.log`):

| 構成 | 予算 | decode tok/s | ITL p50/p95 | CPU GB/s | GPU hit |
|---|---|---:|---:|---:|---:|
| main sync | 計画値 | 11.46 | 84 / 123 | 18.3 | 0.81 |
| 本段階 sync | 計画値 | **14.63** | 67 / 88 | 37.3 | 0.76 |
| 本段階 async | 計画値 | 14.28 | 68 / 92 | 33.8 | 0.76 |
| 本段階 async | 60 GiB(RAM 88 GB、ディスク 34 GB) | 6.12 | 157 / 253 | 14.0 | 0.63(RAM 0.94) |

60 GiB は NVFP4(169 GB)が RAM に入らず、ルートの 6% がディスク読みになる(ドアベルの読み込みが CPU 時間の 2/3)。
main の hit 0.81 と本段階の 0.76 の差(slot 数は本段階の方が多い)は未調査。第13b段階の固定配置か、prefill の経路記録の違いが候補。

# 第18b段階: prefill(作業中)

## 変更

- **吸収型 MLA の sparse attention**(`kernels/cu/glm53/mla.cu`)。prefill・decode 共通。
  - 起動時に `kv_b_proj` から `mla.wkt`(W_K を転置、64 × 512 × 256)と `mla.wv`(W_V、64 × 256 × 512)を作る。
  - 潜在 KV はデバイス上の `latent_cache`(context × 512 BF16、sparse 層ごと)。128K で 11 層 + MTP ≈ 1.5 GB。
    pinned RAM の写しは MTP 検証(旧経路)を読み込んだときだけ持つ。
  - 1 クエリ行 × 16 ヘッドを 1 ブロックで、選んだトークンを 64 個ずつ共有メモリに集め、`mma.m16n8k16`(BF16、F32 累積)で
    スコアと P·L、オンライン softmax。decode は 8 分割して結合。prefill の q_abs と出力射影はヘッドごとの GEMM。
  - スコアの丸め(BF16 の dot、÷16、BF16)は従来の式に合わせた。F64 参照に対して相対 L2 0.0023(カーネル試験)。
  - decode の host 同期(選択 ID の読み戻し)と、選択 2,048 行の K/V 射影(1 層 69 GFLOP/トークン)、3 GB の作業領域がなくなった。
- **上位 512 プールの選択**: スコアは従来と同じ式(CUDA コア)、選択は 1 行 1 ブロックの 8 bit × 4 段の radix select。
  同点は低い番号を優先(従来と同じ集合)。出力は時刻順(decode の並べ替えと同じ)。
- **prefill のワークスペース**: エキスパートのグループを「8 個まで、かつ行数がチャンクの行数まで」にして、グループ用バッファを
  チャンク行数分に(以前は 8 倍)。MLA の中間は KDA 用の領域と共有、出力は in-place。キャッシュの借用はやめ、足りなければ
  チャンクを縮める。`--vram-gib` のときは 8K チャンク分のワークスペースを予算の中に確保する(`prefill_reserve`)。
- **KDA のチャンク prefill**: 1 ヘッド 1 ブロック(64 ブロック)から、ヘッド × 値の列 16 本ずつ(512 ブロック)に。状態は共有メモリ。
  演算と順序は同じ(dim=128 で旧カーネルとの差 3e-5 以下)。
- バッチ prefill を 32 トークンから使う(以前は 512 未満を逐次 decode)。
- `--bench-text FILE --bench-split PREFIX,TURN --max-tokens N`: PREFIX トークンを冷えた状態で prefill、続けて TURN トークン
  (= PREFIX を再利用したターンの TTFT)、その文脈長で decode。

## prefill と step の一致(row3)

自然文で確かめた。コーパス 448 トークンを 1 回のバッチ prefill、その後 64 行を step(`s6/eval-p448`)。
FP8 基準に対する同じ 65 行の KLD は、step だけの評価(`s5`/`s3`)の 0.066〜0.070 / top-1 84.6% に対して
0.064 / 86.2% で、バッチ prefill 後の状態の品質は step と同等。合成列(17 + i mod 101 の周期列)の row3 は、
GEMM と GEMV の丸め差からの近接同点の反転で、引き継ぎの誤りではないと判断した(全エキスパート GPU でも同じ値)。
