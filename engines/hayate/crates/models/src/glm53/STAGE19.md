# 第19段階: デバイス側 decode(phase19/glm53d)

基点は phase18/glm53r 74cf217(第18b段階 WIP)。main(83c5943)を merge 済み。
重みは `/mnt/NVME4TB/hayate/hyt/glm53-stage6-fp8-dense`(dealignai から再変換、NVFP4 エキスパート)、
頻度は `spill/glm53-stage6/frequencies.bin`、計測は `spill/glm53d/{a,b}`(`one.sh` / `driver.sh`)。
この日は量子化トラックの変換・評価が同じ機械で動いており、負荷平均 2〜7。CPU エキスパートの帯域は
31〜38 GB/s の間で揺れた(同じ構成でも ±10%)。比較は近い時間の組で行う。

## A. 第18b段階の確認

- `cargo test`(CPU): 通過。古くなっていた `short_prompt_avoids_uploads_only_when_there_are_misses`
  (バッチ prefill を 32 トークンからにした変更に追随していなかった)を直した。
- GPU の ignore テスト: kernels の glm53 一式 17 件、models の checkpoint 不要 8 件、
  `glm53_prefill_matches_step` / `glm53_decode_stage6_graph`(8 層)、`glm53_decode_cpu_graph_matches_eager`
  (45 層、CPU ミス 1407 件、eager/Graph ビット一致)、`glm53_session_roundtrip_and_cancel`(ビット一致、
  中断はフラグから 0.40 s で `Cancelled { position: 0 }`、再開後 top-1 6/6)、`glm53_cpu_expert_matches_gpu`
  (CPU と GPU のエキスパート出力の相対 L2 1.3e-4)は通過。
- `glm53_prefill_matches_tokenwise_prefix_logits` は**既定の合成列(17 + i mod 101)で失敗**(KLD 0.005〜0.32、
  上限 0.01)。同じ試験を自然文(`GLM53_PMATCH_IDS` = 固定コーパス)で流すと全行 KLD ≤ 0.00035、top-1 全一致で通過。
  周期列では索引プールのスコアが近い同点だらけになり、GEMM と GEMV の丸め差で選択が入れ替わるため
  (第18段階の row3 と同じ現象が全行に広がった)。試験の入力を自然文に替えるかは統合担当の判断。
- main を merge したとき、serve の KV ブロック読み書き(`session_state.rs`)が pinned の潜在 KV を読んでいたので、
  第18b段階のデバイス上の `latent_cache` を読むように直した。

decode(128 トークン、warmup 2、raw、ignore-eos、"Write a Rust function that sorts…"):

| 構成 | decode tok/s | ITL p50 / p95 ms | CPU ms/token | CPU GB/s | GPU hit | host 同期/step |
|---|---:|---:|---:|---:|---:|---:|
| main(83c5943 のバイナリ)sync、計画値 | 13.52 | 65.7 / 114.4 | 33.8 | 32.4 | 0.769 | 65 |
| 18b sync、計画値(≈88 GiB) | 17.07 | 55.7 / 79.1 | 25.0 | 33.5 | 0.824 | 43 |
| 18b async、計画値 | 17.12 | 56.2 / 78.4 | 25.1 | 33.5 | 0.824 | 1 |
| 18b async、60 GiB(RAM 88 GB、残りディスク) | 4.85 | 145.5 / 466.5 | 171 | 10.1 | 0.636(RAM 0.938) | 1 |

18b のヒット率が main より高いのは、常駐分が 12.71 → 10.19 GB に減って slot が 5592 → 5770 に増えたため
(MTP 用の射影 KV バッファを外した分)。60 GiB はディスク段の読み込み(ミスの 6%)が支配的で、第18段階の 6.12 より遅い
(この日は RAM を他のジョブも使っていてページキャッシュが少ない)。

8K prefill(all experts、8192 トークン): 44.3 s(**184.9 tok/s**。第12c段階の all-experts は 70.6 s)。

品質(固定コーパス、prefix 64、448 行、dealignai FP8 基準 `spill/glm53-dealignai/fp8-ref`、eval-mode prefill):

| 経路 | KLD | top-1 | max KLD | NLL 差 |
|---|---:|---:|---:|---:|
| main(7a7cc06 の GLM コード) | 0.0393 | 90.18% | 0.489 | −0.0030 |
| 18b(prefill モード、prefix はバッチ prefill) | 0.0410 | **88.62%** | 0.565 | +0.0052 |
| 18b(step モード、prefix を decode で) | 0.0398 | 89.73% | 0.399 | −0.0019 |
| 18b 対 main(prefill モード) | 0.0114 | 一致 93.97% | 0.238 | |
| 18b 対 main(step モード) | 0.0106 | 一致 94.87% | 0.183 | |

- ゲート(KLD ≤ 0.05 かつ top-1 ≥ 90%)は、prefill モードの top-1 で **1.4 点届かない**(448 行中 7 行)。
  KLD は満たす。基準と同じ step 経路で prefix を作ると 89.73%(main と 2 行差)。
- prefill モードと step モードの差(5 行)は、prefix の処理(18b は 32 トークンからバッチ prefill、main は 512 未満を
  decode)そのものではない: 自然文の pmatch では両者の KLD は 1e-5 台。差は prefix の間の保守(バッチ prefill は
  1 回、step は 64 回)でエキスパートの配置が変わり、CPU/GPU の分担の丸め差が近い同点を入れ替える分と見ている。
  decode 経路そのもの(吸収型 MLA)は main と KLD 0.011 の差で、基準からの距離は同等(0.0398 対 0.0393)。
- main 自身も dealignai 基準では 90.18% でゲートの際にいる。

## B. デバイス側 decode(`--decode-host device`)

### 計測: 第18b段階の 1 トークンの GPU 側

nsys / ncu は開発環境に無い(nix store に CUPTI のライブラリだけ)。代わりに、実形状のカーネル単体ベンチ
(`glm53_decode_kernel_bench` / `glm53_device_kernel_bench`、重みは L2 を超える数のコピーを巡回させて毎回 DRAM から
読む)と、フェーズ別の CUDA event(`--profile`)で測った。

18b のカーネル単体(µs/回、1 トークンの回数を掛けた合計):

| カーネル | µs | 回数/トークン | ms/トークン |
|---|---:|---:|---:|
| mHC(`hc_map` 1 ブロック + collapse/norm) | 166 | 90 | **15.0** |
| router(射影 + 1 スレッドの top-8) | 52 | 42 | 2.2 |
| q/k/v・o_proj などの FP8 GEMV(大きい行列は 1.2〜1.5 TB/s、小さい行列は 7〜14 µs の遅延律速) | | | 約 6 |
| エキスパート gate/up・down(ヒット 8 個: 95 + 42 µs、約 1 TB/s) | | 42 | 約 4.5 |
| sparse 層の選択・MLA・小さい演算・head | | | 約 3 |

合計約 30 ms に、起動の隙間と層ごとの host 同期(43 回/step)が乗って、旧経路の GPU 側は約 37 ms。
全エキスパートが GPU に載る 20 層(dense 3 + MoE 17)で測ると、旧 sync 経路は 13.96 ms/トークン(72 tok/s)で、
45 層に直すと約 31 ms が「ミスを除いた GPU 経路」の下限だった。

### 変更

- **1 トークン = 1 本の CUDA graph**(実行器の stream 1 本)。最初のトークンは eager で流してキャプチャし、以後は再生。
  host 同期はトークンの最後の 1 回(43 → **1 回/step**)。
- 位置・ドアベルの通し番号・トークン ID はデバイスの制御ブロック `{token, position, seq, error, next}`。host は mapped
  メモリに書き、埋め込みカーネルが取り込む。greedy の結果と error は mapped メモリへ返す。
- **ルーティングはデバイスで完結**: router(288 ロジット)→ 最後に到着したブロックが top-8(逐次版と同じ同点処理)→
  デバイス上の表 `[層][エキスパート]`(VRAM slot のアドレス、0 は CPU)を引いて `decode.images` と CPU の計画
  `{now, later, skip, defer}` を書き、ids と入力を層の mailbox(mapped)に置いて READY = seq。
  GPU のヒット分は既存の一括 gate/up・down。表はトークンの境界で slot から 1 回で作り直し、差分があれば H2D するだけ
  (保守・伸縮のあとも graph は作り直さない)。
- **CPU のミス**は既存のドアベルワーカー(E コア 22 で READY を待ち、P コア 8 本で計算、出力を mailbox に書いて DONE)。
  GPU はミスがある層だけスピン待ちカーネル(`%globaltimer` で 120 秒の上限、超えたら error でトークンが失敗)。
  ホスト関数ノードは使わない(起床の遅延が層ごとに乗るため)。起動に失敗したときは host が READY を鳴らしてジョブを
  終わらせる(ワーカーが永久に待たない)。
- **カーネル**(数値は旧逐次 decode とビット一致。演算と縮約順を変えない):
  - GEMV(`dense.cuh`): レーンの担当列と和の順序はそのままに、8 回分の読み込みを先に発行。大きい行列 1.41〜1.56 TB/s、
    小さい行列は 2.8〜6 µs(18b は 7〜14 µs)。gate/up 95 → 64 µs、down 42 → 30 µs。
  - mHC: 24 行のロジットを 24 ブロックで(各ブロックは旧 1 ブロック版と同じ順序)、最後のブロックが Sinkhorn・
    collapse・RMSNorm。166 → **12〜15 µs**。
  - KDA: 入力射影 6 本を 1 起動(64 µs、1.5〜1.6 TB/s)、短い畳み込み・f_b/g_b・decay・beta・再帰・gated norm を
    ヘッドごとに 1 起動(20 µs)。sparse 層: 入力射影 5 本・q_b/wq_b を各 1 起動、norm・潜在 KV 追記・プール末尾・
    プール完成を 1 起動、top-512 選択はデバイスの位置を読む。
  - router + top-8 + 表 + ドアベル 52 → 8.4 µs。縮約 + mHC expand を 1 起動、head は mean+norm、lm_head、2 段の argmax。
  - 1 トークンの起動数は約 600(18b は約 1300 + 層ごとの同期)。
- 近似のつまみ(既定オフ、固定コーパスの KLD で判断):
  - `--skip-cpu-below W`: ルーティング重みが W 未満の CPU ミスを計算しない(GPU のマスクとワーカーの ID フラグ)。
  - `--defer-below W` / `--defer-cpu-experts`(全部): Expert deferral(KTransformers)。重み W 未満の CPU ミスを
    層 k の出力に足さず、層 k+1 の出力で足す(mHC なので係数 `comb_f(k+1)ᵀ comb_a(k+1)ᵀ post(k)`)。
    ワーカーは即時分を先に計算して DONE、遅延分を後で DONE2。最後の MoE 層は遅延しない。
- 保守: 既定は従来どおりトークンの境界で完了・公開(ビット一致を保つ)。`--maintenance-async` では次のトークンの
  実行中に計画し、その次の開始で公開する(ヒット率が 0.824 → 0.800 に落ちて遅くなったので非推奨)。
- 計測用: `--save-routes FILE`(ルーティングの記録)、`--profile`(デバイス経路はフェーズ別 event、graph なし)。
- serve の API(`set_vram_budget`、`export_session`/`import_session`、`prefill_cancellable`、serve の
  `session_*`)はそのまま。prefill・リセット・セッション復元の後は、次のデバイストークンの前に層の stream を同期する。

### 結果(88 GiB = 計画値、128 トークン、warmup 2)

同じ GPU 枠の中で続けて測った組(`spill/glm53d/b/final-*.log`、負荷平均 4.1〜4.5):

| 構成 | decode tok/s | ITL p50 / p95 ms | CPU ms/token(GB/s) | GPU hit | host 同期/step |
|---|---:|---:|---:|---:|---:|
| 18b sync | 19.08 | 50.6 / 73.5 | 22.1(38.0) | 0.824 | 43 |
| **device** | **28.37** | **33.7 / 53.8** | 23.9(35.1) | 0.824 | **1** |
| device + `--defer-below 0.25`(近似) | 31.28 | 26.4 / 61.0 | 25.5(38.5) | 0.794 | 1 |

別の時間帯の device: 28.19(p50 33.9 / p95 52.1、CPU 37.9 GB/s)、25.64(CPU 31.7 GB/s、量子化トラックの変換と同時)。
CPU 帯域が揺れる分だけ速度も揺れる。

1 トークンの内訳(device、eager + event、`devprof88.log`、ms):

| attention(mHC 込み) | FFN mHC + router | GPU エキスパート(ヒット) | CPU 待ち | 縮約 + expand | head | GPU 経路の合計 |
|---:|---:|---:|---:|---:|---:|---:|
| 7.68 | 1.43 | 3.86 | 22.95 | 0.34 | 0.44 | **13.75** |

- **GPU 経路(CPU 待ちを除く)は約 37 ms → 13.75 ms**(eager の event 込み。graph では起動の隙間がさらに減る)。
- host 側(graph の外): 準備 0.19 ms、回収 0.23 ms、保守 1.3 ms(同期スワップ 1 回/トークンの完了待ち)。
- 全エキスパート GPU(20 層): 旧 sync 13.96 ms/トークン(72 tok/s)→ device **6.41 ms(158 tok/s)**。
  フェーズ別 attention 3.05 / router 0.68 / エキスパート 1.91 / 縮約 0.12 / head 0.41。45 層に直すと約 13 ms
  (全エキスパートが GPU にあれば約 75 tok/s)。帯域の下限(約 7 ms)の 2 倍弱。
- 60 GiB(RAM 88 GB、ルートの 6% がディスク段): 18b async 4.85 → device **6.32 tok/s**(p50 111.6 / p95 352.3)。
  ディスク段のミスをワーカーが 1 本ずつバッファ付きで読む部分(約 100 ms/トークン)が支配的。

### 正しさ

- `glm53_device_decode_matches_sync`: prefill(37 トークン)+ 14 step(プール境界をまたぐ)で、device(eager の
  1 トークン目と graph 再生)と逐次 sync の logits・greedy が**ビット一致**(CPU ミス 27,309 件を含む)。
- 固定コーパス 448 行(保守あり): device の logits ファイルは 18b sync と**バイト一致**(prefix の logits も)。
  したがって品質は A の 18b と同じ(KLD 0.0410 / top-1 88.62%)。
- `glm53_device_decode_session_and_budget`: device で export → reset → import の続きがビット一致、
  80 → 60 GiB の縮小(21.5 GB 解放)と 80 GiB への拡大をトークンの間に挟んでも greedy 8/8 一致。
- kernels の既存のビット試験(dense、MoE、router、verification、fusions)は新しい GEMV でも通過。

### 近似のつまみ(固定コーパス、FP8 基準、eval-mode prefill)

| 設定 | KLD | top-1 | eval 448 行の時間 |
|---|---:|---:|---:|
| device(近似なし) | 0.0410 | 88.62% | 25.8 s |
| `--skip-cpu-below 0.15` | 0.0412 | 89.51% | 25.5 s |
| `--skip-cpu-below 0.20` | 0.0456 | 89.29% | 20.3 s |
| `--skip-cpu-below 0.25` | 0.0607 | 87.50% | 16.5 s(bench 41.5 tok/s) |
| `--skip-cpu-below 0.35` | 0.0941 | 85.94% | 11.8 s |
| `--defer-below 0.10` | 0.0397 | 89.96% | 26.0 s |
| `--defer-below 0.20` | 0.0422 | 88.39% | 24.5 s |
| `--defer-below 0.25` | 0.0464 | 90.40% | 25.2 s(bench 31.3 tok/s) |
| `--skip-cpu-below 0.15 --defer-below 0.25` | 0.0475 | 88.39% | 25.6 s |
| `--defer-cpu-experts`(全部) | **0.1618** | 85.04% | 25.1 s |

- 全部の遅延は品質が大きく落ちる(KDA の状態と sparse 層の潜在 KV に、ミスを含まない残差が書き込まれて後のトークンまで残る)。
  採用しない。
- 低い重みのミスは小さい: 重み 0.15 未満は捨てても差が見えない。0.20 で KLD +0.005、0.25 で +0.02。
- いずれも既定オフ。採否は統合担当の判断(ゲート top-1 ≥ 90% は基準側のゆらぎ(±5 行)と同程度)。
- 1 本のプロンプトの bench は、近似で生成文が変わるとルーティングも変わるので比べにくい(`--defer-below 0.20` は
  文が変わってヒット率 0.694、20.3 tok/s)。近似どうしの速さは teacher forcing の eval 時間で比べた。

### その他の計測

- **zero-copy**(`glm53_zero_copy_bandwidth`): GPU が pinned メモリを直接読むと 29.2 GB/s、CPU 8 スレッドは 42.0 GB/s、
  同時に読むと GPU 10.2 + CPU 30.2 = **40.5 GB/s**(CPU 単独と同じ)。DRAM 律速なので、ミスを GPU に直読みさせても
  総帯域は増えない。採用しない。
- **キャッシュ方式**(`spill/glm53d/cache_sim.py`、device 88 GiB の 142 トークンのルート記録、slot 5,770):

  | 方式 | ヒット率 |
  |---|---:|
  | 頻度の静的配置 | 0.786 |
  | 窓 64 + 1 スワップ/トークン(現在の保守の近似。実測 0.824) | 0.809 |
  | 窓 64 + 4 スワップ/トークン | 0.825 |
  | 全層共通 LRU(ミスごとに入れ替え) | **0.878** |
  | slot 1.2 倍 / 1.5 倍(静的) | 0.849 / 0.916 |
  | slot 1.2 倍 / 1.5 倍(LRU) | 0.923 / 0.966 |

  LRU はヒット率が高いが、NVFP4 では VRAM と RAM を合わせてちょうど全エキスパート(171 GB)なので、入れ替えるたびに
  追い出す側の D2H が要り、1 回の入れ替えが 28 MB の DRAM 往復になる。ミス 41 個/トークンを全部入れ替えると
  1.16 GB/トークンで、今の CPU のミス読み出し(0.84 GB/トークン)より DRAM を多く使う。4 スワップも同じ理由で
  帯域の合計は悪化する(0.95 GB 対 0.87 GB)。RAM にコピーを残せる(エキスパートが小さい)LOWBIT なら LRU が効く。

### 50 tok/s までの距離

device の 1 トークン約 35 ms のうち、CPU のミス(DRAM 律速、約 0.84 GB/トークン ÷ 35〜38 GB/s)が約 22〜24 ms、
GPU 経路が約 12〜13 ms。GPU 側だけでは 50 tok/s(20 ms)に届かない。厳密なまま届くには、ミスのバイトを
1/4 程度にする必要がある(C の LOWBIT: 3 bpw 前後で slot が 1.5 倍、ミス 1 個のバイトも 2/3 → CPU 約 5 ms、
GPU 約 12 ms で 50 tok/s 前後の見込み)。近似なら `--skip-cpu-below 0.20` で KLD 0.046(device で約 35 tok/s 相当)。

### 未確認・残り

- serve は `Options::default()`(sync)のまま。device に切り替えるのは `decode_device: true` の 1 行(ビット一致、
  セッション・伸縮・中断の試験は通過)。serve の e2e はこのブランチでは未実施。
- サンプリング: greedy はデバイス。温度つきは従来どおり host(logits 310 KB の D2H + 候補の絞り込み、0.3 ms 程度)。
- MTP(D)はデバイス経路に未対応(device と MTP は同時に指定できない)。
- 60 GiB のディスク段: ワーカーのディスク読みを O_DIRECT・並列にすれば約 2 倍の見込み(未実装)。
- **長文脈の decode(32K / 128K)は未計測**。`--bench-tile K`(prefix を prefill したあと sparse 層の潜在 KV と索引プールを
  K 倍に複製して位置を進める。128K の実 prefill は 1 回の GPU 枠 10 分に収まらないため)と、プロファイルの
  `sparse_select_ms` / `sparse_mla_ms` を入れたところで、ディレクトリ移動のため計測を止めた。
  再開後に `spill/glm53d/b/one.sh` の `long32` / `long128` / `long32p` / `long128p` / `long32sync` を流す
  (`bench.txt` で prefix 32768 または 32000 × 4、turn 512、context 131072)。128K の潜在 KV は設計上
  11 層 × 131072 × 1 KiB = 1.48 GB(VRAM、`--context 131072` で起動時に確保)。
