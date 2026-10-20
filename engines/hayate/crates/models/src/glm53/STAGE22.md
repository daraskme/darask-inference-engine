# 第22段階: LOWBIT(3+4 bit 混合)エキスパートをランタイムへ(phase22/glm53lbx)

基点は main dcaa609(第21段階の統合)。量子化トラックの 150 GB 案(`plans/pass3/b150.json`: nf3 g32 と NVFP4 の
混合、GPTQ)を `glm53_lowbit convert --container packed` で作った `glm53q-p150q34-packed`(HYT tag 6)をそのまま走らせる。
Codex Sol の LOWBIT カーネル(phase19/glm53lb: GPU の gate/up/down・prefill 用 NVFP4 変換、phase19/glm53cpu: CPU 行カーネル)を
組み込んだ。計測は `spill/glm53d/b`(`one.sh` の `lb-*` / `vs-*`)。

## 1. 組み込み

- **HYT**: `QuantSpec::Lowbit { grid, group }`(タグ 6、`[6, grid, group]`)。`matrix_bytes` は仕様書 §2 の表と一致(試験あり)。
- **ローダ**(`experts.rs`): 層に tag 6 が 1 つでもあれば「混合層」。エキスパートごとに gate/up/down の形式
  (`ExpertFormat` = NVFP4 か LOWBIT(grid, group))と画像バイト数(3 行列を 256 B 刻みで並べた合計)を読む。`read_image` は
  既存の並べ方のまま LOWBIT のスケール(有限の E4M3、正のテンソルスケール)を検査する。
- **デバイス decode**: `route` カーネルが選ばれた 8 エキスパートの `ExpertFormat` を `decode.formats` に書き、
  `hayate_glm_lowbit_gate_up/down`(Sol)が形式ごとに計算する。CUDA graph はそのまま。形式の表 `[42][288]` は起動時に 1 回だけ上げる。
- **CPU のミス**: ドアベルのワーカーが各エキスパートの形式で `forward_layer_formats`(Sol)を呼ぶ。
- **prefill**: 混合層は LOWBIT の画像を NVFP4 の画像へ無損失変換(`hayate_glm_lowbit_convert_expert`)してから、既存の展開・グループ GEMM に渡す。
- 対応しないもの(エラーにする): 同期 decode、MTP、旧 prefill、`--maintenance-async`。LOWBIT の層は device decode が前提。

## 2. 可変サイズの GPU スロット

LOWBIT 混合の画像は 10.22 / 11.54 / 12.85 / 14.16 MB(54% / 17% / 0.1% / 29%)。スロットを全部 14.16 MB にすると
小さい画像の分だけ無駄になる。

- GPU のスロットはそのエキスパート自身のバイト数(`expert_bytes`)で確保する。pinned の RAM 画像は最大サイズの一様な
  バッファのまま、コピーは先頭の `expert_bytes` だけ(`copy_*_prefix_async`)。
- cudaMalloc は 2 MiB に切り上げる。11.54 / 12.85 MB の画像は 9〜14% 余るので、無駄が 8% 以上のサイズだけ
  同じ大きさ 8〜16 個を 1 回の確保から切り出す(`Device::alloc_split`。弾力的な縮小の粒度を保つため主なサイズ 10.22 / 14.16 は個別)。
- 入れ替えは「入る側が収まるスロット」だけを相手にする: 保守(`plan_recent_fit`)、文脈の配置(`place_scores`)、弾力的な縮小(`resize` は
  バイト予算で順位順に詰める)。入るエキスパートは勝てるスロットのうち収まる最小のものを使う。
- 入れ替えはスロットの上でそのまま行う(退く側を pinned に D2H、入る側を H2D で上書き)。トークン境界(GPU が止まっている間)にだけ行うので
  `--maintenance-async` は廃止。
- prefill のキャッシュ借用は最大サイズのスロットだけから借りる。
- 一様な既存のチェックポイント(NVFP4 / GPTQ)は全エキスパートが最大サイズなので、確保サイズ・入れ替えの相手とも以前と同じ。

150 GB 案(GPU 予算 81.2 GB): スロット 5,737 → **6,932(+21%)**。再生シミュレーションの予測(+20.8%、静的な頻度順で短文 0.784 → 0.849)と一致。

## 3. 実測

| 条件(150 GB 案、device decode) | tok/s | 1 トークン p50 | GPU hit | CPU ms/token | 備考 |
|---|---:|---:|---:|---:|---|
| 参考: 同じ重みを NVFP4 に展開(`b-p150`) | 27.4 | 33.8 | 0.811 | 23.9 | スロット 5,737 |
| LOWBIT、一様スロット(`lb-short`) | 29.4 | 34.2 | 0.791 | 23.4 | 5,737 |
| **LOWBIT、可変スロット(`lb-short`)** | **31.0** | **30.0** | **0.856** | **19.8** | **6,932** |
| LOWBIT 一様、コード 1K | 18.7 | 37.1 | 0.796 | | probe の入れ替え 2,821 個 |
| **LOWBIT 可変、コード 1K** | **21.9** | **30.1** | **0.861** | | probe 2,622 個(約 2 秒、平均に含まれる) |
| 参考: GPTQ(現行 serve 既定)、短い英文(第21段階) | 22.9 | | 0.727 | | |
| 参考: GPTQ、コード 1K(第21段階) | 19.8 | 33.5 | 0.794 | | |

品質(固定コーパス、prefix 64、448 行、FP8 基準):

| 構成 | KLD | top-1 | max KLD |
|---|---:|---:|---:|
| GPTQ(第21段階) | 0.0363 | 90.18% | 0.584 |
| 150 GB を NVFP4 に展開 | 0.0524 | 90.62% | 0.693 |
| LOWBIT、一様スロット | 0.0498 | 90.40% | 0.825 |
| **LOWBIT、可変スロット** | **0.0496** | **89.06%** | 0.608 |

目標(KLD 0.05 前後、top-1 88% 前後)の範囲。top-1 の差(448 行中 6 行)は CPU/GPU の丸め差と近い同点の入れ替え。

GPU 試験(`#[ignore]`): `glm53_device_decode_matches_sync`・`glm53_device_decode_session_and_budget`(stage6 重み、旧経路の回帰)、
`glm53_context_placement_swaps_images`、`glm53_context_placement_variable_slots`(収まる最小のスロットへ、中身のバイト一致)、
`glm53_boundary_expert_transfers` すべて通過。serve の e2e(150 GB 版)14/14、`decode_speed` 24.2 tok/s(serve の予算は上限 84 GiB で
スロット 5,729、GPTQ は 20.7)。

## 4. 見つけた不具合・教訓

- cudaMalloc は 2 MiB に切り上げる: 7.08 MB の画像を 11,474 個確保すると 18% 余り、86.9 GB の空きに載らず OOM した
  (全部 2 bit の実験)。無駄が多いサイズは 1 回の確保から切り出す。
- pinned の RAM 予算は「GPU が頻度順に取る個数」から見積もる(以前は平均サイズで割って少なく見積もり、計画が「arena が空」で失敗した)。
- 全部 2 bit(89 GB、全エキスパートが GPU)の実験: decode 61〜67 tok/s、8K prefill 18 秒(447 tok/s)だが KLD 0.284 / top-1 77.2%、
  計算問題で誤答。チェックポイントは削除した(品質を優先する方針)。

## 5. 残り

- 文脈の配置(probe)の入れ替え 2,600 個(約 2 秒)は D2H が半分: LOWBIT の RAM に余裕ができたのでエキスパートの RAM コピーを残せば
  D2H が要らない(DRAM 往復が半分)。
- 3 bit の GPU カーネルは NVFP4 より遅い(gate/up 70.7 対 65.8 µs)。復号の計算が律速で 1.0 TB/s 止まり。
- embedding をホストへ(約 1.3 GB = 約 110 スロット)、prefill の作業領域の縮小。
- 弾力的な縮小は 11.54 / 12.85 MB の分だけ粒度が粗い(切り出しの兄弟が残る間は VRAM が戻らない)。
- MTP をデバイス経路で(LOWBIT では未対応)。

## 6. serve の既定(第22段階の最後)

- 重みの既定を `glm53q-p150q34-packed`(150 GB の 3+4 bit)に、自動予算の上限を 90 GiB にした。GPTQ(181 GB)・rawdense は `--glm-hyt` で選べる。
- 6K トークン以上のプロンプトは prefill の塊を縮めず、フルサイズのスロットの半分まで借りる(塊を半分にするとエキスパートの転送が倍になる)。
  これで上限を上げても最初の大きな prefill が遅くならない。エージェントループ(34K + 約 2.2K/ターン、返答 128 トークン、150 GB 版):

| 上限 | 最初の 34K prefill | decode(ターン 1〜4) | GPU のエキスパート |
|---|---:|---|---:|
| 84 GiB | 32.6 s | 20.1 / 18.0 / 14.1 / 17.0 | 5,729 |
| 90 GiB(採用) | 34.1 s | 20.1 / 22.2 / 22.6 / 22.7 | 6,278 |
| 92 GiB | 39.1 s | 24.9 / 21.2 / 23.3 / 25.1 | 6,450 |

- 既定のままの確認(`def-e2e` / `def-bench`): e2e 14/14、`decode_speed` 25.3 tok/s(第21段階の GPTQ では 20.7)、
  エージェントループ 20.8 / 22.7 / 18.5 / 21.4 tok/s、最初の 34K prefill 35.8 s、プロンプト読み込み中のピーク 95.1 GB(カードの空き約 2.7 GiB)。
