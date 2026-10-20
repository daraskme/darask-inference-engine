# Hayate の構成

Hayate はローカルの GLM-5.3-Flash 専用。対象は RTX PRO 6000 Blackwell 96 GB（sm_120a）、i9-12900KS、RAM 128 GB。モデルの計算順序を Rust に直接書き、汎用グラフ IR は使わない。

| crate | 役割 |
|---|---|
| core | HYT 重み形式、テンソル、量子化仕様、メモリ計画、mmap / NVMe I/O |
| kernels | GLM の CUDA と CPU AVX2、CUDA 実行・グラフ、共用行列演算 |
| models | GLM 45 層、MoE 配置・キャッシュ、CPU/GPU 非同期実行、LoRA |
| llm | 会話チェックポイント、KV ブロック再利用、ホストサンプラー |
| server | ローカル HTTP、SSE、GLM テンプレート・出力解析、外部 GPU ジョブとの予算調整 |
| cli | `run glm53`、`serve`、`capabilities --json`、`bench host` |

## 計算と重み

GLM の 45 層は最初の 3 層が dense FFN、残り 42 層は 288 個中 8 個を選ぶ MoE。線形アテンションの再帰状態と、疎アテンションの潜在 KV・インデクサ状態を層ごとに管理する。変換済み HYT の量子化仕様を読み、LOWBIT、EXL3、NVFP4、FP8 を明示的なカーネルで実行する。

エキスパートは頻度と予算に基づき GPU / RAM / 許可時は NVMe に配置。RAM 上のエキスパートを CPU ワーカーで計算して GPU の処理と重ねる。CUDA Graph とホスト可視の通知領域による device decode を使う。GLM の実験用 CLI に MTP 検証が残るが、serve は MTP を使わない。

`kernels::graph` は以前 H3 モジュールにあった汎用 CUDA Graph の実装。`kernels::text` と `kernels::llm` のうち GLM が使う BF16 線形演算・GEMV は残す。モデル固有の画像・動画・他 LLM のカーネルは削除した。

## 会話とサーバー

HTTP 接続は所有データを容量 32 の FIFO に送り、CUDA を所有する 1 本のワーカーが順に処理する。SSE と非ストリームの両方で接続切断を通知し、長い処理を中断する。CUDA の致命的な非同期エラーはプロセスを終了させる。

4 会話のプレフィクスを保持し、再帰状態のチェックポイントと 1024 トークン単位の共有 KV ブロックを再利用する。生成の seed と CDF の順序はサンプラー最適化でも維持する。ただし会話切替や予算変更による CPU/GPU のエキスパート配置差では丸めが変わるため、モデル全体のバッチ不変性や完全な出力一致は保証しない。

VRAM 予算は要求間に変更する。[GPU_SHARING.md](GPU_SHARING.md) と `crates/models/src/glm53/` の各段階の報告を参照。
