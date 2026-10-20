# GLM-5.3-Flash を serve で使う(Studio 向けメモ)

`hayate serve` の常駐チャットモデル `glm53`。OpenAI 形式の `/v1/chat/completions` で、チャットの封筒(ストリーミング SSE、`reasoning_content`、`tool_calls`、最後のチャンクの `usage` と `timings`)を返す。

## 読み込み

```http
POST /v1/models/load   {"id":"glm53","variant":"fp8"}      # variant は省略可。fp8 以外は 400
POST /v1/models/unload {"id":"glm53"}
```

- SSE(`Accept: text/event-stream`)なら `{"stage":"weights",...}` → `{"stage":"warmup",...}` → `{"stage":"ready"}`。
- LLM なので GPU 枠(`gpu.lock`)の外。重いジョブのために降ろされることはない(docs/GPU_SHARING.md)。
- 起動オプション(既定値):`--glm-hyt /mnt/NVME4TB/darask-inference-engine/local/hayate/hyt/glm53x3-sc260`(2026-10-20 午後から。turboderp の EXL3-SC 2.60bpw のエキスパートをトレリスのまま使う(2.5 bpw、層 3 は 3.5 bpw、HYT タグ 7)。注意・密層・共有エキスパート・lm_head は変換時に復号して f100 と同じ FP8。全体 106 GB。FP8 基準で KLD 0.078 / top-1 88.4%。元は Z.ai 原版なので、LoRA なしだと一部を断る。heretic LoRA を載せると f100 + LoRA と同じく断らない。LoRA ありの decode は短文 44.7 tok/s、code1k 40.2、32K の文脈で約 33 tok/s。CPU で計算するミスが重いので、PCIe で運ぶ経路を作って取り戻す予定。設計と実測は crates/models/src/glm53/EXL3.md。速度優先なら `--glm-hyt /mnt/NVME4TB/darask-inference-engine/local/hayate/hyt/glm53q-f100-packed`(f100: GPTQ の 2 bit(n2 g32)・3 bit(nf3 g32)と NVFP4 の混合で全体 100 GB、HYT タグ 6、KLD 0.191 / top-1 81.5%、LoRA ありの decode は短文 61.5 tok/s、32K で約 40 tok/s)。品質をさらに優先するなら `--glm-hyt /mnt/NVME4TB/darask-inference-engine/local/hayate/hyt/glm53q-p150q34-packed`(150 GB、KLD 0.0496 / top-1 89.1%、decode 約 25 tok/s)。GPTQ の `glm53q-q4-gptq`(181 GB、KLD 0.0363)と `-rawdense` は `/mnt/SSD16TB/hayate/hyt` に移した)、`--glm-source /mnt/NVME4TB/darask-inference-engine/local/hayate/src/GLM-5.3-Flash-dealignai-FP8`(tokenizer.json)、`--glm-frequencies /mnt/NVME4TB/darask-inference-engine/local/hayate/spill/glm53-stage6/frequencies.bin`、`--glm-context 131072`、`--glm-vram-gib 0`(自動: 空き − 7 GiB、最大 88 GiB)、`--glm-host-gb 0`(自動: MemTotal − 24 GiB まで)、`--glm-prefill-chunk 8192`。
- decode はデバイス側の経路(1 トークン = 1 本の CUDA graph、第19段階)。エキスパートの GPU 配置は文脈に合わせて動く(第20段階): prefill のあと 4 トークンを decode した時点で、その 4 トークンのルート(同点はプロンプトの直近 32,768 トークンのルートか起動時の頻度)で 1 回入れ替える(数百〜3,000 個、0.2〜2 秒)。GPU のスロットがエキスパートの 85% 以上あるとき(f100 で文脈が短いとき)は、その 4 トークンが実際に使ったエキスパートだけを入れる。

## LoRA

```http
POST /v1/models/load {"id":"glm53","loras":[{"path":"/home/hiroshi/ダウンロード/glm-5.3-my-lora-v3.gguf","strength":1.0}]}
```

- `loras` はそのアダプタの組で読み込む指定(画像の要求の `loras` と同じ形、`strength` の既定 1)。違う組で載っていれば降ろして読み直す(約 80 秒。保存済みの会話のプレフィクスも消える)。`[]` は LoRA なし。`loras` を省いた要求は、載っていればそのまま、未読込なら起動オプション `--glm-lora FILE.gguf[:STRENGTH]`(繰り返し可、既定なし)の組で読み込む。LLM は降ろされないので、一度読み込めば serve の再起動まで続く。
- `/v1/models` の glm53 行に `"loras":[{"path":…,"strength":…}]`。
- 形式は llama.cpp の GGUF LoRA(`general.type = adapter`、`adapter.type = lora`、architecture `glm5-next`、F32 / F16 / BF16)。scale は llama.cpp と同じ strength × alpha / rank(alpha が 0 なら strength)。
- 対象は `attn_output`(o_proj)、`ffn_down_shexp`(共有エキスパートの down)、`ffn_down_exps`(288 エキスパートそれぞれの down)、層 0〜2 の `ffn_gate` / `ffn_up` / `ffn_down`。heretic の LoRA(`attn.o_proj, mlp.down_proj, mlp.down_proj.shared`)はこれで全部。それ以外(decode のカーネルが融合している KDA の f_b / g_b、MLA の kv_b、エキスパートの gate / up など)と MTP 層(blk.45)は読み込み時にエラー。PEFT の safetensors は未対応。
- 重みには混ぜず、y += s·B(A x) を元の線形の後に足す(量子化をやり直さない。小さな差分が BF16 / 4 bit の丸めで消えない)。GPU の decode(CUDA graph と per-op)、prefill のグループ GEMM、CPU のエキスパートがすべて同じ式で足す。GPU 側は F32 で和をとり、出力の BF16 に 1 回丸める。CPU 側は down の F32 出力に足してから丸める。
- VRAM はアダプタ(rank 1 の heretic で約 70 MB)と作業域(8192 行 × rank × 4 バイト / 層)。重みとして予算に入るので、その分 GPU のエキスパートが少し減る。
- MTP とは併用できない(serve は MTP を使っていない)。
- 実測(2026-10-20、`glm-5.3-my-lora-v3.gguf` = heretic の rank 1、`glm53q-p150q34-packed`、512 トークンの固定コーパスの教師強制、prefix 64、device decode):
  - LoRA なし → あり: KLD 0.0273、top-1 一致 92.6%、平均 NLL −0.0089(prefix も decode で通すと 0.0266 / 91.1%)。
  - strength 0: エキスパートの予算をそろえる(`--vram-gib 84` と `84 + 66 MiB`)と、448 行と prefix の logits が LoRA なしとビット一致。そろえないと LoRA の 69 MB の分 GPU のエキスパートが約 5 個減り、CPU と GPU の丸めの差で KLD 0.0047。
  - prefill と decode の食い違い(prefix を prefill で通すか decode で通すか): LoRA なし KLD 0.0107、あり 0.0086 で同程度(prefill のグループ GEMM と decode / CPU エキスパートの LoRA が同じに効いている)。
  - 速度は別プロセスの CPU 負荷で CPU エキスパートの時間がぶれて比べられなかった(200 トークンの生成で LoRA なし 9.5、あり 14.8 tok/s)。追加の計算は 1 トークンあたり小さなカーネル 134 本(rank 1)。

## 一覧と能力

- `/v1/capabilities`: `models[]` に `{"id":"glm53","kind":"chat","variants":["fp8"],"context_length":131072,"resident":true,"elastic_vram":true}`、`chat.models` に `glm53`、`chat.glm53` に下の対応表。
- `/v1/models`: `available` に `glm53`、`context_length.glm53`。載っていれば `resident[]` に
  `{"id":"glm53","variant":"fp8","vram_bytes":…,"context_length":131072,"budget":{…}}`。
- `/v1/gpu`: `serve.models[]` の glm53 行(`"slot":false`)に同じ `budget`:

```json
{"budget_bytes": 64424509440, "default_budget_bytes": 64424509440, "shrunk": false,
 "experts_gpu": 2051, "experts_total": 12096, "decode_tok_s": 10.4}
```

`shrunk: true` の間が「GLM 縮小中」。`decode_tok_s` は直前の返答の実測(予測ではない)。

## リクエスト

```json
{"model": "glm53", "messages": [...], "tools": [...], "stream": true,
 "max_tokens": 4096, "temperature": 1.0, "top_p": 0.95, "top_k": 0, "min_p": 0, "seed": 1,
 "stop": ["..."], "enable_thinking": true, "reasoning_effort": "high", "reasoning_budget": 2048}
```

| キー | 既定 | 説明 |
|---|---|---|
| `messages` | 必須 | `system` / `developer`(system 扱い) / `user` / `assistant` / `tool`。`content` は文字列か text パートの配列(画像・音声は 400)。`null` は空文字 |
| `tools` | なし | OpenAI 形式。JSON Schema の `type: "string"` の引数は文字列のまま、それ以外は JSON として読む |
| `tool_choice` | `"auto"` | `"auto"` だけ受ける(ほかは 400) |
| `max_tokens` / `max_completion_tokens` | 窓の残り、最大 32768 | `プロンプト + max_tokens > 131072` は **HTTP 400**(`prompt (N tokens) + max_tokens (M) exceeds server context (131072 tokens)`)。応答の前に返るので縮めて再送できる |
| `temperature` | 1.0 | 0 で greedy。GLM のカード推奨は 1.0 / top_p 0.95 |
| `top_p` | 0.95 | |
| `top_k` | 0(無効) | |
| `min_p` | 0 | 最尤トークン確率 × min_p 未満を捨てる(vLLM と同じ順: penalty → 温度 → min_p → top_k → top_p) |
| `repetition_penalty` | 1.0 | HF/vLLM 式(プロンプトと返答の全トークン)。generation_config の 1.1 は既定にしていない |
| `seed` | 時刻 | 同じ seed・同じプロンプトで同じ返答 |
| `stop` | なし | 文字列か配列。当たると `finish_reason: "stop"` |
| `enable_thinking` | true | トップレベルでも `chat_template_kwargs` の中でもよい。false で `<think></think>` から始める(過去の assistant ターンと同じ形)。`reasoning_effort: "none"` も思考なし |
| `reasoning_effort` | max | `low` / `high`、それ以外は max(チェックポイントのテンプレートどおり)。トップレベルか `chat_template_kwargs` |
| `reasoning_budget`(別名 `thinking_budget`) | なし | 思考がこのトークン数に達したら `</think>` を差し込んで回答へ進める。思考つきの暴走対策 |
| `clear_thinking` | true | `chat_template_kwargs` かトップレベル。true なら最後の user より前の assistant の思考を描画しない(dealignai の既定) |

`logprobs`、`logit_bias`、`response_format`、`presence_penalty`、`frequency_penalty`、`n != 1` は 400。

## 返答

- ストリームは最初に `{"delta":{"role":"assistant"}}`、思考は `delta.reasoning_content`、本文は `delta.content`、ツール呼び出しは閉じタグの時点で 1 つずつ `delta.tool_calls: [{"index":i,"id":"call_…_i","type":"function","function":{"name":…,"arguments":"{…}"}}]`(一度に全体。Qwen3.8 と同じ)。
- 最後のチャンクに `finish_reason`(`stop` / `length` / `tool_calls`)と `usage`(`prompt_tokens`、`completion_tokens`、`total_tokens`、`prompt_tokens_details.cached_tokens` = 再利用したプロンプトのトークン数)と `timings`(`prefill_ms`、`decode_tok_s` など)。続いて `data: [DONE]`。
- ストリームの途中のエラーは最後の `data: {"error":{"message":…}}` だけ(その後 `[DONE]` は来ない)。応答前のエラーは 4xx/5xx と `{"error":{"message":…}}`。
- 非ストリームは `chat.completion`(`message.content` / `reasoning_content` / `tool_calls`)。

## 思考とツールの約束

- GLM のツール呼び出しは `<tool_call>名前<arg_key>k</arg_key><arg_value>v</arg_value>…</tool_call>`。serve が OpenAI の `tool_calls` に変換する。並列呼び出しはそれぞれ別の `index` と一意の `id`。
- 壊れた呼び出しも落とさない: JSON にならない値は文字列のまま、タグ外のゴミは `arguments._raw` に入れ、`</tool_call>` が無いまま終わった呼び出しも返す。`arguments` は常に正しい JSON 文字列。`max_tokens` で途中で切れた呼び出しだけは返さない(`finish_reason: "length"`)。
- 履歴の `tool_calls[].function.arguments` は JSON 文字列でもオブジェクトでもよい。JSON として読めない文字列はそのまま呼び出しの中に描画する。
- `tool` メッセージは直前の assistant の `tool_calls` の順に並べ替えて描画する(全部に `tool_call_id` があり、重複がなく、すべて対応するときだけ。チェックポイントのテンプレートどおり)。
- `reasoning_content` を送り返す必要はない。送らなければ過去のターンは `<think></think>` になり、プレフィクスの再利用はそのまま効く。
- テンプレートの描画は transformers の `apply_chat_template` とバイト単位で一致(`python-tools/hayate_tools/serve_template_check.py --model glm53`、868 ケース)。違いは上の 3 つの約束(思考なし、`null`、`developer`)だけ。

## プレフィクスの再利用

- 直前の会話をそのまま伸ばす(Studio の追記だけの使い方)なら、追加分だけを prefill する。返答の思考が履歴から消えても、前回のプロンプトの終わりのチェックポイントから続ける。
- 保存する点: プロンプトの終わり、思考の終わり(`</think>`)、返答の終わり、最新の user/tool ターンの前、2048 トークン以上のシステム+ツール部分の後(要約で履歴を書き換えたときもシステム部分は再利用)。1 点 136 MiB(固定長の再帰状態)。
- 4 会話まで覚える(コーディングと指揮官の要求が交互に来ても互いを消さない)。KV は 1024 トークンのブロックを会話間で共有し、切り替えでは違うブロックだけ書き戻す。
- 復元した続きは冷えた実行と**ビット一致とは限らない**(第20段階から)。GPU に置くエキスパートは会話ごとに動き、復元では戻さない(戻すと 1 回 2 秒近い)ので、同じエキスパートを CPU と GPU のどちらで計算したかで丸めが変わる(相対 1e-4)。品質は同じ。同じ seed でも直前の会話によって返答が変わりうる。
- 中断(ストリームを切る)しても、それまでに保存したチェックポイントと KV は残る。次の要求は最後に完了したメッセージまで再利用する。

## 中断と同時要求

- ストリームの接続を切れば止まる。serve は 0.2 秒ごとに接続を見ており、decode 中は次のトークンで止まる(実測: 切った直後の次の要求は自分の prefill 3.4 s + 0.3 s で始まった)。prefill 中はチャンク(既定 8192 トークン)の切れ目で止まるので、長いプロンプトの途中では 1 チャンク分(今の速度で最大 1〜2 分)かかる。非ストリームも接続が切れれば止まる。
- 要求は GPU ワーカー 1 本で FIFO に処理する。待っている間は応答ヘッダが返らないだけで、503 / 429 は返さない(キュー 32 件が埋まれば送信側が待つ)。

## GPU の共有(伸縮)

- 読み込み時に `--glm-vram-gib` まで使う(既定 0 = 自動: 空き − 2 GiB、最大 90 GiB。`Options::vram_budget`)。重み・状態・エキスパートキャッシュ・作業領域(8K prefill 分 8.3 GB を含む)の合計。
- 枠の待ち手(`gpu.d` の申告)の先頭が枠を持つか枠が空いていて、空き VRAM が申告(+1 GiB)に足りないとき、
  serve は GLM を `Executor::set_vram_budget` で縮める
  (順位の低い GPU エキスパートを RAM へ)。枠が空いて待ち手がいなくなれば読み込み時の予算まで戻す。
  どちらも要求の合間(ワーカーが 2 秒アイドル)に行う。長い返答の途中では縮まない。
- `/v1/gpu` と `/v1/models` の `budget.shrunk` が「GLM 縮小中」、`experts_gpu` が GPU 上のエキスパート数。
- 実機(申告 50 GiB の待ち手、空き 30.9 GiB): GLM が 65.1 → 43.5 GB(エキスパート 3,494 → 2,022)に 0.29 s で縮み、
  空きは 51.0 GiB。縮小中もチャットは動く(外したエキスパートは RAM が満杯ならディスク段に落ちるので遅くなる)。
  待ち手が消えると 7.6 s で 65.9 GB(3,544)に戻った。

## 実測(2026-10-18、第20段階、自動予算 = 83.3 GB、GPU のエキスパート 4,753 / 12,096)

`glm53q-q4-gptq` の値(rawdense は約 8% 遅い)。e2e の `decode_speed`(256 トークン)は 20.7 tok/s(第18段階の 3.3)。
エージェントループ(下と同じ 34K + 約 2.2K/ターン、思考なし、`max_tokens` 32):

| ターン | プロンプト | 再利用 | TTFT | decode tok/s | 配置の入れ替え |
|---:|---:|---:|---:|---:|---|
| 1 | 34,157 | 0 | 28.2 s | 13.5 | 393 個 0.28 s |
| 2 | 36,379 | 34,188 | 4.5 s | 15.1 | 532 個 0.35 s |
| 3 | 38,591 | 36,409 | 4.6 s | 17.3 | 262 個 0.18 s |
| 4 | 40,805 | 38,621 | 4.6 s | 16.4 | 244 個 0.16 s |

配置を動かさない第19段階は同じ条件で 12.5 / 14.8 / 14.2 / 13.0 tok/s。コード生成(1K)は 11.0 → 29.8 tok/s(第20段階、run の bench)。

## 実測(2026-10-17、RTX PRO 6000、`glm53-stage6-fp8-dense`、phase18/glm53r e86dc59 のランタイム)

ほかのエージェントの CPU 負荷(量子化評価が P コアを使用)がある状態の値。速度はランタイム側の課題。

| 項目 | 値 |
|---|---|
| 読み込み | 77〜89 s |
| VRAM(コンテキスト 131072) | 65.1 GB(60.6 GiB、nvidia-smi のプロセス 61.0 GiB)。会話が 47K トークンに伸びても変わらない |
| うち | 重み 9.18 GB、状態・作業領域 3.62 GB、GPU エキスパート 49.5 GB(3,494 / 12,096) |
| RAM | エキスパート 80.0 GB(pinned)、潜在 KV 1.48 GB(128K 分、pinned)、チェックポイント約 0.3〜2 GB。残り 42 GB はディスク段 |
| decode(短い文脈) | 3.3 tok/s(HTTP 実測も 3.3) |
| decode(34〜47K 文脈) | 0.4〜0.55 tok/s |
| prefill(冷えた 34,157 トークン) | 415 s(82 tok/s) |

エージェントループ(システム+ツール 34K トークン、1 ターンごとに約 2.2K トークン追加、思考なし、`max_tokens` 32):

| ターン | プロンプト | 再利用 | 新たに prefill | TTFT |
|---:|---:|---:|---:|---:|
| 1 | 34,157 | 0 | 34,157 | 416.1 s |
| 2 | 36,365 | 34,188 | 2,177 | 62.9 s |
| 3 | 38,588 | 36,396 | 2,192 | 66.1 s |
| 4 | 40,807 | 38,619 | 2,188 | 67.1 s |
| 5 | 42,985 | 40,838 | 2,147 | 65.8 s |
| 6 | 45,195 | 43,016 | 2,179 | 68.1 s |
| 7 | 47,406 | 45,226 | 2,180 | 67.1 s |

(8〜10 ターン目は計測の時間切れ。冷えた状態からなら 7 ターン目は 47K の全 prefill で約 580 s。)
同じプロンプトを別の会話の後にもう一度送ると、チェックポイントと KV ブロックの書き戻しだけ(6 ms)で、出力は冷えた実行とビット一致(第19段階まで)。

## 確かめ方

```sh
# テンプレート(transformers と 868 ケース比較、GPU 不要)
PYTHONPATH=python-tools python3 -m hayate_tools.serve_template_check --hayate target/debug/hayate --model glm53
# 実機(GPU は gpu_slot を通す。ポート 7874)
target/release/hayate serve --port 7874 --gpu-lock none --idle-unload 0 &
PYTHONPATH=python-tools python3 -m hayate_tools.glm53_serve_e2e --port 7874 --load [--bench]
```

e2e は capabilities / models / gpu の行、日本語の返答、思考予算、ツール呼び出しの往復、400、中断、FIFO、
別の会話の後の復元が冷えた実行の続きになること(先頭 16 文字以上一致)、decode 速度を確かめる。`--bench` は上のエージェントループ。
