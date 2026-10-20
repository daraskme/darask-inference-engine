# 統合後の検証

検証日：2026-10-20 UTC。GPU：RTX PRO 6000 Blackwell 96GB。
ネイティブOrnithは、短い固定コーパスで下記の品質ゲートを通過しました。
現時点では、比較基準のllama.cpp版の方が高速です。

## ビルドとCPU

ルートからHayate・Tsumuji・ネイティブKogarashiをCUDA込みでビルドしています。
RustのCPUテスト203件、KogarashiのPythonテスト28件、GPU枠のPythonテスト13件が通過しました。
GPUやローカルの実重みが必要なテストは、CPUチェックでは実行していません。
量子化研究の参照デコーダを使うPythonテストは、固定版バックエンドを取得・ビルド済みの場合に実行します。未取得の新規cloneでは、そのテストを理由付きでskipします。
独立したGPU回帰テストでは、コード・JSON生成を交互に各3回行い、生成IDの再現性とlogitsの有限値を確認しました。

```sh
nix develop
./darask check
```

## 全語彙の品質比較

同じOrnith-1.5-122B-Q4_K_M.gguf、FP32 KV、MTP無効で比較しました。
基準はllama.cppの固定コミット9aa51ce4fdd3435a91531ffe16a17414ccc6dc6fです。
英語・日本語・コード・JSONの入力と基準版の生成IDを連結した537トークンをteacher forcingで通し、
各位置の全248,320語彙のFP32 logitsを比較しています。

| 指標 | 結果 |
|---|---:|
| 全位置の有限値チェック | 通過 |
| 平均KLD（基準分布 → 独自版） | 0.0154505 |
| KLD p99 / 最大 | 0.191318 / 0.371552 |
| top-1一致率 | 96.4618% |
| 平均NLL差（独自版 − 基準版） | +0.0017883 nats/token |

最初の移行ゲートには、既存Hayateの条件と同じ平均KLD ≤ 0.15、top-1一致率 ≥ 82%を使用し、両方を満たしました。
これは537位置の移行回帰チェックです。長文や独立した日本語ベンチマークによる品質評価は追加で必要です。
内部の積和と活性化の扱いが異なるため、自由生成のgreedy列は基準版と異なることがあります。

評価用logitsを書き出せます。各行は248,320個のlittle-endian FP32です。

```sh
target/release/kogarashi-native eval --model /path/to/model.gguf \
  --ids input-ids.json --context 1024 --logits output.f32
```

## 速度

コンテキスト容量4096、greedy、128トークン生成、各3反復の中央値です。
ロード時間を除き、最初のトークン以降のdecodeを測っています。
入力は3〜11トークンの短いraw completionです。MTPは両方とも無効です。

| 入力 | Rust＋独自CUDA | 基準llama.cpp |
|---|---:|---:|
| 英語 | 100.9 token/s | 149.8 token/s |
| 日本語 | 100.8 token/s | 146.1 token/s |
| コード | 100.8 token/s | 149.1 token/s |
| JSON | 100.8 token/s | 150.2 token/s |

注意スコア起動数を削減し、要求開始時に作業領域もリセットした最終版の値です。
4課題それぞれで3反復の生成IDが一致しました。測定は逐次実行で、基準版との交互実行ではありません。

## ネイティブ経路

RustがGGUFのQ4_K/Q6_K重みを読み、共有CUDA Driver APIで専用カーネルを起動します。
全48層、Gated DeltaNet、ゲート付き全注意、8/256 routed experts、shared expertを実装しています。
FP32の再帰状態とKVを使用し、単一トークンの経路をCUDA graphとしてキャプチャします。

初期経路はplain decodeです。MTP、バッチprefill、会話先頭の再利用は未実装です。
サンプリングは共有Rust実装、greedyはGPUのargmaxです。
比較基準のMTP対応版は別経路で保持しています。
Hayate・TsumujiについてはCUDAビルドと共通部分の回帰を確認し、今回の統合による速度向上は測定していません。
