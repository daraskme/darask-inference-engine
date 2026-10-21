# 画像入力とキャプションの一括処理

Hayate（GLM-5.3-Flash）、Kogarashi（Ornith-1.5-122B）、Tsumuji（Qwen3.8-Flash-Next）は、モデル自身の視覚エンコーダーで画像を読み、同じモデルの言語デコーダーからテキストを出力します。
推論時の画像デコード・前処理はRust、視覚TransformerとデコーダーはRust＋CUDAです。視覚部分の行列積にはcuBLASを使います。

## 最小の使い方

通常どおりサーバーを起動しておきます。

```sh
nix develop
./darask serve kogarashi
```

別のターミナルで画像フォルダを指定します。

```sh
nix develop
./darask batch kogarashi \
  --input /path/to/images \
  --out local/captions.jsonl \
  --jobs 2 \
  --prompt '画像の内容を、見えている事実に沿って日本語で説明してください。'
```

`hayate`、`tsumuji`にも同じ`batch`コマンドが使えます。Hayateはバッチ開始時にモデルの読み込みも行います。PNG、JPEG、WebP、BMP、PNMを再帰的に探します。
既定値は2件並列、最大256出力トークン、自動リトライ2回です。`--jobs 1`なら完全に順次送信します。

**`--jobs 2`は2件のHTTPリクエストを並列に送信する設定です。** GPUデコーダーは一つの重みと実行状態を使って順次処理します。
2つの巨大モデルをGPUに重複ロードする設定ではありません。画像の転送・CPU前処理を、別の画像のGPU処理と進められます。

各画像が終わるたびに結果をJSONLへ保存します。順番は完了順で、`image`に元画像の絶対パス、`caption`に文章、`status`に成否、`response`にAPI応答が入ります。
同じコマンドを再実行すると成功済みをスキップし、未処理・失敗分を処理します。画像のサイズ・更新時刻、モデル・プロンプト・生成設定が変わると再処理します。
途中で切れた最終行は再開時に復旧します。同じ結果ファイルへの同時書き込みはロックで防ぎます。
失敗が残れば終了コード1です。

```sh
./darask batch tsumuji --input /path/to/images --out local/qwen-captions.jsonl \
  --jobs 2 --max-tokens 512 --detail high --retry 2
```

既定の接続先とモデル名は次のとおりです。

| エンジン | API | モデル名 |
|---|---|---|
| Hayate | `http://127.0.0.1:7870/v1` | `glm53` |
| Kogarashi | `http://127.0.0.1:8080/v1` | `kogarashi-ornith` |
| Tsumuji | `http://127.0.0.1:8090/v1` | `qwen38` |

変更したサーバーには`--endpoint http://HOST:PORT/v1 --model MODEL_ID`を指定します。

## 視覚重み

HayateとTsumujiは、指定された元のsafetensorsモデルディレクトリの`model.visual.*`を読みます。
テキスト用に変換したHYTやNVFP4の実行経路はそのまま使います。

Ornithの言語モデルGGUFには視覚エンコーダーが含まれないため、配布元の視覚テンソル333個だけを別に用意します。
この作業環境では`local/vision/ornith/`に取得済みです。別の環境では次のコマンドで取得できます。約861 MiBで、言語モデル本体は取得しません。

```sh
python3 tools/fetch-ornith-vision.py
```

配布元の固定リビジョンを使用します。必要なら`--revision COMMIT --out DIRECTORY`で変更できます。
配置を変更した場合は、サーバーを`DARASK_ORNITH_VISION=/path/to/vision ./darask serve kogarashi`で起動します。
モデルの利用条件は配布元のライセンスに従ってください。

## API

`POST /v1/chat/completions`のユーザーメッセージで、OpenAI形式の`image_url`とテキストを順に並べます。
画像は`data:image/png;base64,...`のようなbase64データURLで送ります。HTTP画像URLやサーバー上のファイルパスの読み取りは実装していません。
1リクエストに画像8枚まで対応し、複数ターンの画像も処理します。

```json
{
  "model": "kogarashi-ornith",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
      {"type": "text", "text": "この画像を説明してください。"}
    ]
  }],
  "max_tokens": 256,
  "enable_thinking": false
}
```

既定では画像1枚を最大1024視覚トークンに収めます。`detail: low`では最大256です。`auto`と`high`は最大1024です。
モデル固有の平均・標準偏差、パッチ順、結合方法を使い、Qwen系では画像の高さ・幅に対応したM-RoPEと生成後の位置補正を使います。
画像付きリクエストは画像の違いを取り違えないようテキスト専用のプレフィックスキャッシュから分離し、Tsumujiの画像付き生成ではMTPの先読みを無効にしています。

GPUの容量に合わせてサーバーのコンテキスト長も調整できます。例えばTsumujiは`./darask serve tsumuji -- --ctx 8192`です。
自動コンテキストの見積もりでは視覚重みと一時バッファの余裕を確保します。

## 検証

CPU回帰は`./darask check`。実際の重みを使う検証は、共有GPU枠の中で次を実行します。
PyTorchとTransformersは比較用にのみ使います。

```sh
nix develop .#tools
PYTHONPATH="$PWD/engines/hayate/python-tools" python3 -m hayate_tools.gpu_slot \
  --need 88 --who vision-validation --priority 4 --timeout 600 \
  -- python3 tools/validate-vision.py
```

視覚エンコーダーは正方形・横長の画像で公式TransformersのFP32出力と比較します。
数値比較は同じ前処理済みパッチを使い、FP32のcuDNN／行列積でTF32を無効にしています。リサイズ処理のバイト一致を意味しません。
APIは赤／青の識別、同じ画像の再送、画像の前後でテキスト結果が変わらないこと、複数画像の順序、2並列バッチと再開を確認します。
測定値と実画像の結果は`local/validation/vision/`に保存します。
