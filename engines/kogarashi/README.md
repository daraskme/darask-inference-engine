# Kogarashi

Ornith-1.5-122B（Qwen3.5 MoE）向けのRust＋独自CUDA実装です。
推論本体はllama.cppにリンクせず、GGUF Q4_K/Q6_K重みを直接GPUで計算します。

リポジトリのルートから実行してください。

```sh
nix develop
./darask build kogarashi
./darask serve kogarashi
./darask run kogarashi -- --prompt 'The capital of France is' --tokens 128
```

`--model FILE.gguf`、`--context N`、`--port N`で変更できます。
現在はOrnithの3072幅・48層・256 experts・top-8の構成を対象としています。
Qwen3.8と共通のHTTP、トークナイザー、出力整形を使い、計算カーネルは分離しています。

初期のネイティブ経路はplain decode、FP32の再帰状態とKVです。
MTPとバッチprefillは、この経路にはまだ実装していません。
確認した範囲と性能の比較は[検証記録](../../docs/VALIDATION.md)を参照してください。

比較基準の固定版llama.cppは`upstream/gyro-llama`、CUDAのMTP修正は`patches/`に保存しています。
参照用の取得・ビルド方法と既存の量子化実験は[REFERENCE.md](REFERENCE.md)に残しています。
Pythonコードは参照版の補助操作、評価、量子化研究用で、ネイティブ推論経路からは呼びません。

```sh
./darask reference kogarashi
```
