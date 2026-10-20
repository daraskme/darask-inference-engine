# GLM のオフラインツール

`PYTHONPATH=python-tools` を設定し、`nix develop .#tools --impure` の Python 環境で実行する。

- `python3 -m hayate_tools.glm53_convert --help`: 元チェックポイントから HYT へ変換
- `glm53_exl3` / `glm53_nvfp4`: 量子化
- `glm53_quant_study`: NVFP4 の方式・容量・重み誤差・クリッピングを実重みで比較。
  `glm53_convert --scale-search mse-ge1` の調査・採用理由は [量子化調査](../docs/GLM53_QUANT_RESEARCH.md)
- `glm53_reference` / `glm53_eval` / `glm53_compare`: 参照出力・品質比較
- `glm53_serve_e2e`: 実機サーバーの検証
- `serve_template_check`: transformers と GLM チャットテンプレートの一致確認
- `gpu_slot`: 別プロセスの GPU 共有

`tests/` に GLM・量子化のテストと HYT の検証データを置く。
