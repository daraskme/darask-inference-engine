# 統合後の配置

統合ルートは`/mnt/NVME4TB/darask-inference-engine`です。
元の3ディレクトリの内容は同じファイルシステム内で実体を移動しています。
旧パスにディレクトリや互換リンクは残していません。

| 元の内容 | 現在の配置 |
|---|---|
| Hayateの`code/rapid_Diffusers` | `engines/hayate` |
| Kogarashi | `engines/kogarashi` |
| Tsumuji | `engines/tsumuji` |
| Hayateの重み・Studio・学習コード・その他データ | `local/hayate` |
| 元のHayate/Tsumuji Git管理情報 | `local/import-history` |
| 元GitHub Hayateの全参照ミラー | `local/import-history/github-hayate.git` |

共有Rustコードは`common`へ実体を移動し、ルートの一つのCargoワークスペースとCargo.lockで管理しています。
エンジン名を含む既存crate名は、API互換性のため維持しています。
JSONの構文解析は共通実装に集約し、モデル側の値表現とシリアライズ形式は保っています。
モデル固有の重み配置、状態、精度、サンプリング、カーネルの計算順序は各実行器が管理します。

## Hayate Studio

Studioは別プロジェクトの`/mnt/NVME4TB/hayate-studio`へ整理されています。
このリポジトリ内の`local/hayate`に残るStudio関係の参照は、ローカルの互換用途です。公開しません。
Studioとトンネルのサービス、自動起動、ローカルHTTP 200の応答を確認しています。

移動前の設定ファイルのバックアップは`local/migration-backup`、置換対象の記録は`local/path-migration.json`に保存しています。
私的な研究・レビュー記録、環境、ビルド生成物、重みは削除せずローカルに保持しています。

## 開発

Nixの環境はルートの`flake.nix`へ統一しています。
Rustはルートでビルドし、実行ファイルは`target/release`へ出力します。
元のビルド生成物は残していますが、統合後のバイナリはルートから再ビルドしてください。
過去の性能記録は当時のパス・コード・条件の記録であり、統合後の性能を保証するものではありません。
