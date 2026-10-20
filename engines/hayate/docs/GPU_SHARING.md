# GLM と外部ジョブの GPU 共有

GLM は常駐し、`gpu.lock` の外で動く。画像生成・動画生成・学習を実行する別プロセスは、従来どおり `python-tools/hayate_tools/gpu_slot.py` を使える。Hayate 自身の画像・動画 API は削除した。

既定のロックは `/mnt/NVME4TB/darask-inference-engine/local/hayate/spill/gpu.lock`。外部ジョブは隣の `gpu.d/<pid>.json` に `pid`、`who`、`need_gib`、`since` を申告する。`priority`（0 が最優先、既定 5）と `label` も指定できる。長時間待機には 30 分ごとの優先度補正が入る。申告と `flock`、必要 VRAM の待機は `gpu_slot.py` が処理する。

先頭の申告者が枠を持つか枠が空いていて、空き VRAM が `need_gib + 1 GiB` に足りないとき、serve が GLM の GPU エキスパート数を減らす。枠が空き、待ち手が消えると、読み込み時の予算まで増やす。いずれも要求の合間に行い、実行中の会話は割り込まない。

`--yield-idle 2` が調整の既定値（秒）、`0` で調整を無効化する。`--gpu-lock none` でも調整を無効にできる。`GET /v1/gpu` と `GET /v1/models` は `budget.shrunk`、予算、GPU エキスパート数を返す。

GLM 自体は小さな予算でも常駐できる範囲に限界がある。返せる量より申告が大きい場合、外部ジョブは必要な空きができるまで待機する。常駐モデルを自動で降ろす動作はない。
