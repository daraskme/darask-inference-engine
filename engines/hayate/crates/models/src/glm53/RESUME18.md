# 第18段階 再開メモ(runtime トラック)

- ブランチ phase18/glm53r、worktree .worktrees/glm53r(third_party は main への symlink、commit しない)。
- 計測: /mnt/NVME4TB/hayate/spill/glm53r/s3/chunk.sh(A/B/C 済み、D = 60 GiB async と main/stack/new sync の静かな再測定が
  driver2.sh で gpu_slot 待ち)。結果は s3/status と s3/*.log。
- 未解決: prefill-vs-step row3(STAGE18.md の切り分け)。
- 次: 吸収型 sparse attention(prefill + decode)、select の並列 top-k、借用なしの prefill 先読み、
  elastic 縮小の pinned 予備プール、LOWBIT CPU/GPU カーネル(量子化仕様 docs/GLM53_QUANT.md の確定待ち)。
