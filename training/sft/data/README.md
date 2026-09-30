# CSEG-RM SFT split

This repo ships a **stratified subset** (seed 42) so the SFT launcher has
real parquet to open. Overwrite these files with the full splits to
reproduce Table 1.

| File | Shipped n | Full n | Consumer |
|---|---|---|---|
| `sft_train.parquet` | 75 (5 per `(layer, verdict)` cell) | 14,081 | `verl.trainer.sft_trainer` (`messages` column) |
| `sft_dev.parquet` | 14 (1 per available cell) | 795 | same, validation |

SFT dev has 14 cells, not 15: the full SFT dev split has no
`L5_conflict` / `fail` row. Train covers all 15 cells.

These are 32B rejection-sampled, thinking-on, full-JSON targets. The
full split is by `base_index` (seed 42) so no seed question leaks
between train and dev. The subset inherits that split (it is sampled
from the finished parquet, not re-split).

`sft_qwen3_{4b,8b}.yaml` uses `data.train_batch_size: 64`. The 75-row
subset is just large enough for one drop-last step; it is a smoke test,
not a table run.

Configs: `../configs/sft_qwen3_{4b,8b}.yaml` (`data.train_files: ./data/sft_train.parquet`).
