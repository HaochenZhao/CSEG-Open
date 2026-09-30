# CSEG-RM GRPO split

This repo ships a **stratified subset** (seed 42) so the GRPO launcher has
real jsonl to open. Overwrite these files with the full splits to
reproduce Table 1 / Table 2. Do **not** report paper numbers from the
subset.

| File | Shipped n | Full n | Role |
|---|---|---|---|
| `rl_train.jsonl` | 75 (5 per `(layer, verdict)` cell) | 7,320 | Table 1 SFT→GRPO and Table 2 synthesized from-base |
| `rl_val.jsonl` | 15 (1 per cell) | 911 | same, validation (natural mix in the full file) |
| `rl_train_baseline_qwen3_8b_rollout.jsonl` | not shipped | 7,320 | Table 2 natural-rollout ablation (optional) |
| `rl_val_baseline_qwen3_8b_rollout.jsonl` | not shipped | 911 | same, validation |

The full train is balanced per `(layer, verdict)` (15 cells × 488). The
subset keeps that 15-cell cover at 5 rows each. Prompt budget is 20,480
tokens.

verl `RLHFDataset` fields: `prompt`, `reward_model.ground_truth` (gold
verdict), `data_source` / `extra_info.layer_id` (layer id).

Configs under `../configs/` already use these relative paths. The
launcher re-roots them; it must **not** hardcode `rl_train.jsonl`, or
the ablation arm trains on the synthesized split.

`train_batch_size: 64` with 75 train rows is one drop-last step — smoke
only.
