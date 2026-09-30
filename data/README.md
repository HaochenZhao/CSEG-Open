# Data files

Finished files live next to the code that consumes them (not in this
directory). This page is only an index.

| Paper name | In this repo | Full paper n | Path |
|---|---|---|---|
| CSEG-Bench | download: `bash scripts/download_hf.sh bench` ([`VectorZhao/CSEG-Bench`](https://huggingface.co/datasets/VectorZhao/CSEG-Bench)); proprietary `role=system` turns stripped | 1,160 | `../benchmark/eval.jsonl` |
| CSEG-RM SFT | subset | 14,081 / 795 | `../training/sft/data/sft_{train,dev}.parquet` |
| CSEG-RM GRPO | subset | 7,320 / 911 | `../training/data/rl_{train,val}.jsonl` |

Subset construction (seed 42): 5 rows per `(layer, verdict)` on train
(75 rows); 1 row per available cell on dev/val. Use these to inspect
schemas and smoke-test launchers. Do **not** report Table 1 / Table 2
numbers from the subset.

## Full splits (same paths)

Overwrite the subset parquet / jsonl with the full CSEG-RM splits so
the YAML files resolve without edits. Optional Table 2 ablation files:

| Paper name | n | Copy to |
|---|---|---|
| Data-source ablation | 7,320 / 911 | `../training/data/rl_{train,val}_baseline_qwen3_8b_rollout.jsonl` |

Do not place intermediate synthesis dumps here. Only the finished splits
above are needed to train the table models and to score CSEG-Bench.

See `../training/sft/data/README.md` and `../training/data/README.md`
for file-level schemas.
