# CSEG: Cross-Source Evidence Grounding

Training and evaluation code for **CSEG-RM**, a six-layer judge that checks
whether a search agent's answer is grounded in, and correctly reconciles,
evidence from multiple retrieved sources, and for **CSEG-Bench**, the
1,160-example benchmark used to evaluate it.

<!-- TODO: paper link, authors, venue -->
Paper: _coming soon_ ·
Models & data: [Hugging Face collection](https://huggingface.co/collections/VectorZhao/cseg-rm)

Seed trajectories come from a large commercial search assistant. Its
proprietary chat template and tool protocol are not needed to train or
evaluate CSEG-RM. CSEG-Bench `messages` drop the original `role=system`
turn (assistant identity and tool protocol); the scorer already ignores
that role. User / tool / assistant turns are unchanged.

## Released artifacts

| Artifact | Paper row | Hugging Face |
|---|---|---|
| CSEG-Bench (n = 1,160) | Table 1 / Table 2 eval set | [`VectorZhao/CSEG-Bench`](https://huggingface.co/datasets/VectorZhao/CSEG-Bench) |
| Qwen3-4B + 32B-RS SFT (`global_step_220`) | Table 1 SFT | [`VectorZhao/CSEG-RM-SFT-4B`](https://huggingface.co/VectorZhao/CSEG-RM-SFT-4B) |
| Qwen3-8B + 32B-RS SFT (`global_step_220`) | Table 1 SFT | [`VectorZhao/CSEG-RM-SFT-8B`](https://huggingface.co/VectorZhao/CSEG-RM-SFT-8B) |
| Qwen3-4B SFT + GRPO-400 | Table 1 SFT+GRPO (main result) | [`VectorZhao/CSEG-RM-SFT-GRPO-4B`](https://huggingface.co/VectorZhao/CSEG-RM-SFT-GRPO-4B) |
| Qwen3-8B SFT + GRPO-400 | Table 1 SFT+GRPO | [`VectorZhao/CSEG-RM-SFT-GRPO-8B`](https://huggingface.co/VectorZhao/CSEG-RM-SFT-GRPO-8B) |
| Qwen3-4B GRPO-400 from base | Table 2 synthesized | [`VectorZhao/CSEG-RM-GRPO-4B`](https://huggingface.co/VectorZhao/CSEG-RM-GRPO-4B) |
| Qwen3-8B GRPO-400 from base | Table 2 synthesized | [`VectorZhao/CSEG-RM-GRPO-8B`](https://huggingface.co/VectorZhao/CSEG-RM-GRPO-8B) |

All checkpoints are bf16 Hugging Face safetensors, thinking-on, and emit the
full JSON judgement (`--prompt-output-spec full`).

`scripts/download_hf.sh` puts each artifact where the configs and launchers
expect it:

```bash
bash scripts/download_hf.sh bench            # -> benchmark/eval.jsonl
bash scripts/download_hf.sh sft-4b           # -> training/sft/ckpts/sft_qwen3_4b_32b_rs/global_step_220/huggingface
bash scripts/download_hf.sh sft-grpo-4b      # -> training/ckpts/sft_grpo_qwen3_4b/global_step_400/actor/huggingface
bash scripts/download_hf.sh all              # everything (~75 GB)
```

## Pipeline

```
Qwen3-4B / Qwen3-8B
    → 32B-RS SFT          (Table 1 SFT rows)
    → merge FSDP shards   (SFT global_step_220 → huggingface/)
    → GRPO-400            (Table 1 SFT+GRPO rows)
    → merge FSDP shards
    → vLLM serve
    → CSEG-Bench          (n = 1,160)
```

Table 2 is GRPO-from-base on the same synthesized split versus a
natural-rollout ablation split of the same size.

The 32B rejection-sampling teacher run is not part of this repository. The
CSEG-RM SFT data is the finished parquet (thinking-on, full JSON). Training
starts from the public `Qwen/Qwen3-4B` or `Qwen/Qwen3-8B` weights.

## Layout

```
data_filter/     node prompts, router, trajectory scorer
benchmark/       CSEG-Bench gold schema + metrics + runner
training/        SFT / GRPO recipes, layer reward, launchers
scripts/         Hugging Face download helper
data/            paper-name → path index (files live next to consumers)
ckpts/           checkpoint path index
```

`training/rewards/layer_reward.py` loads the JSON parser from
`data_filter/score_trajectory.py`, and `benchmark/predict.py` walks the same
six-layer pipeline, so training reward and evaluation parse outputs
identically.

## Runtime contract (needed to train and to match the tables)

| Piece | File | Role |
|---|---|---|
| Node prompts + router | `data_filter/prompt_config.json` | Per-layer template and `routing` / `label_mapping`. The paper recipe uses the English file. |
| Schema | `benchmark/gold.py` | `GOLD_PATHS`, `NODE_VERDICTS`, `LAYER_ORDER`. Judge JSON puts `verdict` first. |
| Trajectory scorer | `data_filter/score_trajectory.py` | `parse_trajectory`, `score_record`, `_parse_json_from_text`. |
| Prompt variants | `data_filter/prompt_variants.py` | `full` (paper) vs `verdict_only`. |
| Layer reward | `training/rewards/layer_reward.py` | GRPO `compute_score`; same parser as eval. |

## Data files

| Paper name | In this repo | Full n | Path |
|---|---|---|---|
| CSEG-Bench | download (`scripts/download_hf.sh bench`) | 1,160 | `benchmark/eval.jsonl` |
| CSEG-RM SFT | subset, 75 / 14 | 14,081 / 795 | `training/sft/data/sft_{train,dev}.parquet` |
| CSEG-RM GRPO | subset, 75 / 15 | 7,320 / 911 | `training/data/rl_{train,val}.jsonl` |

`eval.jsonl` (~118 MB) is over GitHub's file-size limit, so it is
fetched from the Hub rather than committed. The released file already
omits the source assistant's `role=system` turn.

The CSEG-RM train subsets are a stratified draw (seed 42): 5 rows per
`(layer, verdict)` cell (15 cells × 5 = 75). Dev/val is 1 row per available
cell (SFT dev has 14 cells because the full SFT dev split has no
`L5_conflict`/`fail`). They are enough to inspect the schema and run the
launchers end to end; do **not** report Table 1 / Table 2 numbers from them.
To reproduce the tables, put the full CSEG-RM splits at the **same paths**.

Optional Table 2 ablation: `training/data/rl_{train,val}_baseline_qwen3_8b_rollout.jsonl`.

See `data/README.md`, `training/sft/data/README.md`, and `training/data/README.md`.

## Configs ↔ paper tables

| Paper row | Config | Init |
|---|---|---|
| Table 1 · Qwen3-4B + 32B-RS SFT | `training/sft/configs/sft_qwen3_4b.yaml` | `Qwen/Qwen3-4B` |
| Table 1 · Qwen3-8B + 32B-RS SFT | `training/sft/configs/sft_qwen3_8b.yaml` | `Qwen/Qwen3-8B` |
| Table 1 · Qwen3-4B + SFT + GRPO-400 | `training/configs/sft_grpo_qwen3_4b.yaml` | SFT `global_step_220` |
| Table 1 · Qwen3-8B + SFT + GRPO-400 | `training/configs/sft_grpo_qwen3_8b.yaml` | SFT `global_step_220` |
| Table 2 · synth, from-base 4B | `training/configs/grpo_qwen3_4b.yaml` | `Qwen/Qwen3-4B` |
| Table 2 · synth, from-base 8B | `training/configs/grpo_qwen3_8b.yaml` | `Qwen/Qwen3-8B` |
| Table 2 · natural rollout, 4B | `training/configs/grpo_qwen3_4b_rollout.yaml` | `Qwen/Qwen3-4B` |
| Table 2 · natural rollout, 8B | `training/configs/grpo_qwen3_8b_rollout.yaml` | `Qwen/Qwen3-8B` |

Checkpoint directories (same paths as `scripts/download_hf.sh`):

- `training/sft/ckpts/sft_qwen3_{4b,8b}_32b_rs/global_step_220/huggingface`
- `training/ckpts/<config-name>/global_step_400/actor/huggingface`

## Environment

Pinned stack: `verl==0.8.0`, `vllm==0.11.0` (see `requirements.txt`).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # edit HF_HOME / OPENAI_API_BASE if needed
set -a && source .env && set +a
```

Optional: `export VENV=/path/to/venv` so the launch scripts activate it.

Paper hardware:

- SFT: 1 node × 8 GPUs
- GRPO: 4 nodes × 8 GPUs (`trainer.nnodes: 4` in the YAML). Override `trainer.nnodes=<n>` if your cluster differs.

On some multi-node InfiniBand fabrics you must set `NCCL_CUMEM_ENABLE=0`
(the launchers already export it). Use `--config-dir`, not `--config-path`:
a relative `--config-path` is resolved against the verl package directory.

`scripts/launch_grpo.sh` re-roots data / init / reward / checkpoint paths
from the YAML. It does **not** hardcode dataset filenames. Hardcoding
`data.train_files` as a Hydra override silently trains every ablation on
the synthesized split.

## Train

SFT (thinking-on, full JSON, 3 epochs; the paper reports epoch 1 / `global_step_220`):

```bash
cd training/sft
FOREGROUND=1 bash scripts/launch_sft.sh sft_qwen3_4b
# FOREGROUND=1 bash scripts/launch_sft.sh sft_qwen3_8b
```

verl saves FSDP shards plus a `huggingface/` directory that holds only the
config and tokenizer. Merge the shards before using the SFT checkpoint as the
GRPO init:

```bash
cd training
CKPT_ROOT=./sft/ckpts/sft_qwen3_4b_32b_rs bash scripts/merge_fsdp_ckpts.sh global_step_220
```

Or skip SFT and download the released SFT export: `bash scripts/download_hf.sh sft-4b`.

Main-table GRPO (reads the merged SFT export at the path in the YAML):

```bash
# Ray cluster must already be up (RAY_ADDR, default 127.0.0.1:6379)
cd training
FOREGROUND=1 bash scripts/launch_grpo.sh sft_grpo_qwen3_4b
```

Table 2 from-base / ablation: pass the matching config name to `launch_grpo.sh`.
Extra arguments to either launcher are passed to Hydra as overrides.

Merge GRPO FSDP shards after a run:

```bash
cd training
CKPT_ROOT=./ckpts/sft_grpo_qwen3_4b bash scripts/merge_fsdp_ckpts.sh global_step_400
```

## Smoke tests

These run on the shipped subsets in a few minutes and check that every stage
starts, steps, saves, and merges. They are not table runs.

```bash
# SFT: one step on 8 GPUs, checkpoint to local disk
cd training/sft
FOREGROUND=1 bash scripts/launch_sft.sh sft_qwen3_4b \
    trainer.total_epochs=1 trainer.default_local_dir=/tmp/cseg_smoke/sft_qwen3_4b

# GRPO: two steps, no checkpoints, no validation
cd training
bash ../scripts/download_hf.sh sft-4b
CALIB=2 FOREGROUND=1 bash scripts/launch_grpo.sh sft_grpo_qwen3_4b trainer.nnodes=1

# CSEG-Bench: first 20 records against a served checkpoint
cd benchmark
python run_benchmark.py --model qwen3-4b-sft-grpo --limit 20
```

On the 14-row SFT dev subset, `val/loss` is logged as `nan`: verl's SFT
validation loader drops incomplete batches of `train_batch_size / n_gpus`
rows per rank, so each rank sees no validation batch. The full dev split
does not have this problem.

## Evaluate (Table 1 / Table 2)

Serve the merged weights, then run the six-layer judge on CSEG-Bench.
The released checkpoints are thinking-on and use `--prompt-output-spec full`.

```bash
bash scripts/download_hf.sh bench sft-grpo-4b
vllm serve training/ckpts/sft_grpo_qwen3_4b/global_step_400/actor/huggingface \
    --port 8012 --served-model-name qwen3-4b-sft-grpo

cd benchmark
set -a && source .env && set +a   # OPENAI_API_BASE=http://localhost:8012/v1
python run_benchmark.py \
    --eval-set ./eval.jsonl \
    --config ../data_filter/prompt_config.json \
    --prompt-output-spec full \
    --model qwen3-4b-sft-grpo \
    --label qwen3-4b-sft-grpo
```

Headline metrics exclude type `L6_fail_consensus` (same exclusion as CSEG-RM
training). Gold is `target_label` / `GOLD_PATHS[type]`.

## Not included

- Full CSEG-RM train/dev splits (only the stratified subsets above)
- Data-source ablation jsonl
- 32B teacher sampling that produced the SFT parquet
- Dataset *construction* code (synthesis, flattening, balancing)
- Downstream end-policy agent RL
- Cluster-specific job scripts

## Citation

<!-- TODO: BibTeX -->

## License

Code in this repository is [Apache-2.0](LICENSE). Released CSEG-RM
checkpoints on the Hub use the same license. CSEG-Bench is a separate
dataset release (`VectorZhao/CSEG-Bench`); see that card for its terms.
