# Checkpoints

Released weights are on the Hugging Face Hub
([collection](https://huggingface.co/collections/VectorZhao/cseg-rm)).
`bash ../scripts/download_hf.sh <name>` places them at the paper paths below,
which are also where training writes them:

| Paper row | `download_hf.sh` name | Directory |
|---|---|---|
| Qwen3-4B 32B-RS SFT (epoch 1) | `sft-4b` | `../training/sft/ckpts/sft_qwen3_4b_32b_rs/global_step_220/huggingface` |
| Qwen3-8B 32B-RS SFT (epoch 1) | `sft-8b` | `../training/sft/ckpts/sft_qwen3_8b_32b_rs/global_step_220/huggingface` |
| Qwen3-4B SFT+GRPO-400 | `sft-grpo-4b` | `../training/ckpts/sft_grpo_qwen3_4b/global_step_400/actor/huggingface` |
| Qwen3-8B SFT+GRPO-400 | `sft-grpo-8b` | `../training/ckpts/sft_grpo_qwen3_8b/global_step_400/actor/huggingface` |
| Table 2 · Qwen3-4B GRPO-400 from base | `grpo-4b` | `../training/ckpts/grpo_qwen3_4b/global_step_400/actor/huggingface` |
| Table 2 · Qwen3-8B GRPO-400 from base | `grpo-8b` | `../training/ckpts/grpo_qwen3_8b/global_step_400/actor/huggingface` |

After your own training run, merge the FSDP shards into that step's
`huggingface/` directory (SFT and GRPO layouts are both handled):

```bash
cd ../training
CKPT_ROOT=./sft/ckpts/sft_qwen3_4b_32b_rs bash scripts/merge_fsdp_ckpts.sh global_step_220
CKPT_ROOT=./ckpts/sft_grpo_qwen3_4b bash scripts/merge_fsdp_ckpts.sh global_step_400
```

Serve the merged `huggingface/` directory with vLLM, then run
`benchmark/run_benchmark.py` (see the top-level README).
