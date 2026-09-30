#!/usr/bin/env bash
# Fetch CSEG-Bench and released CSEG-RM checkpoints from the Hugging Face Hub
# into the paths the configs, launchers, and README already use.
#
#   bash scripts/download_hf.sh bench                 # benchmark/eval.jsonl
#   bash scripts/download_hf.sh sft-4b sft-grpo-4b    # any subset of the names below
#   bash scripts/download_hf.sh all
#
# Names: bench, sft-4b, sft-8b, sft-grpo-4b, sft-grpo-8b, grpo-4b, grpo-8b
# Needs the `hf` CLI (huggingface_hub >= 0.34).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORG="${HF_ORG:-VectorZhao}"

declare -A REPO=(
    [sft-4b]=CSEG-RM-SFT-4B           [sft-8b]=CSEG-RM-SFT-8B
    [sft-grpo-4b]=CSEG-RM-SFT-GRPO-4B [sft-grpo-8b]=CSEG-RM-SFT-GRPO-8B
    [grpo-4b]=CSEG-RM-GRPO-4B         [grpo-8b]=CSEG-RM-GRPO-8B
)
declare -A DEST=(
    [sft-4b]=training/sft/ckpts/sft_qwen3_4b_32b_rs/global_step_220/huggingface
    [sft-8b]=training/sft/ckpts/sft_qwen3_8b_32b_rs/global_step_220/huggingface
    [sft-grpo-4b]=training/ckpts/sft_grpo_qwen3_4b/global_step_400/actor/huggingface
    [sft-grpo-8b]=training/ckpts/sft_grpo_qwen3_8b/global_step_400/actor/huggingface
    [grpo-4b]=training/ckpts/grpo_qwen3_4b/global_step_400/actor/huggingface
    [grpo-8b]=training/ckpts/grpo_qwen3_8b/global_step_400/actor/huggingface
)

[[ $# -ge 1 ]] || { sed -n '2,10p' "${BASH_SOURCE[0]}"; exit 1; }
names=("$@")
[[ "${names[*]}" == "all" ]] && names=(bench sft-4b sft-8b sft-grpo-4b sft-grpo-8b grpo-4b grpo-8b)

for name in "${names[@]}"; do
    if [[ "${name}" == "bench" ]]; then
        echo "[download] ${ORG}/CSEG-Bench -> benchmark/eval.jsonl"
        hf download "${ORG}/CSEG-Bench" eval.jsonl --repo-type dataset --local-dir "${ROOT}/benchmark"
    elif [[ -n "${REPO[${name}]:-}" ]]; then
        echo "[download] ${ORG}/${REPO[${name}]} -> ${DEST[${name}]}"
        hf download "${ORG}/${REPO[${name}]}" --local-dir "${ROOT}/${DEST[${name}]}"
    else
        echo "unknown name: ${name}" >&2
        exit 1
    fi
done
