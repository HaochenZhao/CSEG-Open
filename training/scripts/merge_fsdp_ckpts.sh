#!/usr/bin/env bash
# Merge verl FSDP sharded checkpoints into HuggingFace safetensors, writing each
# step's merged weights into that step's own `huggingface/` directory.
#
#   CKPT_ROOT=./ckpts/sft_grpo_qwen3_4b bash scripts/merge_fsdp_ckpts.sh
#   CKPT_ROOT=./ckpts/sft_grpo_qwen3_4b bash scripts/merge_fsdp_ckpts.sh global_step_400
set -euo pipefail

TRAIN="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-}"
CKPT_ROOT="${CKPT_ROOT:-${TRAIN}/ckpts/sft_grpo_qwen3_4b}"
[[ "${CKPT_ROOT}" = /* ]] || CKPT_ROOT="${TRAIN}/${CKPT_ROOT#./}"

if [[ -n "${VENV}" ]]; then
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
fi

steps=()
if [[ $# -gt 0 ]]; then
    steps=("$@")
else
    while IFS= read -r d; do
        steps+=("$d")
    done < <(find "${CKPT_ROOT}" -maxdepth 1 -mindepth 1 -type d -name 'global_step_*' | sort -t_ -k3 -n)
fi

for step_dir in "${steps[@]}"; do
    [[ "${step_dir}" = /* ]] || step_dir="${CKPT_ROOT}/${step_dir}"

    shard_dir="${step_dir}"
    [[ -f "${shard_dir}/fsdp_config.json" ]] || shard_dir="${step_dir}/actor"
    target="${shard_dir}/huggingface"

    if [[ ! -f "${shard_dir}/fsdp_config.json" ]]; then
        echo "[skip] ${step_dir}: no fsdp_config.json"
        continue
    fi

    if [[ -n "$(find "${target}" -maxdepth 1 -name '*.safetensors' -print -quit 2>/dev/null)" ]]; then
        echo "[skip] ${target}: already merged"
        continue
    fi

    echo "=============================================================="
    echo "[merge] ${shard_dir} -> ${target}"

    python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "${shard_dir}" \
        --target_dir "${target}"

    echo "[done] ${target}"
done

echo "ALL_MERGES_COMPLETE"
