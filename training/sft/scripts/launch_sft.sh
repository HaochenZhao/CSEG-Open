#!/usr/bin/env bash
# Single-node 8-GPU SFT for a config under training/sft/configs/.
#
#   bash scripts/launch_sft.sh sft_qwen3_4b
#   FOREGROUND=1 bash scripts/launch_sft.sh sft_qwen3_8b
#
# Arguments containing "=" are Hydra overrides applied to every config:
#   bash scripts/launch_sft.sh sft_qwen3_4b trainer.total_epochs=1 trainer.default_local_dir=/tmp/sft
set -euo pipefail

[[ $# -ge 1 ]] || { echo "usage: launch_sft.sh <config-name> [<config-name>...] [hydra overrides...]" >&2; exit 1; }

SFT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-}"
HF_CACHE="${HF_HUB_CACHE:-${HF_HOME:-${HOME}/.cache/huggingface}}"
NPROC="${NPROC:-8}"

configs=()
overrides=()
for arg in "$@"; do
    if [[ "${arg}" == *=* ]]; then overrides+=("${arg}"); else configs+=("${arg}"); fi
done
(( ${#configs[@]} )) || { echo "no config name given" >&2; exit 1; }
set -- "${configs[@]}"

for cfg in "$@"; do
    [[ -f "${SFT}/configs/${cfg}.yaml" ]] || { echo "no such config: configs/${cfg}.yaml" >&2; exit 1; }
done

if [[ -n "${VENV}" ]]; then
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
fi
cd "${SFT}"

export HF_HUB_CACHE="${HF_CACHE}"
export NCCL_CUMEM_ENABLE=0
if [[ "${HF_HUB_OFFLINE:-0}" == "1" ]]; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
fi

mkdir -p "${SFT}/outputs"

read_cfg() {
    python - "$1" "$2" <<'PY'
import os, sys, yaml
cfg, key = sys.argv[1], sys.argv[2]
node = yaml.safe_load(open(f"configs/{cfg}.yaml"))
for part in key.split("."):
    node = node[part]
if isinstance(node, str) and not os.path.isabs(node) and key.startswith("data."):
    node = os.path.normpath(os.path.join(os.getcwd(), node))
print(node)
PY
}

for cfg in "$@"; do
    model_path="$(read_cfg "${cfg}" model.path)"
    train_file="$(read_cfg "${cfg}" data.train_files)"
    val_file="$(read_cfg "${cfg}" data.val_files)"
    [[ -f "${train_file}" ]] || { echo "[preflight] missing train file: ${train_file}" >&2; exit 1; }
    [[ -f "${val_file}"   ]] || { echo "[preflight] missing val file: ${val_file}" >&2; exit 1; }
    python - "${model_path}" <<'PY'
import sys
from transformers import AutoConfig
AutoConfig.from_pretrained(sys.argv[1])
print(f"[preflight] weights OK: {sys.argv[1]}")
PY
    echo "[preflight] ${cfg}: data OK (${train_file})"
done

for cfg in "$@"; do
    project="$(read_cfg "${cfg}" trainer.project_name)"
    experiment="$(read_cfg "${cfg}" trainer.experiment_name)"
    export TENSORBOARD_DIR="${SFT}/tensorboard_log/${project}/${experiment}"
    mkdir -p "${TENSORBOARD_DIR}"
    log="${SFT}/outputs/sft_${experiment}_$(date +%Y%m%d_%H%M%S).log"

    echo "=============================================================="
    echo "  config     ${cfg}"
    echo "  ckpts      $(read_cfg "${cfg}" trainer.default_local_dir)"
    echo "  tb         ${TENSORBOARD_DIR}"
    echo "  log        ${log}"
    echo "=============================================================="

    if [[ -n "${FOREGROUND:-}" ]]; then
        torchrun --standalone --nproc_per_node="${NPROC}" -m verl.trainer.sft_trainer \
            --config-dir configs --config-name "${cfg}" "${overrides[@]}" 2>&1 | tee "${log}"
    else
        nohup torchrun --standalone --nproc_per_node="${NPROC}" -m verl.trainer.sft_trainer \
            --config-dir configs --config-name "${cfg}" "${overrides[@]}" > "${log}" 2>&1 &
        echo "[launch] pid=$! log=${log} (FOREGROUND=1 to run in the foreground)"
    fi
    echo "[done] ${cfg} -> ${log}"
done
