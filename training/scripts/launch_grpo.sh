#!/usr/bin/env bash
# Launch a GRPO run from a YAML under training/configs/.
#
#   bash scripts/launch_grpo.sh sft_grpo_qwen3_4b
#   CALIB=10 bash scripts/launch_grpo.sh sft_grpo_qwen3_4b
#   FOREGROUND=1 bash scripts/launch_grpo.sh sft_grpo_qwen3_4b
#
# Extra Hydra overrides are passed through:
#   bash scripts/launch_grpo.sh sft_grpo_qwen3_4b actor_rollout_ref.actor.optim.lr=5e-7
#
# Plumbing that cannot live in the YAML:
#
#   NCCL_CUMEM_ENABLE=0
#       On some multi-node InfiniBand setups, NCCL 2.27 defaults this to 1 and
#       allocates communication buffers through cuMem/VMM. Those addresses cannot
#       be registered with the IB HCA via nvidia_peermem, so large allreduces fail
#       with ibv_reg_mr_iova2 / Invalid argument. The failure often looks like a
#       hang: init_process_group and a small barrier succeed either way.
#
#   HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE (optional)
#       Set both to 1 when workers cannot reach huggingface.co. HF_HUB_CACHE
#       should then point at a local snapshot of the base weights.
#
#   Absolute paths for data, the SFT init, the reward fn, the checkpoint dir,
#       and TENSORBOARD_DIR. verl runs the trainer inside a Ray actor
#       (TaskRunner). With rollout mode=async, compute_score runs in agent-loop
#       actors whose CWD is typically /tmp, not this checkout. Relative paths
#       would then miss the files. This script reads those keys back from the
#       YAML and re-roots them; it does not hardcode dataset filenames.
#       Hardcoding data.train_files as a Hydra override silently beat the YAML
#       and collapsed the data-source ablation onto the synthesized split.
#
#   The "'0'" quoting on runtime_env values: Hydra parses +...env_vars.FOO=0 as
#       an int and Ray requires Dict[str, str].
set -euo pipefail

CONFIG="${1:?usage: launch_grpo.sh <config-name> [hydra overrides...]}"
shift || true

TRAIN="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-}"
HF_CACHE="${HF_HUB_CACHE:-${HF_HOME:-${HOME}/.cache/huggingface}}"
RAY_ADDR="${RAY_ADDR:-127.0.0.1:6379}"

[[ -f "${TRAIN}/configs/${CONFIG}.yaml" ]] || { echo "no such config: configs/${CONFIG}.yaml" >&2; exit 1; }

calib_overrides=()
run_tag="${CONFIG}"
if [[ -n "${CALIB:-}" ]]; then
    run_tag="${CONFIG}_calib"
    calib_overrides=(
        "trainer.total_training_steps=${CALIB}"
        "trainer.experiment_name=${run_tag}"
        "trainer.save_freq=-1"
        "trainer.test_freq=-1"
        "trainer.val_before_train=false"
    )
fi

if [[ -n "${VENV}" ]]; then
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
fi
cd "${TRAIN}"

export HF_HUB_CACHE="${HF_CACHE}"
export NCCL_CUMEM_ENABLE=0
export RAY_ADDRESS="${RAY_ADDR}"
if [[ "${HF_HUB_OFFLINE:-0}" == "1" ]]; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
fi

read_cfg() {
    python - "$CONFIG" "$1" <<'PY'
import sys, yaml
node = yaml.safe_load(open(f"configs/{sys.argv[1]}.yaml"))
for part in sys.argv[2].split("."):
    node = node[part]
print(node)
PY
}

project="$(read_cfg trainer.project_name)"

read -r train_files val_files model_path <<<"$(python - "$CONFIG" "$TRAIN" <<'PY'
import os, sys, yaml
cfg = yaml.safe_load(open(f"configs/{sys.argv[1]}.yaml"))
root = sys.argv[2]
def rerooted(p):
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(root, p))
data = cfg["data"]
model = cfg["actor_rollout_ref"]["model"]["path"]
print(rerooted(data["train_files"]), rerooted(data["val_files"]), rerooted(model))
PY
)"
for f in "${train_files}" "${val_files}"; do
    [[ -f "${f}" ]] || { echo "no such data file: ${f}" >&2; exit 1; }
done
echo "[preflight] train=${train_files}"
echo "[preflight] val=${val_files}"
echo "[preflight] model=${model_path}"

export TENSORBOARD_DIR="${TRAIN}/tensorboard_log/${project}/${run_tag}"
python - "$model_path" <<'PY'
import glob, os, sys
from transformers import AutoConfig
path = sys.argv[1]
AutoConfig.from_pretrained(path)
# verl writes only config + tokenizer into global_step_*/huggingface; the weights
# appear there after scripts/merge_fsdp_ckpts.sh.
if os.path.isdir(path) and not glob.glob(os.path.join(path, "*.safetensors")):
    sys.exit(f"[preflight] no *.safetensors in {path}\n"
             f"[preflight] merge the SFT shards first:  "
             f"CKPT_ROOT={os.path.dirname(os.path.dirname(path))} bash scripts/merge_fsdp_ckpts.sh")
print(f"[preflight] weights OK: {path}")
PY

gpus="$(ray status 2>/dev/null | grep -oE '[0-9.]+/[0-9.]+ GPU' | cut -d/ -f2 | cut -d. -f1 | head -1)"
echo "[preflight] ray cluster at ${RAY_ADDR}: ${gpus:-0} GPUs"
[[ "${gpus:-0}" -ge 8 ]] || { echo "ray cluster not reachable or has fewer than 8 GPUs" >&2; exit 1; }

avail="$(df -BG --output=avail "${TRAIN}" | tail -1 | tr -dc '0-9')"
echo "[preflight] ${avail} GiB free on the checkpoint filesystem"
[[ "${avail}" -ge 200 ]] || echo "[preflight] WARNING: low disk; a 400-step 4B run needs ~848 GiB, 8B ~1.71 TiB"

log="${TRAIN}/outputs/launch_${run_tag}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "${TRAIN}/outputs"

cmd=(
    python -m verl.trainer.main_ppo
    --config-dir configs --config-name "${CONFIG}"
    "reward.custom_reward_function.path=${TRAIN}/rewards/layer_reward.py"
    "+ray_kwargs.ray_init.runtime_env.env_vars.HF_HUB_CACHE=${HF_CACHE}"
    "+ray_kwargs.ray_init.runtime_env.env_vars.NCCL_CUMEM_ENABLE='0'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.TENSORBOARD_DIR=${TENSORBOARD_DIR}"
)
if [[ "${HF_HUB_OFFLINE:-0}" == "1" ]]; then
    cmd+=(
        "+ray_kwargs.ray_init.runtime_env.env_vars.HF_HUB_OFFLINE='1'"
        "+ray_kwargs.ray_init.runtime_env.env_vars.TRANSFORMERS_OFFLINE='1'"
    )
fi

ckpt_name="$([[ -n "${CALIB:-}" ]] && echo "_calib_${CONFIG}" || echo "${CONFIG}")"
for kv in "data.train_files=${train_files}" \
          "data.val_files=${val_files}" \
          "actor_rollout_ref.model.path=${model_path}" \
          "trainer.default_local_dir=${TRAIN}/ckpts/${ckpt_name}"; do
    printf '%s\n' "$@" | grep -q "^${kv%%=*}=" || cmd+=("${kv}")
done

cmd+=("${calib_overrides[@]}" "$@")

echo "[launch] config=${CONFIG} tag=${run_tag}"
echo "[launch] tb=${TENSORBOARD_DIR}"
echo "[launch] log=${log}"

if [[ -n "${FOREGROUND:-}" ]]; then
    "${cmd[@]}" 2>&1 | tee "${log}"
else
    nohup "${cmd[@]}" > "${log}" 2>&1 &
    echo "[launch] pid=$! (backgrounded; FOREGROUND=1 to run in the foreground)"
    echo "[launch] follow:  tail -f ${log}"
fi
