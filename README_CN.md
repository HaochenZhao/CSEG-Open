# CSEG: 跨源证据锚定

[English](README.md) | **中文**

![arXiv](https://img.shields.io/badge/arXiv-2610.17827-b31b1b.svg)
![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face-CSEG--RM-yellow)

本仓库是 **CSEG-RM** 与 **CSEG-Bench** 的官方训练与评测代码。CSEG-RM 是一个六层判定器，用于检查搜索智能体的回答是否锚定于、并正确调和多份检索来源中的证据；CSEG-Bench 是用于评测它的 1,160 条轨迹基准。

**论文：** [Benchmarking and Training Reward Models for Cross-Source Evidence Grounding in Search Agents](https://arxiv.org/abs/2610.17827)（[PDF](https://arxiv.org/pdf/2610.17827)）

Haochen Zhao1\*、Minsheng Hao2、Zheng Li1、Yifan Li3、Zikang Liu3、Bowen Qin2、Yuhao Cheng2、Chuang Fan2、Haosheng Zou2、Jeff Chen2†、Sujian Li1‡

1 北京大学 &emsp; 2 腾讯 &emsp; 3 中国人民大学

\* 腾讯实习期间完成。 &emsp; † 项目负责人。 &emsp; ‡ 通讯作者。

**链接：** [arXiv](https://arxiv.org/abs/2610.17827) · [PDF](https://arxiv.org/pdf/2610.17827) · [Hugging Face 合集](https://huggingface.co/collections/VectorZhao/cseg-rm)

## 动态

- **2026-10-13：** 论文已发布于 [arXiv](https://arxiv.org/abs/2610.17827)（`2610.17827`）。

## 摘要

现有搜索智能体的训练与评测高度关注最终答案质量，却很少检查生成答案如何回指检索证据。仅做引用支持判定，并不能说明来源是否独立，也无法判断回答是否正确处理了冲突证据。我们提出 **CSEG**：一棵覆盖完整搜索轨迹的六层决策树。**CSEG-Bench** 包含覆盖九条规范路径的 1,160 条已验证轨迹。**CSEG-RM** 在该基准上达到 83.9% 精确准确率，并位于性能–token 效率帕累托前沿。仅用该奖励监督冻结检索的回答策略，即可将独立判定的平均标签从 2.83 提升到 4.19。

种子轨迹来自腾讯元宝搜索助手的 RLVR rollout 日志（见[数据来源](#数据来源)）。训练与评测 CSEG-RM **不需要**源服务的专有 chat template 与工具协议。CSEG-Bench 的 `messages` 已去掉原始 `role=system` 轮次（助手身份与工具协议）；打分器本身也会忽略该角色。user / tool / assistant 轮次保持不变。

## 已发布资源


| 资源                                       | 论文对应行                 | Hugging Face                                                                              |
| ---------------------------------------- | --------------------- | ----------------------------------------------------------------------------------------- |
| CSEG-Bench（n = 1,160）                    | Table 1 / Table 2 评测集 | `[VectorZhao/CSEG-Bench](https://huggingface.co/datasets/VectorZhao/CSEG-Bench)`          |
| Qwen3-4B + 32B-RS SFT（`global_step_220`） | Table 1 SFT           | `[VectorZhao/CSEG-RM-SFT-4B](https://huggingface.co/VectorZhao/CSEG-RM-SFT-4B)`           |
| Qwen3-8B + 32B-RS SFT（`global_step_220`） | Table 1 SFT           | `[VectorZhao/CSEG-RM-SFT-8B](https://huggingface.co/VectorZhao/CSEG-RM-SFT-8B)`           |
| Qwen3-4B SFT + GRPO-400                  | Table 1 SFT+GRPO（主结果） | `[VectorZhao/CSEG-RM-SFT-GRPO-4B](https://huggingface.co/VectorZhao/CSEG-RM-SFT-GRPO-4B)` |
| Qwen3-8B SFT + GRPO-400                  | Table 1 SFT+GRPO      | `[VectorZhao/CSEG-RM-SFT-GRPO-8B](https://huggingface.co/VectorZhao/CSEG-RM-SFT-GRPO-8B)` |
| Qwen3-4B 从基座 GRPO-400                    | Table 2 合成数据          | `[VectorZhao/CSEG-RM-GRPO-4B](https://huggingface.co/VectorZhao/CSEG-RM-GRPO-4B)`         |
| Qwen3-8B 从基座 GRPO-400                    | Table 2 合成数据          | `[VectorZhao/CSEG-RM-GRPO-8B](https://huggingface.co/VectorZhao/CSEG-RM-GRPO-8B)`         |


全部 checkpoint 为 bf16 Hugging Face safetensors，开启 thinking，并输出完整 JSON 判定（`--prompt-output-spec full`）。

`scripts/download_hf.sh` 会把各资源放到配置与启动脚本期望的路径：

```bash
bash scripts/download_hf.sh bench            # -> benchmark/eval.jsonl
bash scripts/download_hf.sh sft-4b           # -> training/sft/ckpts/sft_qwen3_4b_32b_rs/global_step_220/huggingface
bash scripts/download_hf.sh sft-grpo-4b      # -> training/ckpts/sft_grpo_qwen3_4b/global_step_400/actor/huggingface
bash scripts/download_hf.sh all              # 全部（约 75 GB）
```

## 流水线

```
Qwen3-4B / Qwen3-8B
    → 32B-RS SFT          （Table 1 SFT 行）
    → 合并 FSDP 分片      （SFT global_step_220 → huggingface/）
    → GRPO-400            （Table 1 SFT+GRPO 行）
    → 合并 FSDP 分片
    → vLLM serve
    → CSEG-Bench          （n = 1,160）
```

Table 2 是在同一合成划分上从基座做 GRPO，并与等规模的自然 rollout 消融划分对照。

32B 拒绝采样教师运行不在本仓库中。CSEG-RM 的 SFT 数据是已完成的 parquet（thinking-on，完整 JSON）。训练从公开的 `Qwen/Qwen3-4B` 或 `Qwen/Qwen3-8B` 权重开始。

## 目录结构

```
data_filter/     节点 prompt、路由、轨迹打分器
benchmark/       CSEG-Bench gold schema + 指标 + runner
training/        SFT / GRPO 配方、层奖励、启动脚本
scripts/         Hugging Face 下载脚本
data/            论文名称 → 路径索引（文件放在各消费者旁）
ckpts/           checkpoint 路径索引
```

`training/rewards/layer_reward.py` 从 `data_filter/score_trajectory.py` 加载 JSON 解析器，`benchmark/predict.py` 走同一套六层流水线，因此训练奖励与评测对输出的解析方式一致。

## 重要模块


| 组件             | 文件                                 | 作用                                                                   |
| -------------- | ---------------------------------- | -------------------------------------------------------------------- |
| 节点 prompt + 路由 | `data_filter/prompt_config.json`   | 各层模板以及 `routing` / `label_mapping`。论文配方使用英文文件。                       |
| Schema         | `benchmark/gold.py`                | `GOLD_PATHS`、`NODE_VERDICTS`、`LAYER_ORDER`。判定 JSON 把 `verdict` 放在最前。 |
| 轨迹打分器          | `data_filter/score_trajectory.py`  | `parse_trajectory`、`score_record`、`_parse_json_from_text`。           |
| Prompt 变体      | `data_filter/prompt_variants.py`   | `full`（论文）与 `verdict_only`。                                          |
| 层奖励            | `training/rewards/layer_reward.py` | GRPO 的 `compute_score`；解析器与评测相同。                                     |


## 数据文件


| 论文名称         | 本仓库内容                              | 完整规模 n       | 路径                                          |
| ------------ | ---------------------------------- | ------------ | ------------------------------------------- |
| CSEG-Bench   | 下载（`scripts/download_hf.sh bench`） | 1,160        | `benchmark/eval.jsonl`                      |
| CSEG-RM SFT  | 子集，75 / 14                         | 14,081 / 795 | `training/sft/data/sft_{train,dev}.parquet` |
| CSEG-RM GRPO | 子集，75 / 15                         | 7,320 / 911  | `training/data/rl_{train,val}.jsonl`        |


`eval.jsonl`（约 118 MB）超过 GitHub 文件大小限制，因此从 Hub 拉取而非提交进仓库。发布文件已去掉源助手的 `role=system` 轮次，并遵循[数据来源](#数据来源)中的去标识约束。

CSEG-RM 训练子集是分层抽样（seed 42）：每个 `(layer, verdict)` 格子 5 条（15 格 × 5 = 75）。Dev/val 每个可用格子 1 条（SFT dev 只有 14 格，因为完整 SFT dev 划分没有 `L5_conflict`/`fail`）。它们足以检查 schema 并端到端跑通启动脚本；**不要**用它们报告 Table 1 / Table 2 数字。要复现表格，请把完整 CSEG-RM 划分放到**相同路径**。

可选 Table 2 消融：`training/data/rl_{train,val}_baseline_qwen3_8b_rollout.jsonl`。

详见 `data/README.md`、`training/sft/data/README.md` 和 `training/data/README.md`。

## 配置 ↔ 论文表格


| 论文行                                 | 配置                                            | 初始化                   |
| ----------------------------------- | --------------------------------------------- | --------------------- |
| Table 1 · Qwen3-4B + 32B-RS SFT     | `training/sft/configs/sft_qwen3_4b.yaml`      | `Qwen/Qwen3-4B`       |
| Table 1 · Qwen3-8B + 32B-RS SFT     | `training/sft/configs/sft_qwen3_8b.yaml`      | `Qwen/Qwen3-8B`       |
| Table 1 · Qwen3-4B + SFT + GRPO-400 | `training/configs/sft_grpo_qwen3_4b.yaml`     | SFT `global_step_220` |
| Table 1 · Qwen3-8B + SFT + GRPO-400 | `training/configs/sft_grpo_qwen3_8b.yaml`     | SFT `global_step_220` |
| Table 2 · 合成数据，从基座 4B               | `training/configs/grpo_qwen3_4b.yaml`         | `Qwen/Qwen3-4B`       |
| Table 2 · 合成数据，从基座 8B               | `training/configs/grpo_qwen3_8b.yaml`         | `Qwen/Qwen3-8B`       |
| Table 2 · 自然 rollout，4B             | `training/configs/grpo_qwen3_4b_rollout.yaml` | `Qwen/Qwen3-4B`       |
| Table 2 · 自然 rollout，8B             | `training/configs/grpo_qwen3_8b_rollout.yaml` | `Qwen/Qwen3-8B`       |


Checkpoint 目录（与 `scripts/download_hf.sh` 相同）：

- `training/sft/ckpts/sft_qwen3_{4b,8b}_32b_rs/global_step_220/huggingface`
- `training/ckpts/<config-name>/global_step_400/actor/huggingface`

## 环境

固定依赖：`verl==0.8.0`、`vllm==0.11.0`（见 `requirements.txt`）。

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # 按需修改 HF_HOME / OPENAI_API_BASE
set -a && source .env && set +a
```

可选：`export VENV=/path/to/venv`，让启动脚本自动激活该环境。

论文硬件：

- SFT：1 节点 × 8 GPU
- GRPO：4 节点 × 8 GPU（YAML 中 `trainer.nnodes: 4`）。集群不同时可覆盖 `trainer.nnodes=<n>`。

部分多机 InfiniBand 网络必须设置 `NCCL_CUMEM_ENABLE=0`（启动脚本已导出）。请使用 `--config-dir`，不要用 `--config-path`：相对的 `--config-path` 会相对于 verl 包目录解析。

`scripts/launch_grpo.sh` 会从 YAML 重新锚定 data / init / reward / checkpoint 路径，**不会**写死数据集文件名。若把 `data.train_files` 写成 Hydra override，会在不知不觉中让所有消融都训到合成划分上。

## 训练

SFT（thinking-on，完整 JSON，3 个 epoch；论文报告的是 epoch 1 / `global_step_220`）：

```bash
cd training/sft
FOREGROUND=1 bash scripts/launch_sft.sh sft_qwen3_4b
# FOREGROUND=1 bash scripts/launch_sft.sh sft_qwen3_8b
```

verl 会保存 FSDP 分片，以及一个只含 config 与 tokenizer 的 `huggingface/` 目录。把 SFT checkpoint 用作 GRPO 初始化前，先合并分片：

```bash
cd training
CKPT_ROOT=./sft/ckpts/sft_qwen3_4b_32b_rs bash scripts/merge_fsdp_ckpts.sh global_step_220
```

也可跳过 SFT，直接下载已发布的 SFT 导出：`bash scripts/download_hf.sh sft-4b`。

主表 GRPO（读取 YAML 中路径上已合并的 SFT 导出）：

```bash
# Ray 集群需已启动（RAY_ADDR，默认 127.0.0.1:6379）
cd training
FOREGROUND=1 bash scripts/launch_grpo.sh sft_grpo_qwen3_4b
```

Table 2 从基座 / 消融：把对应配置名传给 `launch_grpo.sh`。额外参数会作为 Hydra override 传给任一启动脚本。

跑完后合并 GRPO 的 FSDP 分片：

```bash
cd training
CKPT_ROOT=./ckpts/sft_grpo_qwen3_4b bash scripts/merge_fsdp_ckpts.sh global_step_400
```

## 评测（Table 1 / Table 2）

先 serving 合并后的权重，再在 CSEG-Bench 上跑六层判定。
已发布 checkpoint 均为 thinking-on，并使用 `--prompt-output-spec full`。

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

头条指标排除类型 `L6_fail_consensus`（与 CSEG-RM 训练相同）。Gold 为 `target_label` / `GOLD_PATHS[type]`。

## 数据来源

按论文 [Ethics Statement](https://arxiv.org/abs/2610.17827)：

种子轨迹来自腾讯元宝搜索助手。纳入的轨迹对应用户已通过专门的用户体验计划同意将数据用于研究。公开发布已去除直接标识符与敏感内容，尽量减少查询文本暴露，并按检索页面适用的许可与条款使用这些页面。CSEG-Bench 的 `messages` 已去掉 `role=system`（助手身份与工具协议）；训练与评测 CSEG-RM 不需要源服务的专有 chat template。基准构建时注入的文档仅存在于受控评测环境中，不会作为真实网页对外展示。

## 引用

如果使用 CSEG、CSEG-Bench 或 CSEG-RM，请引用：

```bibtex
@misc{zhao2026benchmarkingtrainingrewardmodels,
  title={Benchmarking and Training Reward Models for Cross-Source Evidence Grounding in Search Agents},
  author={Haochen Zhao and Minsheng Hao and Zheng Li and Yifan Li and Zikang Liu and Bowen Qin and Yuhao Cheng and Chuang Fan and Haosheng Zou and Jeff Chen and Sujian Li},
  year={2026},
  eprint={2610.17827},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2610.17827},
}
```

## 许可

本仓库代码为 [Apache-2.0](LICENSE)。Hub 上发布的 CSEG-RM checkpoint 使用相同许可。CSEG-Bench 是单独的数据集发布（`VectorZhao/CSEG-Bench`），条款见该数据卡。