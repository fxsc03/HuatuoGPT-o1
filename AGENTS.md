# Repository Guidelines

## Project Structure & Module Organization
该仓库以根目录脚本为主，而不是 `src/` 包结构。训练入口在 `SFT_stage1.py` 和 `RL_stage2.py`；数据构造与清洗脚本包括 `construct_verifiable_medical_problems.py`、`search_for_complex_reasoning_path*.py`、`merge_datasets.py` 等。`ppo_utils/` 存放 PPO 配置与训练器，`configs/` 存放 Accelerate/DeepSpeed 配置，`evaluation/` 包含评测脚本、打分器和 `evaluation/data/eval_data.json`。模型、数据和产物通常位于 `models/`、`data/`、`output_data/`，说明图片放在 `assets/`。

## Build, Test, and Development Commands
优先使用 `uv` 安装依赖：`uv sync`。如果不使用 `uv`，可退回 `pip install -r requirements.txt`。常见开发命令：

- `accelerate launch --config_file ./configs/deepspeed_zero3.yaml ... SFT_stage1.py`：启动第一阶段 SFT。
- `accelerate launch --config_file ./configs/deepspeed_zero3.yaml ... RL_stage2.py`：启动第二阶段 PPO/RL。
- `python evaluation/eval.py --model_name <model> --eval_file evaluation/data/eval_data.json --port <port>`：对已部署模型做评测。
- `bash evaluation/kill_sglang_server.sh`：停止本地 SGLang 服务并释放显存。

## Coding Style & Naming Conventions
目标环境是 Python 3.10+。沿用现有脚本式风格：4 空格缩进，模块/函数/CLI 参数使用 `snake_case`，类名使用 `PascalCase`。新增入口脚本应继续使用 `argparse`，路径默认从仓库根目录解析。仓库未配置 `black`、`ruff` 或 `pytest`，因此提交前至少保证导入、参数名和 JSON 路径保持一致，避免顺手大规模重排现有代码。

## Testing Guidelines
当前没有独立自动化测试目录，评测脚本和小样本跑通就是最低验证标准。修改训练或数据处理逻辑时，先用小 JSON 文件做 smoke test；修改评测逻辑时，至少运行一次 `evaluation/eval.py`。如果新增自动化测试，统一放到 `tests/`，命名为 `test_*.py`，并保持夹具数据轻量。

## Commit & Pull Request Guidelines
现有历史里大量提交信息较短，如 `update`，也有少量 `chore: ...`。新提交请改用更具体的祈使句风格，例如 `fix: handle empty eval output`。PR 应包含变更目的、影响的脚本或数据路径、复现命令，以及必要的日志/指标变化；涉及 API key、模型权重路径或大体积生成文件时，在描述中明确说明，不要把密钥、缓存和临时日志提交进仓库。
