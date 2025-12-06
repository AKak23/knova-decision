# Repository Guidelines

## Project Structure & Module Organization
- Core features live in `feature_one.py`, `feature_two.py`, `feature_three.py`, `feature_four.py`; LLM wrapper in `llm_client.py`; shared指标目录在 `metrics_catalog.py`；手动联调脚本 `test_functions.py`；静态表格 `决策仿真页面指标表.xlsx`。
- 每个功能文件暴露单一主函数，遵循“输入上下文 + LLM prompt + JSON解析”模式，便于独立调用与测试。

## Build, Test, and Development Commands
- 运行功能联调：`python test_functions.py`（需提前设置 `QWEN_API_KEY` 或在 `llm_client.py` 填写 `API_KEY`）。该脚本顺序调用四个功能并打印结果。
- 推荐在虚拟环境内开发：`python -m venv .venv && source .venv/bin/activate`，再安装依赖 `pip install langchain-core langchain-openai`（或兼容版本）。



## Testing Guidelines
- 目前采用手动联调脚本 `test_functions.py`；如新增功能，请添加对应测试函数并在 `__main__` 中调用。
- 在调用前检查llm_client.py中 `QWEN_API_KEY` 是否可用，避免请求失败。
- 若引入自动化测试，文件命名建议 `test_<module>.py`，使用 `pytest` 或内置 `unittest`。




