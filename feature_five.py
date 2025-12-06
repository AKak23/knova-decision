"""功能函数五：根据历史指标与方案信息生成数值预测。"""

import json
from typing import Any, Dict, Optional, Sequence

from langchain_core.messages import HumanMessage, SystemMessage

from llm_client import LLMClient


def predict_metrics(
    customer: str,
    scene: str,
    metrics_data: Sequence[Dict[str, Any]],
    solutions: Sequence[Dict[str, Any]],
    llm_client: Optional[LLMClient] = None,
) -> Dict[str, Any]:
    """Predict future metric values per solution using provided history."""
    client = llm_client or LLMClient()
    metrics_json = json.dumps(metrics_data, ensure_ascii=False, indent=2)
    solutions_json = json.dumps(solutions, ensure_ascii=False, indent=2)

    system = SystemMessage(
        content=(
            "你是通信行业的数据科学家，负责结合方案特性与历史指标做短期预测。"
            "必须按照指定JSON结构返回结果，勿添加额外说明。"
        )
    )

    user = HumanMessage(
        content=(
            f"客群：{customer}\n"
            f"场景：{scene}\n\n"
            "输入指标数据（history_data 为已观测值，predicted_data 需生成未来值）：\n"
            f"{metrics_json}\n\n"
            "三个解决方案（与功能二输出一致）：\n"
            f"{solutions_json}\n\n"
            "任务：基于历史趋势、季节性及各方案特点，生成预测数值，并填入每个 metric 的 predicted_data.metric_value。"
            "需考虑不同方案对指标的正负影响与力度差异。\n"
            "要求：\n"
            "1) 保持 metric_name 与输入一致；\n"
            "2) predicted_data 的日期数量与输入中给出的占位保持一致，并按时间顺序；\n"
            "3) 数值可为小数，体现方案差异（A/B/C 不可完全相同）；\n"
            "4) 输出严格使用以下 JSON 结构：\n"
            '{\n'
            '  "customer": "xxx",\n'
            '  "scene": "yyy",\n'
            '  "solutions": [\n'
            '    {\n'
            '      "name": "方案A",\n'
            '      "metrics": [\n'
            '        {\n'
            '          "metric_name": "ARPU值",\n'
            '          "history_data": [{"date": "2025-11-01", "metric_value": 120.5}],\n'
            '          "predicted_data": [{"date": "2025-11-04", "metric_value": 128.4}]\n'
            '        }\n'
            '      ]\n'
            '    },\n'
            '    {"name": "方案B", "metrics": [ ... ]},\n'
            '    {"name": "方案C", "metrics": [ ... ]}\n'
            '  ]\n'
            '}'
        )
    )

    response_text = client.chat([system, user])
    try:
        return json.loads(response_text)
    except Exception:
        # 如果返回不是严格的JSON，保留原文本供调用方处理。
        return {"raw_response": response_text}
