"""功能函数四：针对客群与场景进行敏感性分析。"""

import json
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from llm_client import LLMClient


def generate_sensitivity_factors(
    customer: str,
    scene: str,
    llm_client: Optional[LLMClient] = None,
) -> Dict[str, Any]:
    """Return top drivers with positive/negative impacts for tornado chart."""
    client = llm_client or LLMClient()

    system = SystemMessage(
        content=(
            "你是通信行业的策略评估专家，擅长进行敏感性分析。"
            "请仅输出JSON，不要添加额外解释。"
        )
    )

    user = HumanMessage(
        content=(
            f"客群：{customer}\n"
            f"场景：{scene}\n\n"
            "请结合客群与场景，给出对结果影响力最大的5-10个因子，按权重从高到低排列。"
            "要求：\n"
            "1) 每个因子提供正向影响值与负向影响值（可视为提升/下降幅度或相对贡献，保留1-2位小数）；\n"
            "2) 数值可为百分比或分数，但正负方向需清晰；\n"
            "3) 输出严格使用JSON结构。\n\n"
            "JSON结构示例：\n"
            '{\n'
            '  "客群": "个人市场-全球通用户",\n'
            '  "场景": "提升当前客群价值",\n'
            '  "factors": [\n'
            '    {"name": "价格敏感度", "positive": 12.5, "negative": -8.3},\n'
            '    {"name": "促销活动", "positive": 15.2, "negative": -6.7},\n'
            '    {"name": "市场趋势", "positive": 13.4, "negative": -6.2}\n'
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
