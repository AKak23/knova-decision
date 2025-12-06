"""功能函数一：根据客群与场景，从指标列表中挑选影响关系最大的三个指标。"""

import json
from typing import List, Optional, Sequence

from langchain_core.messages import HumanMessage, SystemMessage

from llm_client import LLMClient


def select_top_metrics(
    customer_and_scene: Sequence[str],
    metrics: Sequence[str],
    llm_client: Optional[LLMClient] = None,
) -> List[str]:
    """Pick three key metrics via LLM prompt; metrics must be explicitly provided to support multi-tenant scenarios."""
    client = llm_client or LLMClient()

    system = SystemMessage(
        content=(
            "你是通信行业的业务运营专家，请仅在给定的候选指标列表中选择最能反映目标客群和业务场景的三个关键指标。"
            "输出需要严格使用JSON，避免使用额外的自然语言。"
        )
    )
    user = HumanMessage(
        content=(
            "客群与场景信息：\n"
            f"{chr(10).join(str(item) for item in customer_and_scene)}\n\n"
            "可选指标列表（只能从中选取，不得新增）：\n"
            f"{', '.join(metrics)}\n\n"
            "请按重要性排序返回JSON，结构如下：\n"
            '{\n'
            '  "top_indicators": ["指标1","指标2","指标3"],\n'
            '  "rationale": "用1-2句话说明选择依据"\n'
            '}'
        )
    )

    response_text = client.chat([system, user])
    try:
        parsed = json.loads(response_text)
        indicators = parsed.get("top_indicators") or parsed.get("indicators") or []
        return [str(item) for item in indicators][:3]
    except Exception:
        # 如果模型未按JSON返回，则退回原文本以便上层处理。
        return [response_text]
