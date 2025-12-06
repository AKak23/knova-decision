"""功能函数六：根据客群与描述生成影响范围（地区/渠道等）。"""

import json
from typing import Dict, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from llm_client import LLMClient


def infer_impact_scope(
    customer: str,
    description: str,
    llm_client: Optional[LLMClient] = None,
) -> Dict[str, str]:
    """Infer a concise impact scope (<=10 chars) for the given customer profile."""
    client = llm_client or LLMClient()

    system = SystemMessage(
        content=(
            "你是通信行业的市场洞察专家，擅长提炼客群触达的核心影响范围。"
            "必须仅返回JSON，不要额外说明。"
            "影响范围需为地区、渠道或人群特征等短词语，长度不超过10个字。"
            "优先用地区或渠道表述，例如“华东”“一线城市”“电渠”“线上自营”“重点营业厅”等。"
        )
    )

    user = HumanMessage(
        content=(
            f"客群：{customer}\n"
            f"客群描述：{description}\n\n"
            "请基于客群特征与描述，提炼1个最关键的影响范围，要求：\n"
            "1) 影响范围用词语表达，例如“华东”“华南电渠”“一线城市”“高端商务渠道”，不得超过10个字；\n"
            "2) 需给出简短理由，说明为何该范围最重要；\n"
            "3) 输出严格遵循以下JSON结构：\n"
            '{\n'
            '  "impact_scope": "影响范围词语",\n'
            '  "reason": "15-30字的简短理由"\n'
            '}\n'
            "严禁输出Markdown代码块或任何前后缀。"
        )
    )

    response_text = client.chat([system, user])
    try:
        parsed = json.loads(response_text)
        scope = str(parsed.get("impact_scope", "")).strip()
        reason = str(parsed.get("reason", "")).strip()
        return {"impact_scope": scope, "reason": reason}
    except Exception:
        # 如果返回不是严格的JSON，保留原文本供调用方处理。
        return {"raw_response": response_text}
