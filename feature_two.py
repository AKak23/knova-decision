"""功能函数二：生成针对客群、场景与约束条件的推荐方案。"""

import json
from typing import Any, Dict, List, Optional, Sequence, Union

from langchain_core.messages import HumanMessage, SystemMessage

from llm_client import LLMClient


def _normalize_input(data: Union[List[Any], Dict[str, Any], str]) -> str:
    """Convert mixed input formats into a readable string for prompts."""
    if isinstance(data, dict):
        lines: List[str] = []
        for k, v in data.items():
            if isinstance(v, dict):
                inner = "; ".join(f"{ik}:{iv}" for ik, iv in v.items())
                lines.append(f"{k}：{inner}")
            elif isinstance(v, (list, tuple)):
                lines.append(f"{k}：{', '.join(str(item) for item in v)}")
            else:
                lines.append(f"{k}：{v}")
        return "\n".join(lines)
    if isinstance(data, (list, tuple)):
        return "\n".join(str(item) for item in data)
    return str(data)


def _strip_code_fences(text: str) -> str:
    """Remove common Markdown code fences to improve JSON parsing robustness."""
    if text.startswith("```"):
        # Strip leading and trailing fenced blocks
        text = text.strip()
        if text.startswith("```json"):
            text = text[len("```json") :].strip()
        elif text.startswith("```"):
            text = text[3:].strip()
        if text.endswith("```"):
            text = text[: -3].strip()
    return text


def generate_strategies(
    request_data: Union[List[Any], Dict[str, Any], str],
    llm_client: Optional[LLMClient] = None,
) -> Dict[str, Any]:
    """Generate three recommended plans plus an explanation."""
    client = llm_client or LLMClient()
    user_context = _normalize_input(request_data)

    system = SystemMessage(
        content=(
            "你是中国移动的营销与经营决策专家。"
            "请严格遵守输出结构，围绕收入、成本、风险、置信度以及可执行的步骤方案给出建议。"
            "禁止使用Markdown代码块或额外文本，仅返回纯JSON。"
        )
    )

    schema_hint = (
        "输出请使用JSON格式，确保三个方案字段一致：\n"
        '{\n'
        '  "plans": [\n'
        '    {\n'
        '      "方案名称": "方案A",\n'
        '      "收入": "+100万元",\n'
        '      "成本": "投入成本或资源消耗",\n'
        '      "风险": "潜在风险点",\n'
        '      "置信度": "高/中/低或百分比",\n'
        '      "执行清单方案": ["步骤1", "步骤2", "步骤3"]\n'
        '    },\n'
        '    { ... 共3个方案 ... }\n'
        '  ],\n'
        '  "explanation": "约100字的整体解释"\n'
        '}'
    )

    user = HumanMessage(
        content=(
            f"{user_context}\n\n"
            "请结合客群、场景、关键指标、预算与策略偏好，生成3个方案。"
            "务必保证三个方案的字段完全一致，执行清单需为分步骤列表。"
            "收入字段必须用金额表达，并带方向，例如 +120万元 或 -30万元，金额需与方案逻辑相符。"
            "策略偏好字段值为0-1数值（如风险偏好:0.5、权益包强度:0.7），数值越靠近1说明强度越大，越靠近0表明强度越小。请基于这些字段与数值来推导收入/成本/风险。"
            "禁止输出 Markdown 代码块或任何前后缀说明，仅返回可被 json.loads 直接解析的 JSON。"
            "输出字段仅限 方案名称、收入、成本、风险、置信度、执行清单方案、explanation，不要在方案对象里加入“策略偏好”等未列字段。"
            f"\n\n{schema_hint}"
        )
    )

    response_text = client.chat([system, user])
    clean_text = _strip_code_fences(response_text)
    try:
        parsed = json.loads(clean_text)
        plans = parsed.get("plans")
        if isinstance(plans, list):
            for plan in plans:
                if isinstance(plan, dict):
                    plan.pop("策略偏好", None)
        return parsed
    except Exception:
        # 如果返回不是严格的JSON，保留原文本供调用方处理。
        return {"raw_response": response_text, "sanitized": clean_text}
