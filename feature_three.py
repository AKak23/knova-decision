"""功能函数三：根据客群与场景生成数值化的方案输入。"""

import json
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.messages import HumanMessage, SystemMessage

from llm_client import LLMClient


def generate_numeric_inputs(
    customer: str,
    scene: str,
    indicators: Sequence[str],
    llm_client: Optional[LLMClient] = None,
) -> Dict[str, Any]:
    """Generate structured numeric inputs that align with feature_two payload."""
    client = llm_client or LLMClient()
    metrics = [str(item) for item in indicators][:3]

    system = SystemMessage(
        content=(
            "你是通信行业的经营分析专家，需要为决策仿真生成量化输入。"
            "请保持输出字段与示例一致，避免自由发挥或添加额外说明。"
        )
    )

    user = HumanMessage(
        content=(
            f"客群：{customer}\n"
            f"场景：{scene}\n"
            f"关键指标：{', '.join(metrics)}\n\n"
            "请根据上述信息为每个字段赋值，要求：\n"
            "1) 关键指标生成可衡量的目标，格式为“指标名:提升xx%”或“指标名:下降xx%”，xx为0-100的整数；\n"
            "2) 预算上限填写具体金额（示例：\"80万\"或\"200万元\"），不要写范围；\n"
            "3) 响应时间仅写成几天（单位统一为“天”），不要使用周/月等其他单位；\n"
            "4) 策略偏好中的风险偏好、权益包强度、投放预算、触达时机均为0-1的小数；\n"
            "5) 输出仅用JSON，字段名保持一致，不要追加解释性文字。\n\n"
            "JSON结构示例：\n"
            '{\n'
            '  "客群": "个人市场-全球通用户",\n'
            '  "场景": "提升当前客群价值",\n'
            '  "关键指标": ["移动客户ARPU:提升12%", "全球通客户数:提升5%", "公众市场用户5G网络渗透率:提升8%"],\n'
            '  "预算上限": "120万元",\n'
            '  "响应时间": "2周",\n'
            '  "策略偏好": {\n'
            '    "风险偏好": 0.4,\n'
            '    "权益包强度": 0.7,\n'
            '    "投放预算": 0.6,\n'
            '    "触达时机": 0.5\n'
            '  }\n'
            '}'
        )
    )

    response_text = client.chat([system, user])
    try:
        return json.loads(response_text)
    except Exception:
        # 如果返回不是严格的JSON，保留原文本供调用方处理。
        return {"raw_response": response_text}
