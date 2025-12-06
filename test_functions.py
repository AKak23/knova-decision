"""Manual test script for feature_one and feature_two.

Prerequisites:
- 已在当前环境配置好 LangChain 依赖与 QWEN_API_KEY。
- 如需调整模型或 base_url，请修改 llm_client.LLMClient 的默认参数。
"""

from pprint import pprint

from feature_one import select_top_metrics
from feature_two import generate_strategies
from feature_three import generate_numeric_inputs
from feature_four import generate_sensitivity_factors
from feature_five import predict_metrics
from metrics_catalog import DECISION_METRICS


def test_feature_one() -> None:
    sample_context = [
        "客群：个人市场-全球通用户",
        "场景：如何进一步提升当前客群用户价值",
    ]
    top_metrics = select_top_metrics(sample_context, metrics=DECISION_METRICS)
    print("\n=== 功能一：选择关键指标 ===")
    pprint(top_metrics)


def test_feature_two() -> None:
    request_payload = {
        "客群": "个人市场-全球通用户",
        "场景": "如何进一步提升当前客群用户价值",
        "关键指标": [
            "移动客户ARPU:提升12%",
            "全球通客户数:提升5%",
            "公众市场用户5G网络渗透率:提升8%",
        ],
        "预算上限": "120万元",
        "响应时间": "14天",
        "策略偏好": {
            "风险偏好": 0.4,
            "权益包强度": 0.7,
            "投放预算": 0.6,
            "触达时机": 0.5,
        },
    }
    response = generate_strategies(request_payload)
    print("\n=== 功能二：生成方案 ===")
    pprint(response)


def test_feature_three() -> None:
    numeric_payload = generate_numeric_inputs(
        customer="个人市场-全球通用户",
        scene="如何进一步提升当前客群用户价值",
        indicators=[
            "移动客户ARPU",
            "全球通客户数",
            "公众市场用户5G网络渗透率",
        ],
    )
    print("\n=== 功能三：数值生成 ===")
    pprint(numeric_payload)


def test_feature_four() -> None:
    sensitivity = generate_sensitivity_factors(
        customer="个人市场-全球通用户",
        scene="如何进一步提升当前客群用户价值",
    )
    print("\n=== 功能四：敏感性分析 ===")
    pprint(sensitivity)


def test_feature_five() -> None:
    metrics_payload = [
        {
            "metric_name": "ARPU值",
            "history_data": [
                {"date": "2025-11-01", "metric_value": 120.5},
                {"date": "2025-11-02", "metric_value": 130.2},
                {"date": "2025-11-03", "metric_value": 125.8},
            ],
            "predicted_data": [
                {"date": "2025-11-04", "metric_value": 0},
                {"date": "2025-11-05", "metric_value": 0},
                {"date": "2025-11-06", "metric_value": 0},
            ],
        }
    ]
    solutions = [
        {"name": "方案A", "收入": "高提升", "成本": "中等", "风险": "中", "置信度": "高", "执行清单方案": ["动作1", "动作2"]},
        {"name": "方案B", "收入": "中提升", "成本": "低", "风险": "低", "置信度": "中", "执行清单方案": ["动作1", "动作2"]},
        {"name": "方案C", "收入": "稳态", "成本": "低", "风险": "低", "置信度": "中", "执行清单方案": ["动作1", "动作2"]},
    ]
    predictions = predict_metrics(
        customer="个人市场-全球通用户",
        scene="如何进一步提升当前客群用户价值",
        metrics_data=metrics_payload,
        solutions=solutions,
    )
    print("\n=== 功能五：数值预测 ===")
    pprint(predictions)


if __name__ == "__main__":
    test_feature_one()
    test_feature_two()
    test_feature_three()
    test_feature_four()
    test_feature_five()
