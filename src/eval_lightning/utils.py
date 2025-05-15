from typing import List, Dict, Any, Callable
import json
import statistics


def mean_aggregate(scores: List[float]) -> float:
    """计算平均分数"""
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def median_aggregate(scores: List[float]) -> float:
    """计算中位数分数"""
    if not scores:
        return 0.0
    return statistics.median(scores)


def max_aggregate(scores: List[float]) -> float:
    """计算最大分数"""
    if not scores:
        return 0.0
    return max(scores)


def min_aggregate(scores: List[float]) -> float:
    """计算最小分数"""
    if not scores:
        return 0.0
    return min(scores)


def weighted_aggregate(scores: List[float], weights: List[float]) -> float:
    """计算加权平均分数"""
    if not scores or not weights or len(scores) != len(weights):
        return 0.0
    return sum(s * w for s, w in zip(scores, weights)) / sum(weights)


def save_results(results: Dict[str, Any], filepath: str) -> None:
    """保存评估结果到JSON文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def load_results(filepath: str) -> Dict[str, Any]:
    """从JSON文件加载评估结果"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f) 