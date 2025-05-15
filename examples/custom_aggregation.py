"""
自定义聚合函数示例

展示如何在 EvalLightning 中使用自定义聚合函数
"""

import random
from typing import Dict, List

from src.eval_lightning import EvalLightning
from src.eval_lightning.utils import save_results


# 定义一个简单的模型函数
def test_model(sample: Dict) -> float:
    """返回一个基于输入质量的分数"""
    quality = sample.get('quality', 0)
    # 添加一些随机性
    noise = random.uniform(-0.1, 0.1)
    return min(1.0, max(0.0, quality + noise))


# 创建一个简单的数据集
def create_dataset(size=20):
    dataset = []
    for _ in range(size):
        # 生成不同质量等级的样本
        quality = random.uniform(0.0, 1.0)
        dataset.append({'quality': quality, 'id': f"sample_{_}"})
    
    return dataset


# 自定义聚合函数1：丢弃最高和最低分，然后取平均值
def trimmed_mean_aggregate(scores: List[float]) -> float:
    """丢弃最高和最低分，然后计算平均值"""
    if len(scores) <= 2:
        return sum(scores) / len(scores) if scores else 0.0
    
    # 排序并丢弃最高和最低分
    sorted_scores = sorted(scores)
    trimmed_scores = sorted_scores[1:-1]
    
    return sum(trimmed_scores) / len(trimmed_scores)


# 自定义聚合函数2：根据分数的分布情况加权
def distribution_weighted_aggregate(scores: List[float]) -> float:
    """根据分数分布加权计算"""
    if not scores:
        return 0.0
    
    # 计算分数的平均值和标准差
    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    std_dev = variance ** 0.5
    
    # 如果标准差接近0，说明分数很一致，直接返回平均值
    if std_dev < 0.01:
        return mean
    
    # 计算每个分数的权重：越接近平均值的分数权重越高
    weights = [1 / (1 + abs(s - mean) / std_dev) for s in scores]
    
    # 计算加权平均值
    weighted_sum = sum(s * w for s, w in zip(scores, weights))
    total_weight = sum(weights)
    
    return weighted_sum / total_weight


def main():
    # 创建数据集
    dataset = create_dataset(50)
    print(f"Created dataset with {len(dataset)} samples")
    
    # 创建模型
    model = test_model
    
    # 使用不同的聚合函数创建评估器
    evaluator1 = EvalLightning(
        models=[model],
        dataset=dataset,
        aggregate_fn=trimmed_mean_aggregate,
        verbose=True
    )
    
    evaluator2 = EvalLightning(
        models=[model],
        dataset=dataset,
        aggregate_fn=distribution_weighted_aggregate,
        verbose=True
    )
    
    # 运行评估
    print("\nRunning evaluation with trimmed mean aggregation...")
    results1 = evaluator1.evaluate()
    
    print("\nRunning evaluation with distribution weighted aggregation...")
    results2 = evaluator2.evaluate()
    
    # 打印结果比较
    print("\nComparison of aggregation methods:")
    print(f"  Trimmed mean: {results1['test_model']['final_score']:.4f}")
    print(f"  Distribution weighted: {results2['test_model']['final_score']:.4f}")
    
    # 保存结果
    save_results({
        'trimmed_mean': results1,
        'distribution_weighted': results2
    }, "custom_aggregation_results.json")
    print("\nResults saved to custom_aggregation_results.json")


if __name__ == "__main__":
    main() 