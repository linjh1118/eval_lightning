"""
大语言模型评估示例

展示如何使用 EvalLightning 评估大语言模型
"""

import sys
import os
import random
import time
from typing import Dict, List, Any

from src.eval_lightning import EvalLightning
from src.eval_lightning.utils import save_results, weighted_aggregate


# 模拟不同的LLM模型
class MockLLM:
    def __init__(self, name: str, quality: float, latency: float):
        """
        初始化模拟LLM
        
        Args:
            name: 模型名称
            quality: 模型质量 (0-1)
            latency: 模拟延迟 (秒)
        """
        self.name = name
        self.quality = quality
        self.latency = latency
    
    def __call__(self, sample: Dict) -> float:
        """评估样本并返回分数"""
        # 模拟API调用延迟
        time.sleep(self.latency)
        
        # 获取样本难度
        difficulty = sample.get('difficulty', 0.5)
        
        # 计算基础分数：质量高的模型受难度影响较小
        base_score = self.quality * (1 - difficulty * (1 - self.quality))
        
        # 添加一些随机性
        noise = random.uniform(-0.1, 0.1)
        
        return min(1.0, max(0.0, base_score + noise))
    
    def __str__(self) -> str:
        return self.name


# 创建LLM评估数据集
def create_llm_dataset(size=20):
    # 定义一些问题类型
    question_types = [
        "factual_recall",
        "reasoning",
        "creative_writing",
        "code_generation",
        "math_problem"
    ]
    
    # 为每种问题类型定义难度
    difficulty_by_type = {
        "factual_recall": 0.3,
        "reasoning": 0.6,
        "creative_writing": 0.5,
        "code_generation": 0.7,
        "math_problem": 0.8
    }
    
    # 生成数据集
    dataset = []
    for i in range(size):
        # 随机选择问题类型
        q_type = random.choice(question_types)
        
        # 获取基础难度，并添加一些随机变化
        base_difficulty = difficulty_by_type[q_type]
        difficulty = min(1.0, max(0.1, base_difficulty + random.uniform(-0.2, 0.2)))
        
        # 创建样本
        sample = {
            'id': f"question_{i}",
            'type': q_type,
            'difficulty': difficulty,
            'prompt': f"This is a {q_type} question with difficulty {difficulty:.2f}"
        }
        
        dataset.append(sample)
    
    return dataset


# 自定义聚合函数：根据问题类型加权
def type_weighted_aggregate(scores: List[float], samples: List[Dict]) -> float:
    """根据问题类型加权聚合分数"""
    if not scores or not samples or len(scores) != len(samples):
        return 0.0
    
    # 定义不同问题类型的权重
    type_weights = {
        "factual_recall": 1.0,
        "reasoning": 1.5,
        "creative_writing": 0.8,
        "code_generation": 1.2,
        "math_problem": 1.3
    }
    
    # 计算每个样本的权重
    weights = [type_weights.get(sample.get('type', ''), 1.0) for sample in samples]
    
    # 计算加权平均值
    weighted_sum = sum(s * w for s, w in zip(scores, weights))
    total_weight = sum(weights)
    
    return weighted_sum / total_weight


def main():
    # 创建数据集
    dataset = create_llm_dataset(30)
    print(f"Created dataset with {len(dataset)} samples")
    
    # 创建模拟LLM模型
    models = [
        MockLLM("GPT-4-like", quality=0.9, latency=0.5),
        MockLLM("Claude-like", quality=0.85, latency=0.4),
        MockLLM("Llama-like", quality=0.8, latency=0.3),
        MockLLM("Mistral-like", quality=0.75, latency=0.2)
    ]
    
    # 创建自定义聚合函数 (闭包，以便访问数据集)
    def custom_aggregate(scores):
        return type_weighted_aggregate(scores, dataset)
    
    # 初始化评估器
    evaluator = EvalLightning(
        models=models,
        dataset=dataset,
        aggregate_fn=custom_aggregate,
        verbose=True
    )
    
    # 运行并行评估 (使用并行以提高速度)
    print("\nRunning LLM evaluation...")
    results = evaluator.evaluate_parallel(max_workers=4)
    
    # 打印结果
    print("\nLLM Evaluation results:")
    for model_name, result in results.items():
        print(f"  {model_name}: {result['final_score']:.4f} (took {result['time_taken']:.2f}s)")
    
    # 计算不同问题类型的表现
    print("\nPerformance by question type:")
    for q_type in ["factual_recall", "reasoning", "creative_writing", "code_generation", "math_problem"]:
        print(f"\n  {q_type.upper()}:")
        
        # 获取该类型的样本索引
        type_indices = [i for i, sample in enumerate(dataset) if sample['type'] == q_type]
        
        for model_name, result in results.items():
            # 计算该类型问题的平均分
            type_scores = [result['scores'][i] for i in type_indices]
            avg_score = sum(type_scores) / len(type_scores) if type_scores else 0
            
            print(f"    {model_name}: {avg_score:.4f}")
    
    # 保存结果
    save_results(results, "llm_evaluation_results.json")
    print("\nResults saved to llm_evaluation_results.json")


if __name__ == "__main__":
    main() 