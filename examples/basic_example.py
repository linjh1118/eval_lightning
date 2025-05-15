"""
基本使用示例

展示 EvalLightning 的基本用法
"""

import random
from typing import Dict

from src.eval_lightning import EvalLightning
from src.eval_lightning.utils import median_aggregate, save_results


# 定义一些简单的模型函数
def model_a(sample: Dict) -> float:
    """模型A：返回一个基于输入文本长度的分数"""
    text = sample.get('text', '')
    # 简单评分逻辑：文本长度在10-30之间得分最高
    length = len(text)
    if 10 <= length <= 30:
        return 0.8 + random.random() * 0.2
    elif length < 10:
        return 0.5 + random.random() * 0.3
    else:
        return 0.6 + random.random() * 0.3


def model_b(sample: Dict) -> float:
    """模型B：返回一个基于特定关键词的分数"""
    text = sample.get('text', '').lower()
    keywords = ['good', 'great', 'excellent', 'amazing']
    
    score = 0.5  # 基础分
    for keyword in keywords:
        if keyword in text:
            score += 0.1  # 每包含一个关键词加0.1分
    
    return min(1.0, score + random.random() * 0.2)


def model_c(sample: Dict) -> float:
    """模型C：随机返回一个分数，模拟不稳定的模型"""
    return random.random()


# 创建一个简单的数据集
def create_dataset(size=20):
    texts = [
        "This is a good example.",
        "The weather is great today.",
        "I had an excellent experience with the service.",
        "The product quality is amazing.",
        "This is just an average product.",
        "I didn't like the customer service.",
        "The food was delicious and the service was excellent.",
        "The movie was boring and too long.",
        "The hotel room was clean and comfortable.",
        "The application process was complicated."
    ]
    
    # 生成数据集
    dataset = []
    for _ in range(size):
        # 随机选择一个文本，或者生成一个随机长度的文本
        if random.random() < 0.7 and texts:
            text = random.choice(texts)
        else:
            length = random.randint(5, 50)
            text = ' '.join(['word' for _ in range(length)])
        
        dataset.append({'text': text})
    
    return dataset


def main():
    # 创建数据集
    dataset = create_dataset(30)
    print(f"Created dataset with {len(dataset)} samples")
    
    # 创建模型列表
    models = [model_a, model_b, model_c]
    
    # 初始化评估器
    evaluator = EvalLightning(
        models=models,
        dataset=dataset,
        aggregate_fn=median_aggregate,  # 使用中位数聚合
        verbose=True
    )
    
    # 运行评估
    print("\nRunning sequential evaluation...")
    results = evaluator.evaluate()
    
    # 打印结果
    print("\nEvaluation results:")
    for model_name, result in results.items():
        print(f"  {model_name}: {result['final_score']:.4f} (took {result['time_taken']:.2f}s)")
    
    # 保存结果
    save_results(results, "evaluation_results.json")
    print("\nResults saved to evaluation_results.json")
    
    # 运行并行评估
    print("\nRunning parallel evaluation...")
    parallel_results = evaluator.evaluate_parallel(max_workers=4)
    
    # 打印并行结果
    print("\nParallel evaluation results:")
    for model_name, result in parallel_results.items():
        print(f"  {model_name}: {result['final_score']:.4f} (took {result['time_taken']:.2f}s)")


if __name__ == "__main__":
    main() 