"""
Wandb 集成示例

展示如何使用 EvalLightning 与 Wandb 进行模型评估跟踪，评估多个模型在多个数据集上的性能
"""

import random
from typing import Dict, Any, List
import datetime
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 现在可以导入模块
from src.eval_lightning import EvalLightning
from src.eval_lightning.utils import mean_aggregate, save_results
import wandb


# 定义一个简单的模型类，带有step属性
class StepModel:
    def __init__(self, step: int, quality: float = 0.5):
        self.step = step
        self.quality = quality  # 模型质量参数
        self.__name__ = f"model_step_{step}"
    
    def __call__(self, sample: Dict[str, Any]) -> float:
        """评估样本并返回分数"""
        # 获取数据集名称和文本
        dataset_name = sample.get('dataset', '')
        text = sample.get('text', '')
        
        # 基于模型质量和文本特征计算分数
        base_score = self.quality
        
        # 不同数据集有不同的基础分数调整
        dataset_adjustments = {
            'MathVista': 0.05,
            'MathVision': -0.03,
            'MathVerse': 0.02,
            'DynaMath': -0.01,
            'WeMath': 0.04,
            'LogicVista': -0.02
        }
        
        base_score += dataset_adjustments.get(dataset_name, 0)
        
        # 简单的文本评分逻辑
        if len(text) > 20:
            base_score += 0.1
        
        if any(keyword in text.lower() for keyword in ['good', 'great', 'excellent']):
            base_score += 0.2
        
        # 添加一些随机性
        noise = random.random() * 0.2 - 0.1
        
        return min(1.0, max(0.0, base_score + noise))


# 创建一个数据集
def create_dataset(dataset_name: str, size=30):
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
        
        dataset.append({
            'text': text,
            'dataset': dataset_name
        })
    
    return dataset


def evaluate_on_datasets(models, dataset_names):
    """分别评估每个数据集并记录到Wandb"""
    all_results = {}
    
    # 生成一个唯一的运行名称，包含时间戳
    run_name = f"multi_benchmark_eval_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 初始化Wandb
    wandb.init(
        project="eval-lightning-demo",
        name=run_name,
        config={"benchmarks": dataset_names}
    )
    
    # 为每个模型step创建评估结果
    for model in models:
        model_name = model.__name__
        model_step = model.step
        print(f"\nEvaluating model {model_name} (step {model_step})...")
        
        # 在所有数据集上评估当前模型
        metrics = {}
        
        for dataset_name in dataset_names:
            print(f"  Evaluating on {dataset_name}...")
            dataset = create_dataset(dataset_name, size=30)
            
            # 评估模型在当前数据集上的表现
            scores = []
            for sample in dataset:
                score = model(sample)
                scores.append(score)
            
            # 计算平均分数
            avg_score = sum(scores) / len(scores) if scores else 0
            
            # 存储结果
            if dataset_name not in all_results:
                all_results[dataset_name] = {}
            
            all_results[dataset_name][model_name] = {
                'scores': scores,
                'avg_score': avg_score,
                'step': model_step
            }
            
            # 添加到当前step的metrics中
            metrics[f"{dataset_name}_score"] = avg_score
            
            print(f"    Score: {avg_score:.4f}")
        
        # 一次性记录所有数据集的分数到Wandb，确保它们在同一个step
        wandb.log(metrics, step=model_step)
    
    # 关闭Wandb
    wandb.finish()
    
    return all_results


def main():
    # 定义6个数据集
    dataset_names = ['MathVista', 'MathVision', 'MathVerse', 'DynaMath', 'WeMath', 'LogicVista']
    
    # 创建10个不同训练步骤的模型 (200到2000)
    step_values = list(range(200, 2001, 200))  # [200, 400, 600, ..., 2000]
    models = [
        StepModel(step=step, quality=0.3 + (0.5 * i / (len(step_values) - 1)))
        for i, step in enumerate(step_values)
    ]
    
    print(f"Created {len(models)} models with steps from {step_values[0]} to {step_values[-1]}")
    print(f"Evaluating on {len(dataset_names)} benchmarks: {', '.join(dataset_names)}")
    
    # 评估所有数据集
    results = evaluate_on_datasets(models, dataset_names)
    
    # 打印每个数据集的结果
    print("\nResults summary:")
    for dataset_name, model_results in results.items():
        print(f"\n{dataset_name}:")
        for model_name, result in model_results.items():
            print(f"  {model_name}: {result['avg_score']:.4f} (step: {result['step']})")
    
    # 保存结果
    save_results(results, "wandb_evaluation_results.json")
    print("\nResults saved to wandb_evaluation_results.json")
    print("\nCheck your Wandb dashboard to see the tracked results with separate plots for each benchmark!")


if __name__ == "__main__":
    main() 