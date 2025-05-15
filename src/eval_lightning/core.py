from typing import List, Callable, Any, Dict, Optional
from dataclasses import dataclass
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import wandb

@dataclass
class EvalResult:
    """评估结果数据类"""
    score: float
    metadata: Optional[Dict] = None
    step: Optional[int] = None


class EvalLightning:
    """
    轻量级模型评估框架
    
    通过提供模型列表和数据集，对每个模型在所有样本上进行评估，
    并通过聚合函数计算最终得分。
    """
    
    def __init__(
        self,
        models: List[Callable],
        dataset: List[Any],
        aggregate_fn: Optional[Callable] = None,
        verbose: bool = True,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_run_name: Optional[str] = None
    ):
        """
        初始化评估框架
        
        Args:
            models: 模型列表，每个模型是一个接收样本并返回分数的可调用对象
            dataset: 数据集，样本列表
            aggregate_fn: 聚合函数，用于合并多个样本的分数
            verbose: 是否显示详细信息
            use_wandb: 是否使用Wandb记录结果
            wandb_project: Wandb项目名称
            wandb_entity: Wandb实体名称
            wandb_run_name: Wandb运行名称
        """
        self.models = models
        self.dataset = dataset
        self.aggregate_fn = aggregate_fn or self._default_aggregate
        self.verbose = verbose
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.wandb_run_name = wandb_run_name
        self.wandb_run = None
    
    def evaluate(self) -> Dict:
        """
        运行评估过程
        
        Returns:
            包含每个模型评估结果的字典
        """
        results = {}
        
        # 初始化Wandb
        if self.use_wandb:
            self.wandb_run = wandb.init(
                project=self.wandb_project,
                entity=self.wandb_entity,
                name=self.wandb_run_name,
                config={"dataset_size": len(self.dataset)}
            )
        
        for model in self.models:
            model_name = getattr(model, "__name__", str(model))
            if self.verbose:
                print(f"Evaluating {model_name}...")
            
            start_time = time.time()
            scores = []
            
            # 获取模型的step属性（如果存在）
            model_step = getattr(model, "step", 0)
            
            for i, sample in enumerate(self.dataset):
                score = model(sample)
                scores.append(score)
                
                if self.verbose and (i + 1) % 10 == 0:
                    print(f"  Processed {i + 1}/{len(self.dataset)} samples")
            
            final_score = self.aggregate_fn(scores)
            elapsed_time = time.time() - start_time
            
            results[model_name] = {
                'scores': scores,
                'final_score': final_score,
                'time_taken': elapsed_time,
                'step': model_step
            }
            
            # 记录到Wandb
            if self.use_wandb:
                wandb.log({
                    f"score": final_score,
                    f"time": elapsed_time
                }, step=model_step)
            
            if self.verbose:
                print(f"  Final score: {final_score:.4f}")
                print(f"  Time taken: {elapsed_time:.2f}s")
        
        # 关闭Wandb
        if self.use_wandb and self.wandb_run:
            self.wandb_run.finish()
        
        return results
    
    def evaluate_parallel(self, max_workers: int = None) -> Dict:
        """
        并行运行评估过程
        
        Args:
            max_workers: 最大工作进程数
            
        Returns:
            包含每个模型评估结果的字典
        """
        results = {}
        
        # 初始化Wandb
        if self.use_wandb:
            self.wandb_run = wandb.init(
                project=self.wandb_project,
                entity=self.wandb_entity,
                name=self.wandb_run_name,
                config={"dataset_size": len(self.dataset)}
            )
        
        for model in self.models:
            model_name = getattr(model, "__name__", str(model))
            if self.verbose:
                print(f"Evaluating {model_name} in parallel...")
            
            start_time = time.time()
            scores = [None] * len(self.dataset)
            
            # 获取模型的step属性（如果存在）
            model_step = getattr(model, "step", 0)
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_index = {
                    executor.submit(model, sample): i 
                    for i, sample in enumerate(self.dataset)
                }
                
                completed = 0
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    scores[index] = future.result()
                    
                    completed += 1
                    if self.verbose and completed % 10 == 0:
                        print(f"  Processed {completed}/{len(self.dataset)} samples")
            
            final_score = self.aggregate_fn(scores)
            elapsed_time = time.time() - start_time
            
            results[model_name] = {
                'scores': scores,
                'final_score': final_score,
                'time_taken': elapsed_time,
                'step': model_step
            }
            
            # 记录到Wandb
            if self.use_wandb:
                wandb.log({
                    f"{model_name}_score": final_score,
                    f"{model_name}_time": elapsed_time
                }, step=model_step)
            
            if self.verbose:
                print(f"  Final score: {final_score:.4f}")
                print(f"  Time taken: {elapsed_time:.2f}s")
        
        # 关闭Wandb
        if self.use_wandb and self.wandb_run:
            self.wandb_run.finish()
        
        return results
    
    @staticmethod
    def _default_aggregate(scores: List[float]) -> float:
        """默认使用平均值聚合"""
        if not scores:
            return 0.0
        return sum(scores) / len(scores) 