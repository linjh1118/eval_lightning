"""EvalLightning - 轻量级模型评估框架"""

from .core import EvalLightning, EvalResult
from .utils import (
    mean_aggregate,
    median_aggregate,
    max_aggregate,
    min_aggregate,
    weighted_aggregate,
    save_results,
    load_results
)

__version__ = "0.1.0"
__all__ = [
    "EvalLightning", 
    "EvalResult",
    "mean_aggregate",
    "median_aggregate",
    "max_aggregate",
    "min_aggregate",
    "weighted_aggregate",
    "save_results",
    "load_results"
] 