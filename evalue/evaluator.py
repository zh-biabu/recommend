import torch
import numpy as np
from tqdm import tqdm
from .metric import evaluate_all_at_k
from typing import Dict, List, Tuple, Optional, Any

def _move_batch_to_device(batch: Dict[str, torch.Tensor], device) -> Dict[str, torch.Tensor]:
    """Move batch to device."""
    device_batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            device_batch[key] = value.to(device)
        elif isinstance(value, dict):
            device_batch[key] = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                                for k, v in value.items()}
        else:
            device_batch[key] = value
    return device_batch

class Verifier:
    """
    verifier for graph-based recommendation models.
    """
    def __init__(self, config, loader, target, mask):
        self.config = config
        self.loader = loader
        self.target = target
        self.row_id, self.col_id = mask

        self.metrics = config.evaluation.metrics # 要计算的指标列表
        self.device = config.system.device
        self.k_list = config.evaluation.k_values  # 不同的k值

        
    
    def verify(self, model):
        """
        验证模型性能
        Args:
            model: 要验证的模型
            val_loader: 验证数据加载器
        Returns:
            metrics_results: 所有指标的结果字典
        """
        model.eval()
        metrics_results = {}
        
        all_scores = []
        all_targets = []

        outputs = model({})
        user_embeddings = outputs["user_embeddings"]
        item_embeddings = outputs["item_embeddings"]
        pre_score = (torch.matmul(user_embeddings, item_embeddings.T).to("cpu"))
        pre_score[self.row_id, self.col_id] = -1e10
        

        # 计算各项指标
        for k in self.k_list:
            all_metric = evaluate_all_at_k(pre_score, self.target, k)
            for metric_name in self.metrics:
                    metrics_results[f'{metric_name}@{k}'] = all_metric[metric_name]
        return metrics_results


class Tester:
    """
    tester for graph-based recommendation models.
    """
    def __init__(self, config, loader, target, mask):
        self.config = config
        self.loader = loader
        self.target = target
        self.row_id, self.col_id = mask

        self.metrics = config.evaluation.metrics # 要计算的指标列表
        self.device = config.system.device
        self.k_list = config.evaluation.k_values  # 不同的k值
    
    def test(self, model):
        """
        测试模型性能
        Args:
            model: 要测试的模型
            test_loader: 测试数据加载器
        Returns:
            metrics_results: 所有指标的结果字典
        """
        model.eval()
        metrics_results = {}
        
        all_scores = []
        all_targets = []

        outputs = model({})
        user_embeddings = outputs["user_embeddings"]
        item_embeddings = outputs["item_embeddings"]
        pre_score = (torch.matmul(user_embeddings, item_embeddings.T).to("cpu"))
        pre_score[self.row_id, self.col_id] = -1e10
        
        

        # 计算各项指标
        for k in self.k_list:
            all_metric = evaluate_all_at_k(pre_score, self.target, k)
            for metric_name in self.metrics:
                    metrics_results[f'{metric_name}@{k}'] = all_metric[metric_name]
        return metrics_results