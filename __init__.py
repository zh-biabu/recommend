"""
Graph-based Recommendation System

A comprehensive framework for multi-modal graph-based recommendation systems
with support for various datasets, models, and evaluation metrics.
"""

__version__ = "1.0.0"
__author__ = "Recommendation System Team"

from .config import get_config, Config
from .model.mmgcn_model import ModelFactory, MMGCNModel
from .data.graph_data_loader import create_data_loaders, GraphRecommendationDataset
from .train.graph_trainer import GraphTrainer
from .evalue.metric import evaluate_all_at_k
from .evalue.loss import bpr_loss

__all__ = [
    'get_config',
    'Config',
    'ModelFactory',
    'MMFCNModel',
    'create_data_loaders',
    'GraphRecommendationDataset',
    'GraphTrainer',
    'evaluate_all_at_k',
    'bpr_loss'
]
