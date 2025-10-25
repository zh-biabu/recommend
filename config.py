"""
Configuration system for graph recommendation system.
Supports YAML, JSON, and Python dictionary configurations.
"""

import os
import yaml
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""
    data_path: str = "../autodl-tmp/data/scale_data/baby"
    user_col: str = "userID"
    item_col: str = "itemID"
    rating_col: Optional[str] = None
    timestamp_col: Optional[str] = None
    negative_sampling: bool = True
    neg_ratio: int = 1
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    batch_size: int = 256
    num_workers: int = 4
    # Data leakage prevention
    use_time_split: bool = True  # Use temporal split to prevent data leakage
    graph_from_train_only: bool = True  # Build graph only from training data
    num_users : int = -1
    num_items : int = -1
    


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    model_name: str = "MMGCN"
    modal_num: int = 2
    emb_dim: int = 64
    layer_num: int = 2
    dropout: float = 0.3
    activation: str = "prelu"
    use_batch_norm: bool = True
    hidden_dim: int = 256
    concat: bool = False
    k: int = 3


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 100
    learning_rate: float = 0.0001
    weight_decay: float = 1e-4
    optimizer: str = "adam"
    scheduler: str = "plateau"
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    early_stopping_patience: int = 20
    gradient_clip_norm: float = 1.0
    warmup_epochs: int = 5
    eval_every: int = 1
    save_every: int = 10


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    metrics: list = None
    k_values: list = None
    test_batch_size: int = 512
    num_negatives: int = 100
    main_metric: str = "ndcg"
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["precision", "recall", "hit_rate", "ndcg", "map", "mrr"]
        if self.k_values is None:
            self.k_values = [5, 10, 20]


@dataclass
class SystemConfig:
    """System configuration."""
    device: str = "cuda"
    seed: int = 42
    log_level: str = "INFO"
    save_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    results_dir: str = "./results"
    num_gpus: int = 1
    mixed_precision: bool = False


@dataclass
class GraphConfig:
    """Graph construction configuration."""
    graph_type: str = "bipartite"
    add_self_loops: bool = True
    normalize_adj: bool = True
    edge_weight_type: str = "cosine"  # cosine, dot, uniform
    max_neighbors: int = 50


class Config:
    """Main configuration class that combines all configs."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        self.data = DataConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.evaluation = EvaluationConfig()
        self.system = SystemConfig()
        self.graph = GraphConfig()
        
        if config_dict:
            self.update_from_dict(config_dict)
    
    def update_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary."""
        for section_name, section_config in config_dict.items():
            if hasattr(self, section_name):
                section = getattr(self, section_name)
                for key, value in section_config.items():
                    if hasattr(section, key):
                        setattr(section, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'data': asdict(self.data),
            'model': asdict(self.model),
            'training': asdict(self.training),
            'evaluation': asdict(self.evaluation),
            'system': asdict(self.system),
            'graph': asdict(self.graph)
        }
    
    def save_to_yaml(self, filepath: str):
        """Save configuration to YAML file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)
    
    def save_to_json(self, filepath: str):
        """Save configuration to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def from_yaml(cls, filepath: str) -> 'Config':
        """Load configuration from YAML file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)
    
    @classmethod
    def from_json(cls, filepath: str) -> 'Config':
        """Load configuration from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls(config_dict)


# Predefined configurations for common scenarios
def get_baby_config() -> Config:
    """Get configuration for baby dataset."""
    config = Config()
    config.data.data_path = "../autodl-tmp/data/scale_data/baby"
    config.data.user_col = "userID"
    config.data.item_col = "itemID"
    config.data.rating_col = "rating"
    config.data.batch_size = 512
    config.model.emb_dim = 64
    config.training.epochs = 100
    config.training.learning_rate = 0.001
    config.data.num_users=19445
    config.data.num_items=7050
    config.graph.weight_feature = [None]
    return config


def get_clothing_config() -> Config:
    """Get configuration for clothing dataset."""
    config = Config()
    config.data.data_path = "./data/scale_data/clothing"
    config.data.batch_size = 1024
    config.model.emb_dim = 128
    config.training.epochs = 100
    config.training.learning_rate = 0.0005
    return config


def get_sports_config() -> Config:
    """Get configuration for sports dataset."""
    config = Config()
    config.data.data_path = "./data/scale_data/sports"
    config.data.batch_size = 256
    config.model.emb_dim = 64
    config.training.epochs = 80
    config.training.learning_rate = 0.001
    return config


def get_elec_config() -> Config:
    """Get configuration for electronics dataset."""
    config = Config()
    config.data.data_path = "./data/scale_data/elec"
    config.data.batch_size = 256
    config.model.emb_dim = 64
    config.training.epochs = 60
    config.training.learning_rate = 0.001
    return config


# Configuration registry
CONFIG_REGISTRY = {
    'baby': get_baby_config,
    'clothing': get_clothing_config,
    'sports': get_sports_config,
    'elec': get_elec_config,
}


def get_config(dataset_name: str = 'baby', config_file: Optional[str] = None) -> Config:
    """
    Get configuration for a dataset.
    
    Args:
        dataset_name: Name of the dataset ('baby', 'clothing', 'sports', 'elec')
        config_file: Optional path to custom config file (YAML or JSON)
    
    Returns:
        Config object
    """
    if config_file:
        if config_file.endswith('.yaml') or config_file.endswith('.yml'):
            return Config.from_yaml(config_file)
        elif config_file.endswith('.json'):
            return Config.from_json(config_file)
        else:
            raise ValueError(f"Unsupported config file format: {config_file}")
    
    if dataset_name in CONFIG_REGISTRY:
        return CONFIG_REGISTRY[dataset_name]()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(CONFIG_REGISTRY.keys())}")


if __name__ == "__main__":
    # Example usage
    config = get_config('baby')
    print("Baby dataset configuration:")
    print(f"Data path: {config.data.data_path}")
    print(f"Model: {config.model.model_name}")
    print(f"Embedding dim: {config.model.emb_dim}")
    print(f"Training epochs: {config.training.epochs}")
    
    # Save config
    config.save_to_yaml("./configs/baby_config.yaml")
    config.save_to_json("./configs/baby_config.json")
