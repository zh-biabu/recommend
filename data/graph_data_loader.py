"""
Optimized data loading pipeline for graph-based recommendation systems.
Handles multi-modal features, graph construction, and efficient data loading.
"""

import os
import torch
import numpy as np
import pandas as pd
import random
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
import pickle
from sklearn.model_selection import train_test_split
from collections import defaultdict
import warnings


class GraphRecommendationDataset(Dataset):
    """Enhanced dataset for graph-based recommendation with multi-modal features."""
    
    def __init__(
        self,
        config,
        negative_sampling: bool = True,
        neg_ratio: int = 100,
        mode: str = "train",  # train, val, test
        transform: Optional[callable] = None,
    ):
        """
        Initialize graph recommendation dataset.
        
        Args:
            data_path: Path to data directory
            user_col: User ID column name
            item_col: Item ID column name
            rating_col: Rating column name
            timestamp_col: Timestamp column name (optional)
            mode: Dataset mode (train/val/test)
            transform: Optional data transformation
        """
        self.config = config
        self.data_path = config.data.data_path
        self.user_col = config.data.user_col
        self.item_col = config.data.item_col
        self.rating_col = config.data.rating_col
        self.timestamp_col = config.data.timestamp_col
        self.negative_sampling = negative_sampling
        self.neg_ratio = neg_ratio
        self.mode = mode
        self.transform = transform
        self.num_users = config.data.num_users
        self.num_items = config.data.num_items
        
        # Load and process data
        self._load_interactions()
        self._caculate_num()
        self._load_features()
        if self.negative_sampling and self.mode == "train":
            self._generate_negative_samples()
        self.get_statistics()
        
        print(f"Dataset {mode} loaded: {len(self.data)} interactions, "
              f"{self.num_users} users, {self.num_items} items, {self.data.columns}")
    
    def _load_interactions(self):
        """Load interaction data."""
        print(os.path.abspath(self.data_path))
        for file in os.listdir(self.data_path):
            if file.endswith('.csv') and self.mode in file:
                file_path = os.path.join(self.data_path, file)
                self.data = pd.read_csv(file_path)
                break
        else:
            raise FileNotFoundError(f"No .inter file found in {self.data_path}")
        
        # Validate required columns
        required_cols = [self.user_col, self.item_col]
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {file} {missing_cols}")

    def _caculate_num(self):
        if self.num_users == -1:
            file_path = os.path.join(self.data_path, "u_id_mapping.csv")
            self.num_users = len(pd.read_csv(file_path))
        if self.num_items == -1:
            file_path = os.path.join(self.data_path, "i_id_mapping.csv")
            self.num_items = len(pd.read_csv(file_path))

    
    def _load_features(self):
        """Load multi-modal features."""
        self.user_features = {}
        self.item_features = {}
        
        for file in os.listdir(self.data_path):
            file_path = os.path.join(self.data_path, file)
            
            if file.endswith('.npy'):
                features = np.load(file_path, allow_pickle=True)
                
                if 'user' in file.lower():
                    feature_name = file.replace('.npy', '').replace('user', '')
                    self.user_features[feature_name] = torch.tensor(features, dtype=torch.float32)
                elif 'item' in file.lower():
                    feature_name = file.replace('.npy', '').replace('item', '')
                    self.item_features[feature_name] = torch.tensor(features, dtype=torch.float32)
        
        # Ensure all features have the same number of users/items
        for name, features in self.user_features.items():
            if features.shape[0] != self.num_users:
                warnings.warn(f"User feature {name} has {features.shape[0]} rows, expected {self.num_users} user num")
        
        for name, features in self.item_features.items():
            if features.shape[0] != self.num_items:
                warnings.warn(f"Item feature {name} has {features.shape[0]} rows, expected {self.num_items} item num")

    def _generate_negative_samples(self):
        print("Generating negative samples...")
        if self.neg_ratio <=0:
            raise Exception("The number of negative samples must be greater than zero.")
        user_items = defaultdict(set)
        for _, row in self.data.iterrows():
            user_items[row[self.user_col]].add(row[self.item_col])
        all_items = set(range(self.num_items))
        neg_samples = []
        for _, row in self.data.iterrows():
            user_id = row[self.user_col]
            pos_items = user_items[user_id]
            neg_pool = list(all_items - pos_items)
            if len(neg_pool) >= self.neg_ratio:
                neg_samples.append(random.sample(neg_pool, k=self.neg_ratio))
            else:
                neg_samples.append(neg_pool)
                warnings.warn(f"The number of remaining samples is less than the preset value of {self.neg_ratio}, which may lead to errors.")
        self.data['negs'] = neg_samples

    def get_statistics(self):
        """Get dataset statistics."""
        stats = {
            'num_users': self.num_users,
            'num_items': self.num_items,
            'num_interactions': len(self.data),
            'sparsity': 1 - (len(self.data) / (self.num_users * self.num_items)),
            'user_features': list(self.user_features.keys()),
            'item_features': list(self.item_features.keys()),
        }
        print("show dataset basic salution")
        for k,v in stats.items():
            print(k,":",v)
        return stats

    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.data.iloc[idx]
        
        # Basic information
        user_id = torch.tensor(sample[self.user_col], dtype=torch.long)
        item_id = torch.tensor(sample[self.item_col], dtype=torch.long)
        
        
        result = {
            'user_id': user_id,
            'item_id': item_id,
        }
        
        if self.rating_col:
            rating = torch.tensor(sample[self.rating_col], dtype=torch.float32)
            result["rating"] = rating

        if self.negative_sampling and self.mode == "train":
            neg_items_id = torch.tensor(sample["negs"], dtype=torch.long)
            result["neg_items_id"] = neg_items_id

        # Add timestamp if available
        if self.timestamp_col and self.timestamp_col in sample:
            result['timestamp'] = torch.tensor(sample[self.timestamp_col], dtype=torch.long)
        
        # Add features
        # if self.user_features:
        #     user_feat_idx = sample[self.user_col].astype(int)
        #     user_features = {}
        #     for name, features in self.user_features.items():
        #         user_features[name] = features[user_feat_idx]
        #     result['user_features'] = user_features
        
        # if self.item_features:
        #     item_feat_idx = sample[self.item_col].astype(int) - self.item_offset
        #     item_features = {}
        #     for name, features in self.item_features.items():
        #         item_features[name] = features[item_feat_idx]
        #     result['item_features'] = item_features
        
        if self.transform:
            result = self.transform(result)
        
        return result
    




class GraphDataLoader:
    """Data loader for graph-based recommendation with collate function."""
    
    def __init__(
        self,
        config,
        dataset: GraphRecommendationDataset,
        shuffle: bool = True,
        pin_memory: bool = False
    ):
        self.config = config
        self.dataset = dataset
        self.batch_size = config.data.batch_size
        self.shuffle = shuffle
        self.num_workers = config.data.num_workers
        self.pin_memory = pin_memory
        
        # Create data loader
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch):
        """Custom collate function for batching."""
        # Separate different types of data
        user_ids = torch.stack([item['user_id'] for item in batch])
        item_ids = torch.stack([item['item_id'] for item in batch])
        
        # is_positive = torch.stack([item['is_positive'] for item in batch])
        
        result = {
            'user_ids': user_ids,
            'item_ids': item_ids,
            # 'is_positive': is_positive
        }

        if "rating" in batch[0]:
            ratings = torch.stack([item['rating'] for item in batch])
            result["ratings"] = ratings

        if "neg_items_id" in batch[0]:
            neg_items = torch.stack([item['neg_items_id'] for item in batch])
            result["neg_items"] = neg_items
        
        # Handle features
        if 'user_features' in batch[0]:
            user_features = {}
            for key in batch[0]['user_features'].keys():
                user_features[key] = torch.stack([item['user_features'][key] for item in batch])
            result['user_features'] = user_features
        
        if 'item_features' in batch[0]:
            item_features = {}
            for key in batch[0]['item_features'].keys():
                item_features[key] = torch.stack([item['item_features'][key] for item in batch])
            result['item_features'] = item_features
        
        # Handle timestamp if available
        if 'timestamp' in batch[0]:
            timestamps = torch.stack([item['timestamp'] for item in batch])
            result['timestamps'] = timestamps
        
        return result
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)


def create_data_loaders(config) -> Tuple[GraphDataLoader, GraphDataLoader, GraphDataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = GraphRecommendationDataset(
        config,
        mode="train", negative_sampling=config.data.negative_sampling, neg_ratio=config.data.neg_ratio
    )
    val_dataset = GraphRecommendationDataset(
        config,
        mode="val", negative_sampling=False
    )
    test_dataset = GraphRecommendationDataset(
        config,
        mode="test", negative_sampling=False
    )
    
    # Create data loaders
    train_loader = GraphDataLoader(
        config,
        train_dataset, shuffle=True
    )
    val_loader = GraphDataLoader(
        config,
        val_dataset, shuffle=False
    )
    test_loader = GraphDataLoader(
        config,
        test_dataset, shuffle=False
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the data loader
    import sys
    from pathlib import Path

    # 将项目根目录加入 Python 搜索路径
    sys.path.append(str(Path(__file__).parent.parent))  # 根据实际结构调整层级
    print(Path(__file__))

    from config import get_config 
    config = get_config("baby")
    train_loader, val_loader, test_loader = create_data_loaders(
        config
    )
    print("dataset output {user_id item_id [rating neg_items_id]}")
    print("loader output {user_ids item_ids [ratings neg_items]}")
    # Test a batch
    for batch in train_loader:
        print("Batch keys:", batch.keys())
        print("User IDs shape:", batch['user_ids'].shape)
        print("Item IDs shape:", batch['item_ids'].shape)
        print("Ratings shape:", batch['ratings'].shape)
        print("negs shape:", batch['neg_items'].shape)
        break
