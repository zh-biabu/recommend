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
        data_path: str,
        user_col: str = "userID",
        item_col: str = "itemID",
        rating_col: str = "rating",
        timestamp_col: Optional[str] = None,
        negative_sampling: bool = True,
        neg_ratio: int = 100,
        mode: str = "train",  # train, val, test
        transform: Optional[callable] = None
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
        self.data_path = data_path
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.timestamp_col = timestamp_col
        self.negative_sampling = negative_sampling
        self.neg_ratio = neg_ratio
        self.mode = mode
        self.transform = transform
        
        # Load and process data
        self._load_interactions()
        self._build_mappings()
        self._load_features()
        if self.negative_sampling and self.mode == "train":
            self._generate_negative_samples()
        self.get_statistics()
        self.num_users = len(self.user2id)
        self.num_items = len(self.item2id)
        
        print(f"Dataset {mode} loaded: {len(self.data)} interactions, "
              f"{len(self.user2id)} users, {len(self.item2id)} items, {self.data.columns}")
    
    def _load_interactions(self):
        """Load interaction data."""
        for file in os.listdir(self.data_path):
            if file.endswith('.csv') and self.mode in file:
                file_path = os.path.join(self.data_path, file)
                self.data = pd.read_csv(file_path)
                break
        else:
            raise FileNotFoundError(f"No .inter file found in {self.data_path}")
        
        # Validate required columns
        required_cols = [self.user_col, self.item_col, self.rating_col]
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {file} {missing_cols}")
    
    def _build_mappings(self):
        """Build user and item ID mappings."""
        for file in os.listdir(self.data_path):
            # print(file)
            if file.endswith('.csv') and "u_id_mapping" in file:
                file_path = os.path.join(self.data_path, file)
                df = pd.read_csv(file_path)
                self.id2user = df[self.user_col].to_dict()
                self.user2id = pd.Series(df.index, index=df[self.user_col]).to_dict()
            if file.endswith('.csv') and "i_id_mapping" in file:
                file_path = os.path.join(self.data_path, file)
                df = pd.read_csv(file_path)
                self.id2item = df[self.item_col].to_dict()
                self.item2id = pd.Series(df.index, index=df[self.item_col]).to_dict()
        print(len(self.id2item),len(self.user2id))
        
        # Add offset for items (to distinguish from users in graph)
        self.num_nodes = len(self.user2id) + len(self.item2id)
        
        # Map IDs
        self.data['user_id_mapped'] = self.data[self.user_col].map(self.user2id)
        self.data['item_id_mapped'] = self.data[self.item_col].map(self.item2id)
    
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
            if features.shape[0] != len(self.user2id):
                warnings.warn(f"User feature {name} has {features.shape[0]} rows, expected {len(self.user2id)}")
        
        for name, features in self.item_features.items():
            if features.shape[0] != len(self.item2id):
                warnings.warn(f"Item feature {name} has {features.shape[0]} rows, expected {len(self.item2id)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.data.iloc[idx]
        
        # Basic information
        user_id = torch.tensor(sample['user_id_mapped'], dtype=torch.long)
        item_id = torch.tensor(sample['item_id_mapped'], dtype=torch.long)
        rating = torch.tensor(sample[self.rating_col], dtype=torch.float32)
        
        result = {
            'user_id': user_id,
            'item_id': item_id,
            'rating': rating,
        }
        
        if self.mode == "train":
            neg_items = torch.tensor(sample["neg_items"], dtype=torch.long)
            result["neg_items"] = neg_items

        # Add timestamp if available
        if self.timestamp_col and self.timestamp_col in sample:
            result['timestamp'] = torch.tensor(sample[self.timestamp_col], dtype=torch.long)
        
        # Add features
        # if self.user_features:
        #     user_feat_idx = sample['user_id_mapped'].astype(int)
        #     user_features = {}
        #     for name, features in self.user_features.items():
        #         user_features[name] = features[user_feat_idx]
        #     result['user_features'] = user_features
        
        # if self.item_features:
        #     item_feat_idx = sample['item_id_mapped'].astype(int) - self.item_offset
        #     item_features = {}
        #     for name, features in self.item_features.items():
        #         item_features[name] = features[item_feat_idx]
        #     result['item_features'] = item_features
        
        if self.transform:
            result = self.transform(result)
        
        return result
    
    def get_statistics(self):
        """Get dataset statistics."""
        stats = {
            'num_users': len(self.user2id),
            'num_items': len(self.item2id),
            'num_interactions': len(self.data),
            'sparsity': 1 - (len(self.data) / (len(self.user2id) * len(self.item2id))),
            'user_features': list(self.user_features.keys()),
            'item_features': list(self.item_features.keys()),
        }
        
        return stats

    def _generate_negative_samples(self):
        print("Generating negative samples...")
        user_items = defaultdict(set)
        for _, row in self.data.iterrows():
            user_items[row['user_id_mapped']].add(row['item_id_mapped'])
        all_items = set(range(len(self.item2id)))
        neg_samples = []
        for _, row in self.data.iterrows():
            user_id = row['user_id_mapped']
            pos_items = user_items[user_id]
            neg_pool = list(all_items - pos_items)
            neg_samples.append(random.choices(neg_pool, k=self.neg_ratio))
        self.data['neg_items'] = neg_samples


class GraphDataLoader:
    """Data loader for graph-based recommendation with collate function."""
    
    def __init__(
        self,
        dataset: GraphRecommendationDataset,
        batch_size: int = 256,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = False
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # Create data loader
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch):
        """Custom collate function for batching."""
        # Separate different types of data
        user_ids = torch.stack([item['user_id'] for item in batch])
        item_ids = torch.stack([item['item_id'] for item in batch])
        ratings = torch.stack([item['rating'] for item in batch])
        
        # is_positive = torch.stack([item['is_positive'] for item in batch])
        
        result = {
            'user_ids': user_ids,
            'item_ids': item_ids,
            'ratings': ratings,
            # 'is_positive': is_positive

        }

        if "neg_items" in  batch[0]:
            neg_items = torch.stack([item['neg_items'] for item in batch])
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


def create_data_loaders(
    data_path: str,
    batch_size: int = 256,
    num_workers: int = 4,
    negative_sampling: bool = True,
    neg_ratio: int = 100
) -> Tuple[GraphDataLoader, GraphDataLoader, GraphDataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = GraphRecommendationDataset(
        data_path, mode="train", negative_sampling=negative_sampling, neg_ratio=neg_ratio
    )
    val_dataset = GraphRecommendationDataset(
        data_path, mode="val", negative_sampling=False
    )
    test_dataset = GraphRecommendationDataset(
        data_path, mode="test", negative_sampling=False
    )
    
    # Create data loaders
    train_loader = GraphDataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = GraphDataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = GraphDataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the data loader
    train_loader, val_loader, test_loader = create_data_loaders(
        "./data/scale_data/baby",
        batch_size=32
    )
    
    # Test a batch
    for batch in train_loader:
        print("Batch keys:", batch.keys())
        print("User IDs shape:", batch['user_ids'].shape)
        print("Item IDs shape:", batch['item_ids'].shape)
        print("Ratings shape:", batch['ratings'].shape)
        break
