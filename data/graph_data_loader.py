import os
import torch
import numpy as np
import pandas as pd
import random
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
import warnings
from collections import defaultdict


class GraphRecDataset(Dataset):
    """
    Unified dataset class for graph-based recommendation systems.
    Supports both full dataset loading and subset creation.
    """
    
    def __init__(self, config, df=None, mode: str = "train"):
        """
        Initialize dataset.
        
        Args:
            config: Configuration object
            df: DataFrame for subset creation (None for full dataset)
            mode: Dataset mode (train/val/test)
        """
        self.config = config
        self.df = df
        self.mode = mode
        self.parent = None  # Reference to parent dataset if this is a subset
        self.is_subset = df is not None
        
        # Initialize common attributes
        self.user_col = config.data.user_col
        self.item_col = config.data.item_col
        self.rating_col = config.data.rating_col
        self.timestamp_col = config.data.timestamp_col
        self.splitting_label = config.data.splitting_label
        self.filter_out_new_users = config.data.filter_out_new_users
        
        # Initialize dataset-specific attributes
        self.user_features = {}
        self.item_features = {}
        self.user_item_interaction = None
        self.all_items = None
        self.user_num = config.data.num_users
        self.item_num = config.data.num_items
        
        if self.is_subset:
            self._init_subset()
        else:
            self._init_full_dataset()
    
    def _init_full_dataset(self):
        """Initialize full dataset with all data and features."""
        print(f"Initializing full dataset from {self.config.data.data_path}")
        
        # Load interaction data
        self._load_interactions()
        
        # Calculate number of users and items
        self._calculate_num()
        
        # Load multi-modal features
        self._load_features()
        
        # Build user-item interaction index
        self._build_user_item_interaction()
        
        # Print statistics
        self.get_statistics()
        
        print(f"Full dataset initialized: {len(self.df)} interactions, "
              f"{self.user_num} users, {self.item_num} items")
    
    def _init_subset(self):
        """Initialize subset dataset."""
        print(f"Initializing subset dataset (mode: {self.mode}) with {len(self.df)} interactions")
        
        # Validate required columns
        required_cols = [self.user_col, self.item_col]
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in subset: {missing_cols}")
    
    def _load_interactions(self):
        """Load interaction data for full dataset."""
        print(f"Loading interaction data from {self.config.data.data_path}")
        
        # Find the interaction file
        interaction_file = None
        for file in os.listdir(self.config.data.data_path):
            if file.endswith('.inter'):
                interaction_file = file
                break
        
        if not interaction_file:
            raise FileNotFoundError(f"No interaction file found in {self.config.data.data_path}")
        
        file_path = os.path.join(self.config.data.data_path, interaction_file)
        self.df = pd.read_csv(file_path, sep=self.config.data.sep)
        
        # Validate required columns
        required_cols = [self.user_col, self.item_col, self.splitting_label]
        if self.rating_col:
            required_cols.append(self.rating_col)
        if self.timestamp_col:
            required_cols.append(self.timestamp_col)
        
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    def _calculate_num(self):
        """Calculate number of users and items if not provided in config."""
        if self.user_num == -1:
            file_path = os.path.join(self.config.data.data_path, "u_id_mapping.csv")
            if os.path.exists(file_path):
                self.user_num = len(pd.read_csv(file_path))
            else:
                self.user_num = len(self.df[self.user_col].unique())
        
        if self.item_num == -1:
            file_path = os.path.join(self.config.data.data_path, "i_id_mapping.csv")
            if os.path.exists(file_path):
                self.item_num = len(pd.read_csv(file_path))
            else:
                self.item_num = len(self.df[self.item_col].unique())
    
    def _load_features(self):
        """Load multi-modal features for full dataset."""
        print(f"Loading features from {self.config.data.data_path}")
        
        for file in os.listdir(self.config.data.data_path):
            file_path = os.path.join(self.config.data.data_path, file)
            
            if file.endswith('.npy'):
                try:
                    features = np.load(file_path, allow_pickle=True)
                    
                    if 'user' in file.lower():
                        feature_name = file.replace('.npy', '').replace('user_', '').replace('_user', '').replace('user', '')
                        self.user_features[feature_name] = torch.tensor(features, dtype=torch.float32)
                    elif 'item' in file.lower():
                        feature_name = file.replace('.npy', '').replace('item_', '').replace('_item', '').replace('item', '')
                        self.item_features[feature_name] = torch.tensor(features, dtype=torch.float32)
                except Exception as e:
                    warnings.warn(f"Failed to load feature file {file}: {e}")
        
        # Validate feature dimensions
        self._validate_features()
    
    def _validate_features(self):
        """Validate that features have correct dimensions."""
        for name, features in self.user_features.items():
            if features.shape[0] != self.user_num:
                warnings.warn(f"User feature {name} has {features.shape[0]} rows, expected {self.user_num}")
        
        for name, features in self.item_features.items():
            if features.shape[0] != self.item_num:
                warnings.warn(f"Item feature {name} has {features.shape[0]} rows, expected {self.item_num}")
    
    def _build_user_item_interaction(self):
        """Build user-item interaction index for efficient negative sampling."""
        print("Building user-item interaction index...")
        self.user_item_interaction = defaultdict(set)
        for _, row in self.df.iterrows():
            user_id = row[self.user_col]
            item_id = row[self.item_col]
            self.user_item_interaction[user_id].add(item_id)
        
        # Get all unique items
        self.all_items = set(self.df[self.item_col].unique())

    def get_features(self):
        root_dataset = self
        while root_dataset.parent:
            root_dataset = root_dataset.parent
        return root_dataset.user_features, root_dataset.item_features
    
    def get_statistics(self):
        """Get dataset statistics."""
        if self.is_subset and self.parent:
            return self.parent.get_statistics()
        
        stats = {
            'num_users': self.user_num,
            'num_items': self.item_num,
            'num_interactions': len(self.df),
            'sparsity': 1 - (len(self.df) / (self.user_num * self.item_num)),
            'user_features': list(self.user_features.keys()),
            'item_features_dimensions': {k: v.shape[1] for k, v in self.item_features.items()},
            'user_feature_dimensions': {k: v.shape[1] for k, v in self.user_features.items()},
        }
        
        if not self.is_subset:
            stats.update({
                'train_ratio': len(self.df[self.df[self.splitting_label] == 0]) / len(self.df),
                'val_ratio': len(self.df[self.df[self.splitting_label] == 1]) / len(self.df),
                'test_ratio': len(self.df[self.df[self.splitting_label] == 2]) / len(self.df),
            })
        
        print("Dataset Statistics:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        
        return stats
    
    def split(self):
        """
        Split the full dataset into train, validation, and test subsets.
        
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        if self.is_subset:
            raise ValueError("Cannot split a subset dataset")
        
        print("Splitting dataset into train/val/test...")
        dfs = []
        
        # Split based on splitting label
        for i in range(3):  # 0: train, 1: val, 2: test
            temp_df = self.df[self.df[self.splitting_label] == i].copy()
            temp_df.drop(self.splitting_label, inplace=True, axis=1)
            dfs.append(temp_df)
        
        # Filter out new users in val/test sets if enabled
        if self.filter_out_new_users:
            train_users = set(dfs[0][self.user_col].values)
            for i in [1, 2]:  # val and test sets
                # Keep only users present in training set
                mask = dfs[i][self.user_col].isin(train_users)
                removed_count = len(dfs[i]) - mask.sum()
                if removed_count > 0:
                    warnings.warn(f"Filtered out {removed_count} interactions with new users in {'val' if i==1 else 'test'} set")
                dfs[i] = dfs[i][mask]
        
        # Create subset datasets
        train_dataset = self.copy(dfs[0], mode="train")
        val_dataset = self.copy(dfs[1], mode="val")
        test_dataset = self.copy(dfs[2], mode="test")
        
        print(f"Split results - Train: {len(train_dataset)} interactions, "
              f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset
    
    def copy(self, new_df, mode="train"):
        """
        Create a new subset dataset from a DataFrame.
        
        Args:
            new_df: DataFrame for the new subset
            mode: Dataset mode (train/val/test)
            
        Returns:
            New GraphRecDataset instance
        """
        # Create new dataset instance
        new_dataset = GraphRecDataset(self.config, new_df, mode)
        
        # Set parent reference
        new_dataset.parent = self
        
        # Copy user and item counts
        new_dataset.user_num = self.user_num
        new_dataset.item_num = self.item_num
        
        return new_dataset
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.df.iloc[idx]
        
        # Basic information
        user_id = torch.tensor(sample[self.user_col], dtype=torch.long)
        item_id = torch.tensor(sample[self.item_col], dtype=torch.long)
        
        result = {
            'user_id': user_id,
            'item_id': item_id,
        }
        
        # Add rating if available
        if self.rating_col and self.rating_col in sample:
            rating = torch.tensor(sample[self.rating_col], dtype=torch.float32)
            result["rating"] = rating
        
        # Add timestamp if available
        if self.timestamp_col and self.timestamp_col in sample:
            result['timestamp'] = torch.tensor(sample[self.timestamp_col], dtype=torch.long)
        
        # # Get the root dataset for feature access
        # root_dataset = self
        # while root_dataset.parent is not None:
        #     root_dataset = root_dataset.parent
        
        # # Add user features
        # if root_dataset.user_features:
        #     user_features = {}
        #     for name, features in root_dataset.user_features.items():
        #         user_features[name] = features[user_id]
        #     result['user_features'] = user_features
        
        # # Add item features
        # if root_dataset.item_features:
        #     item_features = {}
        #     for name, features in root_dataset.item_features.items():
        #         item_features[name] = features[item_id]
        #     result['item_features'] = item_features
        
        return result


class GraphDataLoader:
    """Data loader for graph-based recommendation with dynamic negative sampling."""
    
    def __init__(
        self,
        config,
        dataset: GraphRecDataset,
        shuffle: bool = True,
        pin_memory: bool = False
    ):
        self.config = config
        self.dataset = dataset
        self.batch_size = config.data.batch_size
        self.shuffle = shuffle
        self.num_workers = config.data.num_workers
        self.pin_memory = pin_memory
        self.negative_sampling = config.data.negative_sampling if hasattr(config.data, 'negative_sampling') else False
        self.neg_ratio = config.data.neg_ratio if hasattr(config.data, 'neg_ratio') else 100
        
        # Get root dataset for interaction data access
        self.root_dataset = dataset
        while self.root_dataset.parent is not None:
            self.root_dataset = self.root_dataset.parent
        
        # Create data loader
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
            collate_fn=self._collate_fn
        )
    
    def _generate_negative_samples(self, user_ids: torch.Tensor) -> torch.Tensor:
        """
        Dynamically generate negative samples for a batch of users.
        
        Args:
            user_ids: Batch of user IDs
            
        Returns:
            Tensor of negative item IDs with shape (batch_size, neg_ratio)
        """
        if self.neg_ratio <= 0:
            raise ValueError("Negative ratio must be greater than zero")
        
        if self.root_dataset.user_item_interaction is None:
            raise ValueError("User-item interaction index not available in root dataset")
        
        neg_items = []
        
        for user_id in user_ids:
            user_id = user_id.item()
            # Get positive items for this user
            pos_items = self.root_dataset.user_item_interaction.get(user_id, set())
            # Get candidate negative items (all items not in positive)
            neg_candidates = list(self.root_dataset.all_items - pos_items)
            
            if len(neg_candidates) == 0:
                warnings.warn(f"No negative candidates available for user {user_id}")
                # Use a dummy item if no candidates available
                neg_candidates = [0]
            
            # Sample negative items
            if len(neg_candidates) >= self.neg_ratio:
                sampled_neg = random.sample(neg_candidates, k=self.neg_ratio)
            else:
                # If not enough candidates, sample with replacement
                sampled_neg = random.choices(neg_candidates, k=self.neg_ratio)
                warnings.warn(f"Not enough negative candidates for user {user_id}. "
                              f"Requested {self.neg_ratio}, but only {len(neg_candidates)} available. "
                              "Using sampling with replacement.")
            
            neg_items.append(sampled_neg)
        
        return torch.tensor(neg_items, dtype=torch.long)
    
    def _collate_fn(self, batch):
        """Custom collate function for batching with dynamic negative sampling."""
        if not batch:
            return {}
        
        # Separate different types of data
        user_ids = torch.stack([item['user_id'] for item in batch])
        item_ids = torch.stack([item['item_id'] for item in batch])
        
        result = {
            'user_ids': user_ids,
            'item_ids': item_ids,
        }
        
        # Add ratings if available
        if 'rating' in batch[0]:
            ratings = torch.stack([item['rating'] for item in batch])
            result["ratings"] = ratings
        
        # Add timestamps if available
        if 'timestamp' in batch[0]:
            timestamps = torch.stack([item['timestamp'] for item in batch])
            result['timestamps'] = timestamps
        
        # Add user features if available
        if 'user_features' in batch[0]:
            user_features = {}
            for key in batch[0]['user_features'].keys():
                user_features[key] = torch.stack([item['user_features'][key] for item in batch])
            result['user_features'] = user_features
        
        # Add item features if available
        if 'item_features' in batch[0]:
            item_features = {}
            for key in batch[0]['item_features'].keys():
                item_features[key] = torch.stack([item['item_features'][key] for item in batch])
            result['item_features'] = item_features
        
        # Generate negative samples dynamically for training
        if self.negative_sampling and self.dataset.mode == "train":
            result["neg_items"] = self._generate_negative_samples(user_ids)
        
        return result
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)


def create_data_loaders(config) -> Tuple[GraphDataLoader, GraphDataLoader, GraphDataLoader]:
    """
    Create train, validation, and test data loaders with optimized data loading.
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create full dataset (loads all data and features once)
    full_dataset = GraphRecDataset(config)
    
    # Split into train/val/test subsets
    train_dataset, val_dataset, test_dataset = full_dataset.split()
    
    # Create data loaders
    train_loader = GraphDataLoader(
        config,
        train_dataset, 
        shuffle=True
    )
    
    val_loader = GraphDataLoader(
        config,
        val_dataset, 
        shuffle=False
    )
    
    test_loader = GraphDataLoader(
        config,
        test_dataset, 
        shuffle=False
    )
    
    print(f"Data loaders created - Train batches: {len(train_loader)}, "
          f"Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


# Example usage and test
def test_data_pipeline():
    """Test the data pipeline with a sample configuration."""
    # Create a mock config
    class Config:
        pass
    
    config = Config()
    config.data = Config()
    config.data.data_path = "./data"  # Update with your data path
    config.data.user_col = "user_id"
    config.data.item_col = "item_id"
    config.data.rating_col = "rating"
    config.data.timestamp_col = "timestamp"
    config.data.splitting_label = "split"
    config.data.filter_out_new_users = True
    config.data.num_users = -1
    config.data.num_items = -1
    config.data.batch_size = 32
    config.data.num_workers = 4
    config.data.negative_sampling = True
    config.data.neg_ratio = 10
    
    try:
        # Create full dataset
        print("Creating full dataset...")
        full_dataset = GraphRecDataset(config)
        
        # Test split method
        print("\nTesting split method...")
        train_dataset, val_dataset, test_dataset = full_dataset.split()
        
        print(f"Train dataset: {len(train_dataset)} interactions, is_subset: {train_dataset.is_subset}")
        print(f"Val dataset: {len(val_dataset)} interactions, is_subset: {val_dataset.is_subset}")
        print(f"Test dataset: {len(test_dataset)} interactions, is_subset: {test_dataset.is_subset}")
        
        # Test feature access through subset
        print("\nTesting feature access...")
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            print(f"Sample keys: {list(sample.keys())}")
            if 'user_features' in sample:
                print(f"User features available: {list(sample['user_features'].keys())}")
            if 'item_features' in sample:
                print(f"Item features available: {list(sample['item_features'].keys())}")
        
        # Create data loaders
        print("\nCreating data loaders...")
        train_loader, val_loader, test_loader = create_data_loaders(config)
        
        # Test training loader with negative sampling
        print("\nTesting training loader...")
        for batch_idx, batch in enumerate(train_loader):
            print(f"Batch {batch_idx + 1}:")
            print(f"  User IDs shape: {batch['user_ids'].shape}")
            print(f"  Item IDs shape: {batch['item_ids'].shape}")
            if 'ratings' in batch:
                print(f"  Ratings shape: {batch['ratings'].shape}")
            if 'neg_items' in batch:
                print(f"  Negative items shape: {batch['neg_items'].shape}")
            
            # Stop after first batch for testing
            if batch_idx == 0:
                break
        
        print("\nData pipeline test completed successfully!")
        
    except Exception as e:
        print(f"Error testing data pipeline: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_data_pipeline()
