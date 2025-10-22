"""
Main entry point for graph-based recommendation system.
Orchestrates the entire pipeline from data loading to model evaluation.
"""

import os
import sys
from dgl.view import defaultdict
import torch
import argparse
import random
import numpy as np
from pathlib import Path

# Add current directory to path for imports
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_CURRENT_DIR)

from config import get_config
from data.graph_data_loader import create_data_loaders
from model import ModelFactory
# from model.graph_constructor import GraphConstructor
from train import GraphTrainer
from log.deep_learning_logger import get_logger
from evalue import Verifier,Tester


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_device(config) -> str:
    """Setup and return the device to use."""
    if config.system.device == "cuda" and torch.cuda.is_available():
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        config.system.device ="cpu"
        device = "cpu"
        print("Using CPU")
    
    return device


def prepare_data(config):
    """Prepare data loaders and extract features."""
    print("Loading data...")
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(config)
    
    # Extract dataset information
    train_dataset = train_loader.dataset
    
    # Extract features
    user_features = train_dataset.user_features
    item_features = train_dataset.item_features
    
    print(f"Data loaded: {config.data.num_users} users, {config.data.num_items} items")
    print(f"User features: {list(user_features.keys())}")
    print(f"Item features: {list(item_features.keys())}")
    
    return train_loader, val_loader, test_loader, user_features, item_features


def build_graph_and_model(config, train_loader, user_features, item_features):
    """Build the recommendation graph and model."""
    print("Building graph and model...")
    
    # Create model
    model = ModelFactory.create_MMGCN(
        config=config,
        user_features=user_features,
        item_features=item_features
    )
    
    # Build graph ONLY from training interactions (critical for avoiding data leakage)
    train_dataset = train_loader.dataset
    interactions = []
    
    print("Extracting training interactions for graph construction...")
    
    # Extract ONLY positive interactions from training data
    for idx in range(len(train_dataset)):
        sample = train_dataset[idx]
        user_id = sample['user_id'].item()
        item_id = sample['item_id'].item()  # Remove offset
        if "rating" in sample:
            rating = sample['rating'].item()
        else:
            rating = 1
        interactions.append((user_id, item_id, rating))
    
    
    print(f"Extracted {len(interactions)} positive interactions for graph construction")
    
    # Build graph using ONLY training data
    graph = model.build_graph(interactions)
    model.graph_constructor.creat_feature_weight()
    model.graph_constructor.creat_feature_weight(feature="text_feat")
    
    print(f"Graph built from training data only: {graph.num_nodes()} nodes, {graph.num_edges()} edges")
    print("⚠️  Important: Graph constructed using only training data to prevent data leakage")
    
    return model, graph

def mask_index(config, target_loader, need_mask_loaders):
    user_col = config.data.user_col
    item_col = config.data.item_col
    num_users = config.data.num_users
    num_items = config.data.num_items
    target_df = target_loader.dataset.data
    need_mask_df = [loader.dataset.data for loader in need_mask_loaders]
    target = torch.zeros((num_users, num_items))
    mask = torch.ones((num_users, num_items))
    target_cache = target_df.groupby(user_col)[item_col].apply(list).to_dict()

    for user_id, item_id in target_cache.items():
        target[user_id][item_id] = 1

    for df in need_mask_df:
        mask_cache = df.groupby(user_col)[item_col].apply(list).to_dict()
        for user_id, item_id in mask_cache.items():
            mask[user_id][item_id] = -float("inf")
            
    return target,mask

 

def train_model(config, model, train_loader, val_loader, test_loader):
    """Train the model."""
    print("Starting training...")
    
    # Setup logger
    logger = get_logger("RecommendationSystem")
    
    # Create trainer
    trainer = GraphTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        logger=logger
    )
    
    # Train
    training_results = trainer.train()
    
    # Save best model
    best_model_path = os.path.join(config.system.save_dir, "best_model.pth")
    trainer.save_best_model(best_model_path)
    
    return training_results, trainer


def evaluate_model(config, model, test_loader, device):
    """Evaluate the model on test set."""
    print("Evaluating model...")
    
    model.eval()
    all_scores = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Move batch to device
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)
                elif isinstance(value, dict):
                    batch[key] = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                                 for k, v in value.items()}
            
            # Forward pass
            result = model.forward(batch)
            scores = result['scores']
            
            all_scores.append(scores.cpu())
            all_targets.append(batch['ratings'].cpu())
    
    # Combine results
    all_scores = torch.cat(all_scores, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Convert to binary targets
    binary_targets = (all_targets > 0).float()
    
    # Evaluate metrics
    from evalue.metric import evaluate_all_at_k
    
    test_metrics = {}
    for k in config.evaluation.k_values:
        batch_scores = all_scores.unsqueeze(0)
        batch_targets = binary_targets.unsqueeze(0)
        
        k_metrics = evaluate_all_at_k(batch_scores, batch_targets, k)
        for metric_name, value in k_metrics.items():
            test_metrics[f"test_{metric_name}@{k}"] = value.item()
    
    return test_metrics


def save_results(config, training_results, test_metrics, model_info):
    """Save results to files."""
    print("Saving results...")
    
    # Save training results
    results = {
        'training_results': training_results,
        'test_metrics': test_metrics,
        'model_info': model_info,
        'config': config.to_dict()
    }
    
    results_path = os.path.join(config.system.results_dir, "results.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    import json
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"Results saved to {results_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Graph-based Recommendation System")
    parser.add_argument("--dataset", type=str, default="baby", 
                       help="Dataset name (baby, clothing, sports, elec)")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to custom config file")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (cpu, cuda, auto)")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Load configuration
    config = get_config(args.dataset, args.config)
    
    # Override device if specified
    if args.device != "auto":
        config.system.device = args.device
    
    if args.seed != None:
        config.system.seed = args.seed
    
    # Set seed
    set_seed(config.system.seed)

    # Setup device
    device = setup_device(config)
    
    print("=" * 60)
    print("Graph-based Recommendation System")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Device: {device}")
    print(f"Model: {config.model.model_name}")
    print(f"Embedding dim: {config.model.emb_dim}")
    print(f"Epochs: {config.training.epochs}")
    print("=" * 60)
    
    try:
        # Prepare data
        train_loader, val_loader, test_loader, user_features, item_features = prepare_data(config)
        # Build graph and model
        model, graph = build_graph_and_model(config, train_loader, user_features, item_features)
        model = model.to(device)
        
        # Get model info
        model_info = model.get_model_info()
        print(f"Model parameters: {model_info['total_parameters']:,}")
        print(model.graph_constructor.get_graph_statistics())

        # init trainer,verifier,tester
        print(f"init trainer,verifier,tester")

        val_target, val_mask = mask_index(config, val_loader, [train_loader, test_loader])
        test_target, test_mask = mask_index(config, test_loader,[train_loader, val_loader])

        trainer = GraphTrainer(model, train_loader, config)
        verifier = Verifier(config, val_loader,val_target, val_mask)
        tester = Tester(config, test_loader,test_target, test_mask)

        # Train model
        training_results = trainer.train(verifier)
        
        # Evaluate model
        test_metrics = tester.test(model)
        
        # Print results
        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print("Training Results:")
        print(f"  Best epoch: {training_results['best_epoch']}")
        print(f"  Best validation metric: {training_results['best_val_metric']:.4f}")
        print(f"  Training time: {training_results['training_time']/3600:.2f} hours")
        
        print("\nTest Metrics:")
        for metric, value in test_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Save results
        save_results(config, training_results, test_metrics, model_info)
        
        print("\nTraining completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())