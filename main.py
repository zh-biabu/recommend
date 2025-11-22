"""
Main entry point for graph-based recommendation system.
Orchestrates the entire pipeline from data loading to model evaluation.
"""

import os
import sys
from dgl.view import defaultdict
import torch
import torch.nn.functional as F
import argparse
import random
import numpy as np
from pathlib import Path
import time
import json

# Add current directory to path for imports
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_CURRENT_DIR)

from config import get_config
from data.graph_data_loader import create_data_loaders
from model import ModelFactory
# from model.graph_constructor import GraphConstructor
from train import GraphTrainer
from log.deep_learning_logger import get_logger
from evalue import Verifier, Tester, mig_loss_func, mmgcn_loss


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
    user_features, item_features = train_loader.dataset.get_features()
    
    print(f"Data loaded: {config.data.num_users} users, {config.data.num_items} items")
    print(f"User features: {list(user_features.keys())}")
    print(f"Item features: {list(item_features.keys())}")
    
    return train_loader, val_loader, test_loader, user_features, item_features


def build_graph_and_model(config, train_loader, user_features, item_features):
    """Build the recommendation graph and model."""
    print("Building graph and model...")
    
    # Create model
    model = ModelFactory.create_Model(
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
    model.creat_feature_weight()

    try:
        print(f"Graph built from training data only: {graph.num_nodes()} nodes, {graph.num_edges()} edges")
    except Exception as e:
        pass
    print("⚠️  Important: Graph constructed using only training data to prevent data leakage")
    
    return model, graph

def mask_index(config, target_loader, need_mask_loaders):
    user_col = config.data.user_col
    item_col = config.data.item_col
    num_users = config.data.num_users
    num_items = config.data.num_items
    target_df = target_loader.dataset.df
    need_mask_df = [loader.dataset.df for loader in need_mask_loaders]
    target = torch.zeros((num_users, num_items))
    col_id = []
    row_id = []
    target_cache = target_df.groupby(user_col)[item_col].apply(list).to_dict()

    for user_id, item_id in target_cache.items():
        target[user_id][item_id] = 1


    for df in need_mask_df:
        mask_cache = df.groupby(user_col)[item_col].apply(list).to_dict()
        for user_id, item_id in mask_cache.items():
            col_id.extend(item_id)
            row_id.extend([user_id] * len(item_id))
    mask = (row_id, col_id)
    return target, mask

 

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
    
    # 生成带时间戳的文件名：results_20251022_1530.json
    timestamp = time.strftime("%Y%m%d_%H%M", time.localtime())
    filename = f"results_{timestamp}.json"  # 时间作为文件名的一部分

    # 拼接完整路径
    results_path = os.path.join(config.system.results_dir, filename)

    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    import json
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"Results saved to {results_path}")


def run_single_experiment(config, dataset_name: str):
    """
    执行一次完整的训练 + 验证 + 测试流程，返回结果字典。
    方便在超参数搜索时多次调用。
    """
    # Set seed
    set_seed(config.system.seed)

    # Setup device
    device = setup_device(config)

    print("=" * 60)
    print("Graph-based Recommendation System")
    print("=" * 60)
    print(f"Dataset: {dataset_name}")
    print(f"Device: {device}")
    print(f"Model: {config.model.model_name}")
    print(f"Embedding dim: {config.model.emb_dim}")
    print(f"Epochs: {config.training.epochs}")
    print("=" * 60)

    # Prepare data
    train_loader, val_loader, test_loader, user_features, item_features = prepare_data(config)
    # Build graph and model
    model, graph = build_graph_and_model(config, train_loader, user_features, item_features)
    model = model.to(device)

    print(model)

    # Get model info
    model_info = model.get_model_info()
    print(f"Model parameters: {model_info['total_parameters']:,}")

    # init trainer,verifier,tester
    print(f"init trainer,verifier,tester")

    val_target, val_mask = mask_index(config, val_loader, [train_loader])
    test_target, test_mask = mask_index(config, test_loader,[train_loader])

    trainer = GraphTrainer(model, train_loader, config, loss_func=model.loss_func)
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

    return {
        "config": config.to_dict(),
        "training_results": training_results,
        "test_metrics": test_metrics,
        "model_info": model_info,
    }


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
    parser.add_argument("--hparam_search", action="store_true", default=True,
                       help="Whether to run hyper-parameter search (Bayesian Optimization with Optuna)")
    parser.add_argument("--max_trials", type=int, default=10,
                       help="Number of trials for hyper-parameter search")
    
    args = parser.parse_args()

    # 超参数搜索模式（使用 Optuna 做 Bayesian Optimization）
    if args.hparam_search:
        try:
            import optuna
        except ImportError:
            print("需要安装 Optuna 才能使用 Bayesian Optimization 超参数搜索：pip install optuna")
            return 1

        all_trials = []
        trial_results = {}

        def objective(trial):
            # 为每个 trial 重新加载一份 config，避免相互污染
            config = get_config(args.dataset, args.config)

            # Override device if specified
            if args.device != "auto":
                config.system.device = args.device

            if args.seed is not None:
                # 使用基础 seed + trial 编号，保证可复现又有差异
                config.system.seed = args.seed + trial.number

            # 使用 Optuna 定义搜索空间（Bayesian Optimization 会基于历史 trial 自适应采样）
            # lr = trial.suggest_float("training.learning_rate", 1e-4, 1e-2, log=True)
            # weight_decay = trial.suggest_categorical("training.weight_decay", [1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
            layer_num = trial.suggest_int("model.layer_num", 1, 3)

            graph_v_k = trial.suggest_int("graph.v_k", 3, 10)
            graph_t_k = trial.suggest_int("graph.t_k", 3, 10)
            gcn_v_k = trial.suggest_int("model.gcn_v_k", 1, 10)
            gcn_t_k = trial.suggest_int("model.gcn_t_k", 1, 10)
            k = trial.suggest_int("model.k", 1, 10)

            alpha = trial.suggest_float("model.alpha", 0.1, 0.5, log=True)
            # beta = trial.suggest_float("model.beta", 0.1, 0.5, log=True)
            hidden_unit = trial.suggest_categorical("model.hidden_unit", [128, 256, 512])

            # config.training.learning_rate = lr
            # config.training.weight_decay = weight_decay
            config.model.layer_num = layer_num
            config.graph.v_k = graph_v_k
            config.graph.t_k = graph_t_k
            config.model.gcn_v_k = gcn_v_k
            config.model.gcn_t_k = gcn_t_k
            config.model.k = k
            config.model.alpha = alpha
            config.model.beta = 1-alpha
            config.model.hidden_unit = hidden_unit

            print("\n" + "#" * 60)
            print(f"Optuna Trial {trial.number}")
            print("#" * 60)
            print("Trial config (partial):")
            print(f"  lr={config.training.learning_rate}, "
                  f"wd={config.training.weight_decay}, "
                  f"layer_num={config.model.layer_num}, "
                  f"graph_v_k={config.graph.v_k}, "
                  f"graph_t_k={config.graph.t_k}, "
                  f"gcn_v_k={config.model.gcn_v_k}, "
                  f"gcn_t_k={config.model.gcn_t_k}, "
                  f"k={config.model.k}, "
                  f"alpha={config.model.alpha}, "
                  f"beta={config.model.beta}, "
                  f"hidden_unit={config.model.hidden_unit}")

            result = run_single_experiment(config, args.dataset)
            val_metric = float(result["training_results"]["best_val_metric"])

            summary = {
                "trial_id": trial.number,
                "val_metric": val_metric,
                "params": {
                    # "learning_rate": lr,
                    # "weight_decay": weight_decay,
                    "layer_num": layer_num,
                    "graph_v_k": graph_v_k,
                    "graph_t_k": graph_t_k,
                    "gcn_v_k": gcn_v_k,
                    "gcn_t_k": gcn_t_k,
                    "mmgcn_k": k,
                    "alpha": alpha,
                    "beta": 1-alpha,
                    "hidden_unit": hidden_unit,
                },
                "config": result["config"],
                "test_metrics": result["test_metrics"],
            }
            all_trials.append(summary)
            trial_results[trial.number] = summary

            return val_metric

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=args.max_trials)

        if len(study.trials) == 0:
            print("Hyper-parameter search failed: no successful trials.")
            return 1

        best_trial = study.best_trial
        best_summary = trial_results.get(best_trial.number, None)

        timestamp = time.strftime("%Y%m%d_%H%M", time.localtime())
        filename = f"hparam_search_optuna_{timestamp}.json"
        results_path = os.path.join("./results", filename)
        os.makedirs(os.path.dirname(results_path), exist_ok=True)

        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset": args.dataset,
                    "max_trials": args.max_trials,
                    "best_value": float(study.best_value),
                    "best_trial_number": best_trial.number,
                    "best_params": best_trial.params,
                    "best_summary": best_summary,
                    "all_trials": all_trials,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print("\n" + "=" * 60)
        print("Hyper-parameter search (Bayesian Optimization) finished")
        print(f"Best val metric: {study.best_value:.4f}")
        print("Best trial params：")
        for k, v in best_trial.params.items():
            print(f"  {k}: {v}")
        print(f"Results saved to {results_path}")

        return 0

    # 普通单次训练模式
    # Load configuration
    config = get_config(args.dataset, args.config)
    # config.save_to_yaml("./sgrec.yaml")
    # Override device if specified
    if args.device != "auto":
        config.system.device = args.device

    if args.seed is not None:
        config.system.seed = args.seed

    try:
        _ = run_single_experiment(config, args.dataset)
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())