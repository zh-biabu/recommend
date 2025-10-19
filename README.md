# Graph-based Recommendation System

A comprehensive framework for multi-modal graph-based recommendation systems with support for various datasets, models, and evaluation metrics.

## Features

- **Multi-modal Graph Construction**: Support for user and item features from multiple modalities (image, text, etc.)
- **Flexible Model Architecture**: MMFCN (Multi-Modal Feature Combination Network) with graph neural networks
- **Comprehensive Evaluation**: Multiple metrics including Precision@K, Recall@K, NDCG@K, MAP@K, MRR@K
- **Configurable Training**: Support for various optimizers, schedulers, and training strategies
- **Easy Experimentation**: YAML/JSON configuration system for easy hyperparameter tuning

## Project Structure

```
recommend/
├── config.py                 # Configuration system
├── main.py                   # Main entry point
├── data/
│   ├── graph_data_loader.py  # Enhanced data loading
│   └── dataset.py           # Original dataset classes
├── model/
│   ├── mmfcn_model.py       # Enhanced MMFCN model wrapper
│   ├── graph_constructor.py # Graph construction utilities
│   ├── MMFCN/              # Original MMFCN implementation
│   └── common/             # Common utilities
├── train/
│   ├── graph_trainer.py    # Enhanced training pipeline
│   └── optimize.py         # Optimizer utilities
├── evalue/
│   ├── metric.py           # Evaluation metrics
│   └── loss.py             # Loss functions
└── log/
    └── deep_learning_logger.py  # Logging utilities
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install torch torch-geometric dgl pandas numpy scikit-learn pyyaml tensorboard
```

## Usage

### Basic Usage

```bash
# Train on baby dataset with default settings
python main.py --dataset baby

# Train on clothing dataset with custom device
python main.py --dataset clothing --device cuda

# Use custom configuration file
python main.py --dataset baby --config configs/custom_config.yaml
```

### Command Line Arguments

- `--dataset`: Dataset name (baby, clothing, sports, elec)
- `--config`: Path to custom configuration file
- `--device`: Device to use (cpu, cuda, auto)
- `--seed`: Random seed for reproducibility

### Configuration

The system uses a flexible configuration system. You can:

1. Use predefined configurations for each dataset
2. Create custom YAML or JSON configuration files
3. Override specific parameters programmatically

Example configuration:

```yaml
data:
  data_path: "./data/scale_data/baby"
  batch_size: 512
  negative_sampling: true
  neg_ratio: 1

model:
  model_name: "MMFCN"
  emb_dim: 64
  layer_num: 2
  dropout: 0.3

training:
  epochs: 100
  learning_rate: 0.001
  optimizer: "adam"
  scheduler: "plateau"
```

## Data Format

The system expects data in the following format:

```
data/scale_data/{dataset_name}/
├── {dataset_name}.inter      # Interaction data (user_id, item_id, rating)
├── u_id_mapping.csv          # User ID mappings (optional)
├── i_id_mapping.csv          # Item ID mappings (optional)
├── user{modal}feat.npy       # User features for each modality
└── item{modal}feat.npy       # Item features for each modality
```

## Model Architecture

The MMFCN model consists of:

1. **Graph Construction**: Builds bipartite graph from user-item interactions
2. **Multi-modal Feature Integration**: Combines features from different modalities
3. **Graph Neural Network**: Uses graph convolution for node representation learning
4. **Final Prediction**: Computes user-item interaction scores

## Evaluation Metrics

The system supports comprehensive evaluation metrics:

- **Precision@K**: Fraction of recommended items that are relevant
- **Recall@K**: Fraction of relevant items that are recommended
- **Hit Rate@K**: Fraction of users with at least one relevant recommendation
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **MAP@K**: Mean Average Precision
- **MRR@K**: Mean Reciprocal Rank

## Training Features

- **Early Stopping**: Prevents overfitting based on validation metrics
- **Learning Rate Scheduling**: Supports plateau, step, and cosine annealing
- **Gradient Clipping**: Prevents exploding gradients
- **Mixed Precision**: Optional support for faster training
- **Checkpointing**: Automatic model saving and resuming

## Logging and Monitoring

- **TensorBoard Integration**: Real-time training monitoring
- **Comprehensive Logging**: Detailed training and evaluation logs
- **Results Saving**: Automatic saving of training results and metrics

## Example Results

```
============================================================
FINAL RESULTS
============================================================
Training Results:
  Best epoch: 45
  Best validation metric: 0.1234
  Training time: 1.25 hours

Test Metrics:
  test_precision@5: 0.1234
  test_recall@5: 0.2345
  test_ndcg@5: 0.3456
  test_map@5: 0.2345
  test_mrr@5: 0.4567
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{graph_recommendation_system,
  title={Graph-based Recommendation System},
  author={Recommendation System Team},
  year={2024},
  url={https://github.com/your-repo/graph-recommendation-system}
}
```
