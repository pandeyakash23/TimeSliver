# TimeSliver - Synthetic Data Classification

This folder contains the TimeSliver implementation for **binary classification on synthetic time series data**.

## Directory Structure

```
synthetic_data/
├── main/
│   ├── config.py              # Centralized configuration and paths
│   ├── utils.py               # Shared utilities (data loading, evaluation)
│   ├── timesliver.py          # TimeSliver network architecture
│   ├── train_model.py         # Training script with CLI args
│   ├── test_model.py          # Testing/evaluation script
│   ├── temporal_attribution.py # Compute importance scores
│   ├── train_with_masking.py  # Train on masked time points
│   └── test_with_masking.py   # Test masked model
├── data/
│   ├── x_train.npy            # Training features
│   ├── x_valid.npy            # Validation features
│   ├── x_test.npy             # Test features
│   ├── y_train.npy            # Training labels
│   ├── y_valid.npy            # Validation labels
│   ├── y_test.npy             # Test labels
│   ├── sax_train.npy          # SAX-encoded training data
│   ├── sax_valid.npy          # SAX-encoded validation data
│   └── sax_test.npy           # SAX-encoded test data
├── model/                     # Saved model checkpoints
└── claude.md                  # This documentation
```

## Key Parameters (Synthetic Data Specific)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `NUM_CLASSES` | 2 | Binary classification |
| `SAX_ALPHABET_SIZE` | 90 | SAX discretization levels |
| `D_OUT` | 36 | CNN output channels |
| `MAX_M` | 1 | Number of motif scales |
| `CNN_KERNEL_SIZE` | 3 | Convolution kernel size |
| `POOL_KERNEL_SIZE` | 3 | Pooling kernel size |

## File Descriptions

### `config.py`
Centralized configuration containing:
- Dataset parameters (NUM_CLASSES, SAX_ALPHABET_SIZE)
- Model hyperparameters (D_OUT, MAX_M, kernel sizes)
- Training defaults (epochs, learning rate, batch sizes)
- Path utilities (`get_data_path()`, `get_model_path()`, etc.)
- Device detection (`get_device()`)

### `utils.py`
Shared utilities:
- `load_data(split, load_sax=True)` - Load train/valid/test data
- `create_dataloader()` - Create PyTorch DataLoader
- `create_masked_dataloader()` - DataLoader with importance masking
- `evaluate_model()` - Compute accuracy, AUC-ROC, precision, recall
- `print_metrics()` - Display evaluation results
- `count_parameters()` - Count trainable parameters

### `timesliver.py`
TimeSliver network architecture:
- `TimeSlicerDataset` - PyTorch Dataset class
- `TimeSliverNetwork` - Main model
  - **No projection layer** (direct input, unlike EEG/FordA)
  - Multi-scale CNN with SAX embedding
  - Motif-document interaction matrices
  - `forward_motif_importance()` for attribution
  - `calculate_motif_level_new()` for importance computation

### `train_model.py`
Training script with CLI arguments:
```bash
python train_model.py --epochs 2500 --lr 0.001 --device cuda
python train_model.py --d-out 36 --max-m 1 --no-tensorboard
```

### `test_model.py`
Evaluation script:
```bash
python test_model.py --device cuda
python test_model.py --split valid --model-path /path/to/model.pth
```

### `temporal_attribution.py`
Compute per-time-point importance scores:
```bash
python temporal_attribution.py --split all --device cuda
python temporal_attribution.py --split train --batch-size 512
```

### `train_with_masking.py`
Train using only top-k% important time points:
```bash
python train_with_masking.py --top-percent 20 --device cuda
python train_with_masking.py --top-percent 50 --method main
```

### `test_with_masking.py`
Evaluate masked model:
```bash
python test_with_masking.py --device cuda
python test_with_masking.py --top-percent 30  # Override saved percentage
```

## Architecture Details

### Model Architecture (Synthetic Data)
```
Input (x) ────────────────────────────────┐
    │                                     │
    ▼                                     │
[No Projection - Direct Input]            │
    │                                     │
    ▼                                     │
┌─────────────────────────────────────────┤
│  Multi-scale CNN (kernel=3)             │
│  ├── Scale 1: motif size 7              │
│  └── ... (max_m scales)                 │
└─────────────────────────────────────────┤
    │                                     │
    ▼                                     │
┌─────────────────────────────────────────┤
│  SAX Embedding (vocab=90)               │
│  + Positional Encoding                  │
└─────────────────────────────────────────┤
    │                                     │
    ▼                                     │
┌─────────────────────────────────────────┤
│  Motif-Document Interaction             │
│  (BMM between CNN output & SAX embed)   │
└─────────────────────────────────────────┤
    │                                     │
    ▼                                     │
┌─────────────────────────────────────────┤
│  ReLU + MaxPool (kernel=3)              │
└─────────────────────────────────────────┤
    │                                     │
    ▼                                     │
┌─────────────────────────────────────────┐
│  Linear (9*11*max_m → num_classes)      │
└─────────────────────────────────────────┘
    │
    ▼
  Logits (2 classes)
```

### Key Differences from EEG/FordA
1. **No projection layer** - Input goes directly to CNN (no `nn.Linear(d_model, d_model)`)
2. **SAX alphabet size = 90** (vs 25 for EEG, 40 for FordA)
3. **Binary classification** (2 classes)
4. **CNN kernel size = 3** (creates 7-size motifs)
5. **Uses `calculate_motif_level_new()`** method for attribution

## Workflow

```
┌─────────────────┐
│ 1. Train Model  │
│ train_model.py  │
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│ 2. Test Model       │
│ test_model.py       │
└────────┬────────────┘
         │
         ▼
┌─────────────────────────┐
│ 3. Temporal Attribution │
│ temporal_attribution.py │
└────────┬────────────────┘
         │
         ▼
┌──────────────────────────┐
│ 4. Train with Masking    │
│ train_with_masking.py    │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ 5. Test Masked Model     │
│ test_with_masking.py     │
└──────────────────────────┘
```

## Quick Start

```bash
cd /home/apa2237/TimeSliver/synthetic_data/main

# 1. Train the base model
python train_model.py --epochs 2500 --lr 0.001 --device cuda

# 2. Evaluate on test set
python test_model.py --device cuda

# 3. Compute temporal attribution scores
python temporal_attribution.py --split all --device cuda

# 4. Train with top 20% important time points
python train_with_masking.py --top-percent 20 --device cuda

# 5. Evaluate masked model
python test_with_masking.py --device cuda
```

## Notes

- **TensorBoard is optional** - Due to TensorFlow/numpy version conflicts, TensorBoard logging is wrapped in try/except. Training works without it.
- **Binary metrics** - Evaluation includes AUC-ROC, precision, and recall for binary classification.
- **d_model is data-dependent** - The model dimension is automatically extracted from the data feature dimension.
- **Model checkpoints** - Saved to `model/best.pth` (base) and `model/best_masking.pth` (masked).
