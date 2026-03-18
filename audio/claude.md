# TimeSliver - Audio Classification

This folder contains the TimeSliver implementation for **10-class classification on audio time series data**.

## Directory Structure

```
audio/
├── main/
│   ├── config.py              # Centralized configuration and paths
│   ├── utils.py               # Shared utilities (data loading, evaluation)
│   ├── timesliver.py          # TimeSliver network architecture
│   ├── train_model.py         # Training script with CLI args
│   ├── test_model.py          # Testing/evaluation script
│   ├── temporal_attribution.py # Compute importance scores
│   ├── train_with_masking.py  # Train with masked time points
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

## Key Parameters (Audio Specific)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `NUM_CLASSES` | 10 | 10-class audio classification |
| `SAX_ALPHABET_SIZE` | 400 | SAX discretization levels (very high) |
| `D_IN` | 40 | Input feature dimension |
| `D_MODEL` | 40 | Projection dimension (q) |
| `D_OUT` | 12 | CNN output channels (d) |
| `MAX_M` | 1 | Number of motif scales |
| `CNN_BRANCHES` | 1 | Single branch (kernel=1 only) |
| `POSITIONAL_ENCODING` | No | Not used in forward pass |
| `REDUCTION` | AvgPool2d((1,3)) | 2D pooling |
| `LINEAR_INPUT` | d_out * 133 * max_m | Final layer input size |

## Architecture Details

### Key Features (Audio)
1. **Single CNN branch** - Point-wise convolutions with kernel=1
2. **NO positional encoding** - Not used in forward pass
3. **AvgPool2d reduction** - `(1, 3)` pooling kernel
4. **Very high SAX alphabet** - 400 discretization levels
5. **10 classes** - Multi-class audio classification

### Model Architecture (Audio)
```
Input (x) ─────────────────────────────────┐
    │                                      │
    ▼                                      │
┌──────────────────────────────────────────┤
│  Projection Layer (d_in → d_model)       │
│  (40 → 40)                               │
└──────────────────────────────────────────┤
    │                                      │
    ▼                                      │
┌──────────────────────────────────────────┤
│  [NO Positional Encoding]                │
└──────────────────────────────────────────┤
    │                                      │
    ▼                                      │
┌──────────────────────────────────────────┤
│  Single CNN Branch (point-wise)          │
│  ├── Conv1d (d_model → 16, k=1)          │
│  ├── ReLU                                │
│  ├── Conv1d (16 → 32, k=1)               │
│  ├── ReLU                                │
│  └── Conv1d (32 → d_out, k=1)            │
│  → No receptive field expansion          │
└──────────────────────────────────────────┤
    │                                      │
    ▼                                      │
┌──────────────────────────────────────────┤
│  AvgPool2d((1, 3))                       │
└──────────────────────────────────────────┤
    │                                      │
    ▼                                      │
┌──────────────────────────────────────────┐
│  Linear (d_out*133*max_m → 10 classes)   │
│  (1596 → 10)                             │
└──────────────────────────────────────────┘
    │
    ▼
  Logits (10 classes)
```

## File Descriptions

### `config.py`
Centralized configuration containing:
- Dataset parameters (NUM_CLASSES=10, SAX_ALPHABET_SIZE=400, D_IN=40)
- Model hyperparameters (D_MODEL=40, D_OUT=12, MAX_M=1)
- Training defaults (epochs=5000, lr=0.0003)
- Path utilities

### `utils.py`
Shared utilities:
- `load_data()` - Load train/valid/test data
- `create_dataloader()` - Create PyTorch DataLoader
- `evaluate_model()` - Compute accuracy, confusion matrix
- `print_metrics()` - Display evaluation results
- `masking_function()` - Mask time points by importance
- `create_masked_dataloader()` - DataLoader with masking

### `timesliver.py`
TimeSliver network architecture:
- `TimeSlicerDataset` - PyTorch Dataset class
- `TimeSliverNetwork` - Main model with single CNN branch
- **NO positional encoding in forward()**
- Point-wise convolutions (kernel=1)
- `AvgPool2d((1, 3))` reduction

### `train_model.py`
Training script with CLI arguments:
```bash
python train_model.py --epochs 5000 --lr 0.0003 --device cuda
python train_model.py --d-out 12 --max-m 1 --no-tensorboard
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
Train model using only top-k% important time points:
```bash
python train_with_masking.py --top-percent 20 --device cuda
python train_with_masking.py --top-percent 50 --method captum --sub-method ig
```

### `test_with_masking.py`
Evaluate masked model:
```bash
python test_with_masking.py --device cuda
python test_with_masking.py --top-percent 30
```

## Quick Start

```bash
cd /home/apa2237/TimeSliver/audio/main

# 1. Train the model
python train_model.py --epochs 5000 --lr 0.0003 --device cuda

# 2. Evaluate on test set
python test_model.py --device cuda

# 3. Compute temporal attribution scores
python temporal_attribution.py --split all --device cuda

# 4. Train with masking (keep top 20% important time points)
python train_with_masking.py --top-percent 20 --device cuda

# 5. Test masked model
python test_with_masking.py --device cuda
```

## Notes

- **TensorBoard is optional** - Due to TensorFlow/numpy version conflicts, TensorBoard logging is wrapped in try/except.
- **Point-wise architecture** - All CNN kernels are size 1, so no temporal motif extraction.
- **Very high SAX alphabet** - SAX_ALPHABET_SIZE=400 for fine discretization.
- **10-class classification** - More classes than other datasets.
- **Model checkpoints** - Saved to `model/best.pth` and `model/best_masking.pth`.
