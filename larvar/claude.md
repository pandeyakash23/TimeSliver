# TimeSliver - Larvar Classification

This folder contains the TimeSliver implementation for **4-class classification on Larvar time series data**.

## Directory Structure

```
larvar/
├── main/
│   ├── config.py              # Centralized configuration and paths
│   ├── utils.py               # Shared utilities (data loading, evaluation)
│   ├── timesliver.py          # TimeSliver network architecture
│   ├── train_model.py         # Training script with CLI args
│   ├── test_model.py          # Testing/evaluation script
│   └── temporal_attribution.py # Compute importance scores
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

## Key Parameters (Larvar Specific)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `NUM_CLASSES` | 4 | 4-class classification |
| `SAX_ALPHABET_SIZE` | 40 | SAX discretization levels (higher than SeqComb) |
| `D_OUT` | 36 | CNN output channels |
| `MAX_M` | 1 | Number of motif scales |
| `CNN_BRANCHES` | 1 | Single branch (4-size motifs only) |
| `POSITIONAL_ENCODING` | No | Not used in forward pass |
| `REDUCTION` | AvgPool2d | 2D pooling (vs 3D in SeqComb) |
| `LINEAR_INPUT` | 360 * max_m | Final layer input size |

## Architecture Details

### Key Differences from SeqComb
1. **Single CNN branch** - Only 4-size motifs (no 7-size branch)
2. **NO positional encoding** - Not used in forward pass
3. **AvgPool2d reduction** - 2D pooling instead of 3D
4. **Higher SAX alphabet** - 40 vs 20

### Model Architecture (Larvar)
```
Input (x) ─────────────────────────────────┐
    │                                      │
    ▼                                      │
┌──────────────────────────────────────────┤
│  Projection Layer (d_in → d_model)       │
└──────────────────────────────────────────┤
    │                                      │
    ▼                                      │
┌──────────────────────────────────────────┤
│  [NO Positional Encoding]                │
└──────────────────────────────────────────┤
    │                                      │
    ▼                                      │
┌──────────────────────────────────────────┤
│  Single CNN Branch                       │
│  ├── Conv1d (d_model → 16, k=2)          │
│  ├── ReLU                                │
│  ├── Conv1d (16 → 32, k=2)               │
│  ├── ReLU                                │
│  └── Conv1d (32 → d_out, k=2)            │
│  → Creates 4-size motifs                 │
└──────────────────────────────────────────┤
    │                                      │
    ▼                                      │
┌──────────────────────────────────────────┤
│  AvgPool2d((2,2))                        │
└──────────────────────────────────────────┤
    │                                      │
    ▼                                      │
┌──────────────────────────────────────────┐
│  Linear (360*max_m → 4 classes)          │
└──────────────────────────────────────────┘
    │
    ▼
  Logits (4 classes)
```

## File Descriptions

### `config.py`
Centralized configuration containing:
- Dataset parameters (NUM_CLASSES=4, SAX_ALPHABET_SIZE=40)
- Model hyperparameters (D_OUT=36, MAX_M=1)
- Training defaults (epochs=2500, lr=0.002)
- Path utilities

### `utils.py`
Shared utilities:
- `load_data()` - Load train/valid/test data
- `create_dataloader()` - Create PyTorch DataLoader
- `evaluate_model()` - Compute accuracy, confusion matrix
- `print_metrics()` - Display evaluation results

### `timesliver.py`
TimeSliver network architecture:
- `TimeSlicerDataset` - PyTorch Dataset class
- `TimeSliverNetwork` - Main model with single CNN branch
- **NO positional encoding in forward()**
- `AvgPool2d` reduction instead of `AvgPool3d`

### `train_model.py`
Training script with CLI arguments:
```bash
python train_model.py --epochs 2500 --lr 0.002 --device cuda
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

## Quick Start

```bash
cd /home/apa2237/TimeSliver/larvar/main

# 1. Train the model
python train_model.py --epochs 2500 --lr 0.002 --device cuda

# 2. Evaluate on test set
python test_model.py --device cuda

# 3. Compute temporal attribution scores
python temporal_attribution.py --split all --device cuda
```

## Notes

- **TensorBoard is optional** - Due to TensorFlow/numpy version conflicts, TensorBoard logging is wrapped in try/except.
- **Simpler architecture** - Single CNN branch, no positional encoding.
- **Higher SAX resolution** - SAX_ALPHABET_SIZE=40 for finer discretization.
- **Model checkpoints** - Saved to `model/best.pth`.
