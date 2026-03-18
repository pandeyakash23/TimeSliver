# TimeSliver - SeqComb-UV (Univariate) Classification

This folder contains the TimeSliver implementation for **4-class classification on SeqComb Univariate time series data**.

## Directory Structure

```
seqcomb_uv/
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

## Key Parameters (SeqComb-UV Specific)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `NUM_CLASSES` | 4 | 4-class classification |
| `SAX_ALPHABET_SIZE` | 20 | SAX discretization levels |
| `D_OUT` | 36 | CNN output channels |
| `MAX_M` | 1 | Number of motif scale groups |
| `CNN_BRANCHES` | 2 | Dual branch (4-size and 7-size motifs) |
| `LINEAR_INPUT` | 360 * max_m | Final layer input size (different from MV!) |

## Architecture Details

### Key Difference from SeqComb-MV
- **Linear input size**: 360 * max_m (vs 720 * max_m in MV)
- This is due to different feature dimensions in univariate vs multivariate data

### Model Architecture (SeqComb-UV)
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
│  Positional Encoding                     │
└──────────────────────────────────────────┤
    │                                      │
    ├──────────────┬───────────────────────┤
    │              │                       │
    ▼              ▼                       │
┌─────────┐  ┌─────────┐                   │
│ CNN1    │  │ CNN2    │                   │
│ k=2,2,2 │  │ k=3,3,3 │                   │
│ 4-motif │  │ 7-motif │                   │
└────┬────┘  └────┬────┘                   │
     │            │                        │
     ▼            ▼                        │
┌──────────────────────────────────────────┤
│  Concatenate Branches                    │
└──────────────────────────────────────────┤
    │                                      │
    ▼                                      │
┌──────────────────────────────────────────┤
│  AvgPool3d((2,1,2))                      │
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
- Dataset parameters (NUM_CLASSES=4, SAX_ALPHABET_SIZE=20)
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
- `TimeSliverNetwork` - Main model with dual CNN branches
- `PositionalEncoding` - Sinusoidal positional encoding
- `forward_motif_importance()` - For attribution calculation
- Includes normalization in `calculate_motif_level()`

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
cd /home/apa2237/TimeSliver/seqcomb_uv/main

# 1. Train the model
python train_model.py --epochs 2500 --lr 0.002 --device cuda

# 2. Evaluate on test set
python test_model.py --device cuda

# 3. Compute temporal attribution scores
python temporal_attribution.py --split all --device cuda
```

## Notes

- **TensorBoard is optional** - Due to TensorFlow/numpy version conflicts, TensorBoard logging is wrapped in try/except.
- **Dual CNN branches** - Uses both 4-size and 7-size motifs for multi-scale pattern detection.
- **Different from MV** - Linear layer input is 360*max_m (not 720*max_m).
- **Model checkpoints** - Saved to `model/best.pth`.
