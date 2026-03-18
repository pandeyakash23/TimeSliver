# TimeSliver - FordA Dataset

An interpretable time series classification pipeline using CNN-based motif extraction with temporal attribution capabilities.

---

## Directory Structure

```
fordA/
├── README.md                   # Project overview
├── claude.md                   # This file - detailed documentation
│
├── data/                       # Data preparation and storage
│   ├── sax.py                  # SAX tokenization utility
│   ├── run_data_prep.ipynb     # Data preparation notebook
│   ├── x_train.npy             # Training inputs (one-hot encoded)
│   ├── x_valid.npy             # Validation inputs
│   ├── x_test.npy              # Test inputs
│   ├── y_train.npy             # Training labels
│   ├── y_valid.npy             # Validation labels
│   ├── y_test.npy              # Test labels
│   ├── sax_train.npy           # SAX encoded training data
│   ├── sax_valid.npy           # SAX encoded validation data
│   └── sax_test.npy            # SAX encoded test data
│
└── main/                       # Core model and training code
    ├── config.py               # Centralized configuration
    ├── utils.py                # Shared utility functions
    ├── timesliver.py           # Network architecture
    ├── train_model.py          # Baseline model training
    ├── test_model.py           # Model evaluation
    ├── temporal_attribution.py # Calculate importance scores
    ├── train_with_masking.py   # Train with masked time points
    ├── test_with_masking.py    # Test masked model
    │
    └── model/                  # Saved models and configs
        ├── best.pth            # Best baseline model weights
        ├── best_masking.pth    # Best masked model (full object)
        ├── save_dict.npy       # Model architecture config
        ├── init_lr.npy         # Initial learning rate
        ├── categorical_size.npy # Input feature dimension
        ├── importance_train.npy # Attribution scores (train)
        ├── importance_valid.npy # Attribution scores (valid)
        ├── importance_test.npy  # Attribution scores (test)
        ├── top_per.npy         # Masking top percentage
        └── which_imp.npy       # Attribution method used
```

---

## File Descriptions

### `data/sax.py`
SAX (Symbolic Aggregate approXimation) tokenization utility.

**Function:** `sax_tokenizer(time_series, alphabet_size=10, word_length=1)`
- Normalizes time series via z-score
- Uses Gaussian quantile breakpoints to discretize values
- Maps continuous values to discrete symbols (0 to alphabet_size-1)

### `data/run_data_prep.ipynb`
Jupyter notebook for data preparation:
1. Loads raw FordA UCR dataset
2. Splits data: 60% train, 20% valid, 20% test (stratified)
3. Applies SAX tokenization (alphabet_size=70, word_length=1)
4. One-hot encodes SAX symbols
5. Saves `.npy` files for x, y, and sax arrays

---

### `main/config.py`
Centralized configuration module with all hyperparameters and paths.

**Constants:**
| Category | Parameter | Default |
|----------|-----------|---------|
| Data | `SEQUENCE_LENGTH` | 500 |
| Data | `SAX_ALPHABET_SIZE` | 70 |
| Data | `NUM_CLASSES` | 2 |
| Model | `D_OUT` | 36 |
| Model | `MAX_M` | 1 |
| Training | `TRAIN_BATCH_SIZE` | 512 |
| Training | `TRAIN_LR` | 0.001 |
| Training | `TRAIN_EPOCHS` | 5000 |
| Masking | `MASK_TRAIN_LR` | 0.001 |
| Masking | `MASK_TRAIN_EPOCHS` | 2500 |
| Testing | `TEST_BATCH_SIZE` | 256 |

**Utility Functions:**
- `get_device(device_arg)` - Auto-detect CUDA/CPU
- `get_data_path(split, data_type)` - Path to data files
- `get_model_path(model_type)` - Path to model checkpoints
- `get_importance_path(split, method, sub_method)` - Path to attribution scores
- `get_config_path(name)` - Path to config files

---

### `main/utils.py`
Shared utility functions for data loading, masking, and evaluation.

**Data Loading:**
- `load_data(split, load_sax=True)` - Load arrays for a split
- `create_dataloader(split, batch_size, shuffle, load_sax)` - Create PyTorch DataLoader

**Masking:**
- `masking_function(ohe, seq_len, importance, top_percent)` - Zero out low-importance time points
- `load_masked_data(split, top_percent, method, sub_method)` - Load pre-masked data
- `create_masked_dataloader(...)` - DataLoader with masking applied

**Evaluation:**
- `evaluate_model(model, dataloader, device, num_samples)` - Returns accuracy + confusion matrix
- `print_metrics(results, split_name)` - Pretty-print results

**Model Config:**
- `save_model_config(num_classes, q, d, max_m)` - Save architecture params
- `load_model_config()` - Load saved params
- `count_parameters(model)` - Count trainable parameters

---

### `main/timesliver.py`
Core network architecture.

**Classes:**

`TimeSlicerDataset(Dataset)` - PyTorch Dataset
- Inputs: ohe, sax, classes, seq_len, output, n_samples
- Returns tuples of (ohe, sax, classes, seq_len, label)

`PositionalEncoding(nn.Module)` - Sinusoidal position encoding
- Adds temporal position information to embeddings

`TimeSliverNetwork(nn.Module)` - Main classification network
- **Architecture:**
  ```
  Input [batch, 500, 70]
      ↓
  Linear Projection → Positional Encoding
      ↓
  CNN Stack: Conv1d(70→16→32→36, kernel=4)
      ↓
  SAX Pooling: AvgPool1d (3 layers, kernel=4)
      ↓
  Motif-Document Interaction Matrix
      ↓
  AvgPool2d Reduction → Linear(18*35, num_classes)
  ```

- **Key Methods:**
  - `forward(x, sax, seq_len)` - Standard classification
  - `forward_motif_importance(x, sax, seq_len)` - Forward with gradient hooks for attribution
  - `calculate_motif_level(dp, m_i, initial_cam)` - Compute motif importance from gradients
  - `assigning_importance(mo_level, kernel_size, unwrapped_len)` - Map motif importance to sequence positions

---

### `main/train_model.py`
Baseline model training script.

**Usage:**
```bash
python train_model.py --epochs 5000 --lr 0.001 --device cuda
python train_model.py --batch-size 256 --d-out 48
python train_model.py --no-tensorboard
```

**CLI Arguments:**
| Arg | Default | Description |
|-----|---------|-------------|
| `--epochs` | 5000 | Training epochs |
| `--lr` | 0.001 | Learning rate |
| `--batch-size` | 512 | Batch size |
| `--device` | auto | cuda/cpu |
| `--d-out` | 36 | CNN output dim |
| `--max-m` | 1 | Motif scales |
| `--tensorboard` | true | Enable logging |

**Outputs:** `model/best.pth` (state_dict)

---

### `main/test_model.py`
Model evaluation script.

**Usage:**
```bash
python test_model.py --device cuda
python test_model.py --split valid
python test_model.py --model-path ./model/checkpoint.pth
```

**CLI Arguments:**
| Arg | Default | Description |
|-----|---------|-------------|
| `--split` | test | train/valid/test |
| `--batch-size` | 256 | Batch size |
| `--device` | auto | cuda/cpu |
| `--model-path` | best.pth | Model checkpoint |

**Output:** Accuracy and confusion matrix

---

### `main/temporal_attribution.py`
Calculate per-time-point importance scores.

**Usage:**
```bash
python temporal_attribution.py --split all --device cuda
python temporal_attribution.py --split test --batch-size 512
```

**CLI Arguments:**
| Arg | Default | Description |
|-----|---------|-------------|
| `--split` | all | train/valid/test/all |
| `--batch-size` | 1024 | Batch size |
| `--device` | auto | cuda/cpu |
| `--model-path` | best.pth | Model checkpoint |

**Algorithm:**
1. Forward pass with gradient hooks capturing CNN motif outputs
2. Backpropagate loss to get gradients
3. `calculate_motif_level()`: Compute importance via `ReLU(sign × d_comp × q_comp)`
4. `assigning_importance()`: Map reduced-length importance back to original 500 positions
5. Apply edge weighting for convolution boundary effects

**Outputs:** `model/importance_{train,valid,test}.npy` (shape: samples × 500)

---

### `main/train_with_masking.py`
Train model using only top-k% important time points.

**Usage:**
```bash
python train_with_masking.py --top-percent 20 --device cuda
python train_with_masking.py --top-percent 50 --method captum --sub-method ig
```

**CLI Arguments:**
| Arg | Default | Description |
|-----|---------|-------------|
| `--top-percent` | required | % of time points to keep (1-100) |
| `--epochs` | 2500 | Training epochs |
| `--lr` | 0.001 | Learning rate |
| `--batch-size` | 512 | Batch size |
| `--device` | auto | cuda/cpu |
| `--method` | main | main/transformer/captum |
| `--sub-method` | None | attn/grad (transformer) or ig/gs/dl/dlshap (captum) |

**Process:**
1. Load importance scores for specified method
2. Sort time points by importance (descending)
3. Zero out bottom (100-top_percent)% positions in OHE and SAX
4. Train new model on masked data

**Outputs:** `model/best_masking.pth` (full model object)

---

### `main/test_with_masking.py`
Evaluate masked model.

**Usage:**
```bash
python test_with_masking.py --device cuda
python test_with_masking.py --top-percent 30  # Override saved config
python test_with_masking.py --method captum --sub-method ig
```

**CLI Arguments:**
| Arg | Default | Description |
|-----|---------|-------------|
| `--split` | test | train/valid/test |
| `--top-percent` | saved | Override masking % |
| `--batch-size` | 256 | Batch size |
| `--device` | auto | cuda/cpu |
| `--method` | saved | Attribution method |
| `--sub-method` | saved | Sub-method |
| `--model-path` | best_masking.pth | Model path |

---

## Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA PREPARATION                            │
│  run_data_prep.ipynb                                                │
│  Raw FordA → SAX Encoding → One-Hot → Train/Valid/Test Split        │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│                         BASELINE TRAINING                           │
│  python train_model.py --epochs 5000 --lr 0.001                     │
│  → model/best.pth                                                   │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│                         BASELINE TESTING                            │
│  python test_model.py --split test                                  │
│  → Accuracy, Confusion Matrix                                       │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      TEMPORAL ATTRIBUTION                           │
│  python temporal_attribution.py --split all                         │
│  → model/importance_{train,valid,test}.npy                          │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      MASKED MODEL TRAINING                          │
│  python train_with_masking.py --top-percent 20                      │
│  → model/best_masking.pth                                           │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      MASKED MODEL TESTING                           │
│  python test_with_masking.py --split test                           │
│  → Accuracy, Confusion Matrix (masked)                              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# 1. Prepare data (run notebook or ensure .npy files exist)

# 2. Train baseline model
cd main
python train_model.py --epochs 5000 --device cuda

# 3. Test baseline
python test_model.py

# 4. Compute temporal attribution
python temporal_attribution.py --split all

# 5. Train with masking (keep top 20% important time points)
python train_with_masking.py --top-percent 20

# 6. Test masked model
python test_with_masking.py
```

---

## Key Design Decisions

1. **Centralized Config (`config.py`)**: All hyperparameters in one place, overridable via CLI
2. **Shared Utilities (`utils.py`)**: DRY principle - data loading, masking, evaluation extracted
3. **CLI Arguments**: All scripts support `--device`, `--batch-size`, etc. for flexibility
4. **Model Saving**:
   - Baseline: `state_dict` only (smaller, requires architecture rebuild)
   - Masked: Full model object (self-contained, larger)
5. **Attribution Methods**: Supports TimeSliver native, transformer attention, and Captum methods
