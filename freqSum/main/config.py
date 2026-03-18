"""
Centralized configuration for TimeSliver - Synthetic Data.
All hyperparameters and paths can be modified here or overridden via CLI.
"""
import os
import torch

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# =============================================================================
# DATA SETTINGS
# =============================================================================
SAX_ALPHABET_SIZE = 90
NUM_CLASSES = 2  # Binary classification

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================
D_OUT = 36      # Output dimension for CNN (d)
MAX_M = 1       # Number of motif scales

# =============================================================================
# TRAINING SETTINGS
# =============================================================================
TRAIN_BATCH_SIZE = 512
TRAIN_LR = 0.001
TRAIN_EPOCHS = 2500

# Masking training
MASK_TRAIN_LR = 0.001
MASK_TRAIN_EPOCHS = 2500

# =============================================================================
# TESTING SETTINGS
# =============================================================================
TEST_BATCH_SIZE = 256

# =============================================================================
# DEVICE SETTINGS
# =============================================================================
def get_device(device_arg=None):
    """Get the appropriate device for training/inference."""
    if device_arg:
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

# =============================================================================
# FILE PATHS
# =============================================================================
def get_data_path(split, data_type="x"):
    """
    Get path to data file.

    Args:
        split: 'train', 'valid', or 'test'
        data_type: 'x' for input, 'y' for labels, 'sax' for SAX encoded

    Returns:
        Full path to the data file
    """
    if data_type == "sax":
        filename = f"sax_{split}.npy"
    else:
        filename = f"{data_type}_{split}.npy"
    return os.path.join(DATA_DIR, filename)

def get_model_path(model_type="best"):
    """
    Get path to model file.

    Args:
        model_type: 'best' for baseline, 'best_masking' for masked model

    Returns:
        Full path to the model file
    """
    if model_type == "best":
        return os.path.join(MODEL_DIR, "best.pth")
    elif model_type == "best_masking":
        return os.path.join(MODEL_DIR, "best_masking.pth")
    else:
        return os.path.join(MODEL_DIR, f"{model_type}.pth")

def get_importance_path(split, method="main", sub_method=None):
    """
    Get path to importance scores file.

    Args:
        split: 'train', 'valid', or 'test'
        method: Attribution method ('main', 'transformer', 'captum')
        sub_method: Sub-method for transformer/captum

    Returns:
        Full path to the importance file
    """
    if method in ["transformer", "captum"] and sub_method:
        filename = f"importance_{split}_{sub_method}.npy"
    else:
        filename = f"importance_{split}.npy"

    if method == "main":
        return os.path.join(MODEL_DIR, filename)
    else:
        return os.path.join(BASE_DIR, method, "model", filename)

def get_config_path(name):
    """Get path to saved configuration file."""
    return os.path.join(MODEL_DIR, f"{name}.npy")
