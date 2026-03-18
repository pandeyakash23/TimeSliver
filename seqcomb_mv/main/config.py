"""
Configuration settings for TimeSliver on SeqComb-MV (Multivariate) dataset.
"""
import os
import torch

# =============================================================================
# Dataset Parameters
# =============================================================================
NUM_CLASSES = 4
SAX_ALPHABET_SIZE = 20

# =============================================================================
# Model Architecture
# =============================================================================
D_OUT = 36  # CNN output channels
MAX_M = 1   # Number of motif scale groups (uses 2 CNN branches internally)

# Architecture specifics:
# - 2 CNN branches (4-size and 7-size motifs)
# - Uses positional encoding
# - AvgPool3d reduction
# - Linear input: 720 * max_m

# =============================================================================
# Training Defaults
# =============================================================================
TRAIN_EPOCHS = 2500
TRAIN_LR = 0.002
TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 256

# =============================================================================
# Paths
# =============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "main", "model")

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)


def get_data_path(split, data_type="x"):
    """
    Get path to data file.

    Args:
        split: 'train', 'valid', or 'test'
        data_type: 'x', 'y', or 'sax'

    Returns:
        Path to the data file
    """
    filename = f"{data_type}_{split}.npy"
    return os.path.join(DATA_DIR, filename)


def get_model_path(name="best"):
    """
    Get path to model checkpoint.

    Args:
        name: Model name (e.g., 'best', 'best_masking')

    Returns:
        Path to model file
    """
    return os.path.join(MODEL_DIR, f"{name}.pth")


def get_config_path(name):
    """
    Get path to config file.

    Args:
        name: Config name (e.g., 'save_dict', 'init_lr')

    Returns:
        Path to config file
    """
    return os.path.join(MODEL_DIR, f"{name}.npy")


def get_importance_path(split):
    """
    Get path to importance scores file.

    Args:
        split: 'train', 'valid', or 'test'

    Returns:
        Path to importance file
    """
    return os.path.join(MODEL_DIR, f"importance_{split}.npy")


def get_device(device_str=None):
    """
    Get the device to use for training/inference.

    Args:
        device_str: Optional device string ('cuda', 'cuda:0', 'cpu')

    Returns:
        torch.device object
    """
    if device_str:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
