"""
Utility functions for TimeSliver - Audio Dataset.
Contains data loading, masking, and evaluation helpers.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
from math import ceil

import config
from timesliver import TimeSlicerDataset


def load_data(split, load_sax=True):
    """
    Load data for a given split.

    Args:
        split: 'train', 'valid', or 'test'
        load_sax: Whether to load SAX encoded data

    Returns:
        Dictionary containing loaded data arrays
    """
    x = np.load(config.get_data_path(split, "x"), allow_pickle=True)
    y = np.load(config.get_data_path(split, "y"), allow_pickle=True)

    data = {
        "x": x,
        "y": y,
        "classes": np.argmax(x, axis=2),
        "seq_len": np.array([x.shape[1]] * len(x)),
        "n_samples": x.shape[0],
        "seq_length": x.shape[1],
        "feature_dim": x.shape[-1],
    }

    if load_sax:
        data["sax"] = np.load(config.get_data_path(split, "sax"), allow_pickle=True)

    return data


def create_dataloader(split, batch_size, shuffle=True, load_sax=True):
    """
    Create a DataLoader for a given split.

    Args:
        split: 'train', 'valid', or 'test'
        batch_size: Batch size for the DataLoader
        shuffle: Whether to shuffle the data
        load_sax: Whether to load SAX encoded data

    Returns:
        DataLoader and metadata dictionary
    """
    data = load_data(split, load_sax)

    dataset = TimeSlicerDataset(
        ohe=data["x"],
        sax=data["sax"],
        classes=data["classes"],
        seq_len=data["seq_len"],
        output=data["y"],
        n_samples=data["n_samples"],
    )

    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

    return loader, data


def masking_function(ohe, seq_len, importance, top_percent):
    """
    Mask time points based on importance scores.

    Keeps the top_percent most important time points and zeros out the rest.

    Args:
        ohe: Input data array [samples, seq_len, features]
        seq_len: Sequence lengths array
        importance: Importance scores [samples, seq_len]
        top_percent: Percentage of top important time points to keep (0-100)

    Returns:
        Masked data array
    """
    revised_x = ohe.copy()
    num_samples = ohe.shape[0]

    for k in range(num_samples):
        length = int(seq_len[k])
        # Sort by importance (descending)
        sorted_indices = np.argsort(importance[k, :length])[::-1]
        # Calculate how many tokens to keep
        top_num_tokens = int(ceil(length * top_percent / 100))
        # Get indices to mask (low importance)
        mask_indices = tuple(sorted_indices[top_num_tokens:].tolist())
        # Zero out low importance positions
        revised_x[k, mask_indices, :] = 0

    return revised_x


def load_masked_data(split, top_percent, method="main", sub_method=None):
    """
    Load data with masking applied based on importance scores.

    Args:
        split: 'train', 'valid', or 'test'
        top_percent: Percentage of top important time points to keep
        method: Attribution method ('main', 'transformer', 'captum')
        sub_method: Sub-method for transformer/captum

    Returns:
        Dictionary containing masked data arrays
    """
    data = load_data(split, load_sax=True)

    if top_percent < 100:
        importance = np.load(
            config.get_importance_path(split, method, sub_method), allow_pickle=True
        )
        data["x"] = masking_function(data["x"], data["seq_len"], importance, top_percent)
        data["sax"] = masking_function(
            data["sax"], data["seq_len"], importance, top_percent
        )
        data["classes"] = np.argmax(data["x"], axis=2)

    return data


def create_masked_dataloader(
    split, batch_size, top_percent, method="main", sub_method=None, shuffle=True
):
    """
    Create a DataLoader with masked data.

    Args:
        split: 'train', 'valid', or 'test'
        batch_size: Batch size for the DataLoader
        top_percent: Percentage of top important time points to keep
        method: Attribution method
        sub_method: Sub-method for transformer/captum
        shuffle: Whether to shuffle the data

    Returns:
        DataLoader and metadata dictionary
    """
    data = load_masked_data(split, top_percent, method, sub_method)

    dataset = TimeSlicerDataset(
        ohe=data["x"],
        sax=data["sax"],
        classes=data["classes"],
        seq_len=data["seq_len"],
        output=data["y"],
        n_samples=data["n_samples"],
    )

    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

    return loader, data


def evaluate_model(model, dataloader, device, num_samples):
    """
    Evaluate a model on a dataset.

    Args:
        model: The model to evaluate
        dataloader: DataLoader with evaluation data
        device: Device to run evaluation on
        num_samples: Total number of samples in dataset

    Returns:
        Dictionary with accuracy and confusion matrix
    """
    model.eval()
    predicted_labels = torch.zeros((num_samples, 1))
    actual_labels = torch.zeros((num_samples, 1))
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            x, sax, classes, seq_len, actual = batch
            x = x.to(device)
            sax = sax.to(device)
            seq_len = seq_len.to(device).float()

            # Forward pass
            logits = model(x, sax, seq_len)
            probs = nn.Softmax(dim=1)(logits)
            preds = torch.argmax(probs, dim=1)

            batch_size = preds.size(0)
            predicted_labels[count : count + batch_size, 0] = preds
            actual_labels[count : count + batch_size, 0] = actual
            count += batch_size

    predicted_labels = predicted_labels.cpu().numpy().reshape(-1)
    actual_labels = actual_labels.cpu().numpy().reshape(-1)

    accuracy = accuracy_score(actual_labels, predicted_labels)
    conf_matrix = confusion_matrix(actual_labels, predicted_labels)

    model.train()

    return {"accuracy": accuracy, "confusion_matrix": conf_matrix}


def save_model_config(num_classes, q, d, max_m):
    """Save model configuration for later loading."""
    config_dict = {"num_classes": num_classes, "q": q, "d": d, "max_m": max_m}
    np.save(config.get_config_path("save_dict"), config_dict)


def load_model_config():
    """Load saved model configuration."""
    return np.load(config.get_config_path("save_dict"), allow_pickle=True).item()


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_metrics(results, split_name="Test"):
    """Print evaluation metrics."""
    print(f"\n{split_name} Results:")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Confusion Matrix:")
    print(results["confusion_matrix"])
