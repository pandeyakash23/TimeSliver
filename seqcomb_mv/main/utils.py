"""
Utility functions for TimeSliver on SeqComb-MV dataset.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix

import config
from timesliver import TimeSlicerDataset


def load_data(split, load_sax=True):
    """
    Load data for a given split.

    Args:
        split: 'train', 'valid', or 'test'
        load_sax: Whether to load SAX data

    Returns:
        Dictionary containing data arrays and metadata
    """
    x = np.load(config.get_data_path(split, "x"), allow_pickle=True)
    y = np.load(config.get_data_path(split, "y"), allow_pickle=True)

    classes = np.argmax(x, axis=2)
    seq_len = np.array([x.shape[1]] * len(x))

    data = {
        "x": x,
        "y": y,
        "classes": classes,
        "seq_len": seq_len,
        "n_samples": x.shape[0],
        "seq_length": x.shape[1],
        "feature_dim": x.shape[-1],
    }

    if load_sax:
        sax = np.load(config.get_data_path(split, "sax"), allow_pickle=True)
        data["sax"] = sax

    return data


def create_dataloader(split, batch_size, shuffle=True, load_sax=True):
    """
    Create a DataLoader for the given split.

    Args:
        split: 'train', 'valid', or 'test'
        batch_size: Batch size
        shuffle: Whether to shuffle data
        load_sax: Whether to load SAX data

    Returns:
        Tuple of (DataLoader, data_info dict)
    """
    data = load_data(split, load_sax=load_sax)

    dataset = TimeSlicerDataset(
        ohe=data["x"],
        sax=data["sax"],
        classes=data["classes"],
        seq_len=data["seq_len"],
        output=data["y"],
        n_samples=data["n_samples"],
    )

    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader, data


def evaluate_model(model, dataloader, device, n_samples):
    """
    Evaluate model on a dataset.

    Args:
        model: The model to evaluate
        dataloader: DataLoader with evaluation data
        device: Device to run on
        n_samples: Total number of samples

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()

    predicted_labels = torch.zeros((n_samples, 1))
    actual_labels = torch.zeros((n_samples, 1))
    count = 0

    with torch.no_grad():
        for x, sax, classes, seq_len, labels in dataloader:
            x = x.to(device)
            sax = sax.to(device)
            seq_len = seq_len.to(device).float()
            labels = labels.to(device)

            logits = model(x, sax, seq_len)
            probs = nn.Softmax(dim=1)(logits)
            preds = torch.argmax(probs, dim=1)

            batch_size = preds.size(0)
            predicted_labels[count : count + batch_size, 0] = preds.cpu()
            actual_labels[count : count + batch_size, 0] = labels.cpu()
            count += batch_size

    predicted_labels = predicted_labels.numpy().reshape(-1)
    actual_labels = actual_labels.numpy().reshape(-1)

    accuracy = accuracy_score(actual_labels, predicted_labels)
    conf_matrix = confusion_matrix(actual_labels, predicted_labels)

    return {
        "accuracy": accuracy,
        "confusion_matrix": conf_matrix,
        "predictions": predicted_labels,
        "actuals": actual_labels,
    }


def print_metrics(results, split_name="Test"):
    """
    Print evaluation metrics.

    Args:
        results: Dictionary from evaluate_model
        split_name: Name of the split for display
    """
    print(f"\n{split_name} Results:")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Confusion Matrix:")
    print(results["confusion_matrix"])


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_model_config():
    """Load saved model configuration."""
    config_path = config.get_config_path("save_dict")
    return np.load(config_path, allow_pickle=True).item()


def save_model_config(num_classes, q, d, max_m):
    """Save model configuration."""
    save_dict = {"num_classes": num_classes, "q": q, "d": d, "max_m": max_m}
    np.save(config.get_config_path("save_dict"), save_dict)
