"""
Calculate temporal attribution scores for SeqComb-MV samples.

This script computes per-time-point importance scores that identify
which parts of the signal are most relevant for the model's predictions.

Usage:
    python temporal_attribution.py --split train
    python temporal_attribution.py --split test --device cuda:0
    python temporal_attribution.py --split all  # Process all splits
"""
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
import utils
from timesliver import TimeSlicerDataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Calculate temporal attribution scores for SeqComb-MV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "valid", "test", "all"],
        help="Data split to calculate attribution for. Use 'all' for all splits.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda, cuda:0, cpu). Auto-detected if not specified.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model checkpoint. Uses default best.pth if not specified.",
    )
    return parser.parse_args()


def load_model(model_path, device):
    """Load the trained model for attribution."""
    model = torch.load(model_path, map_location=device)
    model = model.to(device)
    model.eval()

    print(f"Model loaded with {utils.count_parameters(model):,} parameters")

    return model


def calculate_attribution(model, dataloader, device, num_samples, max_seq_len):
    """
    Calculate temporal attribution scores for a dataset.

    Args:
        model: The trained model
        dataloader: DataLoader with the data
        device: Device to run on
        num_samples: Total number of samples
        max_seq_len: Maximum sequence length

    Returns:
        Numpy array of importance scores [num_samples, max_seq_len]
    """
    init_lr = np.load(config.get_config_path("init_lr"), allow_pickle=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(init_lr))

    store_importance = torch.zeros((num_samples, max_seq_len)).to(device)
    count = 0

    for batch_idx, (x, sax, classes, seq_len, labels) in enumerate(dataloader):
        x = x.to(device)
        sax = sax.to(device)
        seq_len = seq_len.to(device).float()
        labels = labels.to(device)
        batch_size = len(labels)

        # Forward pass with gradient hooks
        logits, cam, initial_cam, _ = model.forward_motif_importance(x, sax, seq_len)

        # Compute loss and backpropagate to get gradients
        logits = nn.Softmax(dim=-1)(logits)
        logits = logits[:, labels].reshape((-1,))
        target = torch.zeros(logits.shape).to(device)
        loss = criterion(logits, target)

        optimizer.zero_grad()
        loss.backward()

        # Get gradient tensors
        cam = cam[0]
        initial_cam = initial_cam[0]
        print(f"Size of the gradient: {cam.size()}")
        print(f"Size of the initial_cam: {initial_cam.size()}")

        # Normalize gradients per sample
        for prot in range(cam.size(0)):
            cam[prot, ...] = cam[prot, ...] / (abs(torch.max(cam[prot, ...])) + 1e-18)

        # Calculate motif-level importance
        for m_i in range(cam.size(-1)):
            mo_level_imp = model.calculate_motif_level(
                cam[..., m_i], m_i + 1, initial_cam, None
            )

            kernel_size = max_seq_len - mo_level_imp.size(-1) + 1
            store_importance[count : count + batch_size, ...] += model.assigning_importance(
                mo_level_imp, kernel_size, max_seq_len
            )

        count += batch_size

        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed {count}/{num_samples} samples")

    # Convert to numpy
    with torch.no_grad():
        importance = store_importance.cpu().numpy()

    return importance


def process_split(split, model, device, batch_size):
    """
    Process a single data split.

    Args:
        split: 'train', 'valid', or 'test'
        model: The trained model
        device: Device to run on
        batch_size: Batch size for processing
    """
    print(f"\nProcessing {split} split...")

    # Load data
    data = utils.load_data(split, load_sax=True)

    dataset = TimeSlicerDataset(
        ohe=data["x"],
        sax=data["sax"],
        classes=data["classes"],
        seq_len=data["seq_len"],
        output=data["y"],
        n_samples=data["n_samples"],
    )

    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    print(f"  Samples: {data['n_samples']}")
    print(f"  Sequence length: {data['seq_length']}")

    # Calculate attribution
    importance = calculate_attribution(
        model, dataloader, device, data["n_samples"], data["seq_length"]
    )

    # Save results
    output_path = config.get_importance_path(split)
    np.save(output_path, importance)
    print(f"  Saved to: {output_path}")

    return importance


def main():
    """Main entry point."""
    args = parse_args()

    device = config.get_device(args.device)
    print(f"Using device: {device}")

    # Load model
    model_path = args.model_path or config.get_model_path("best")
    print(f"Loading model from: {model_path}")
    model = load_model(model_path, device)

    # Determine which splits to process
    if args.split == "all":
        splits = ["train", "valid", "test"]
    else:
        splits = [args.split]

    # Process each split
    start_time = time.time()

    for split in splits:
        process_split(split, model, device, args.batch_size)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
