"""
Test the masked TimeSliver model on Synthetic Data.

This script evaluates a model that was trained with masked time points.

Usage:
    python test_with_masking.py --device cuda
    python test_with_masking.py --top-percent 30  # Override masking percentage
"""
import argparse
import time
import numpy as np
import torch
import torch.nn as nn

import config
import utils


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test masked TimeSliver model for Synthetic Data classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "valid", "test"],
        help="Data split to evaluate on",
    )
    parser.add_argument(
        "--top-percent",
        type=int,
        default=None,
        help="Override masking percentage. Uses saved value if not specified.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.TEST_BATCH_SIZE,
        help="Batch size for testing",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda, cuda:0, cpu). Auto-detected if not specified.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Attribution method. Uses saved value if not specified.",
    )
    parser.add_argument(
        "--sub-method",
        type=str,
        default=None,
        help="Sub-method for transformer/captum. Uses saved value if not specified.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model. Uses default best_masking.pth if not specified.",
    )
    return parser.parse_args()


def load_saved_config():
    """Load saved masking configuration from training."""
    config_dict = {}

    try:
        config_dict["top_percent"] = int(
            np.load(config.get_config_path("top_per"), allow_pickle=True)
        )
    except FileNotFoundError:
        config_dict["top_percent"] = 100

    try:
        config_dict["method"] = str(
            np.load(config.get_config_path("which_imp"), allow_pickle=True)
        )
    except FileNotFoundError:
        config_dict["method"] = "main"

    try:
        config_dict["sub_method"] = str(
            np.load(config.get_config_path("sub_method"), allow_pickle=True)
        )
    except FileNotFoundError:
        config_dict["sub_method"] = None

    return config_dict


def load_model(model_path, device):
    """
    Load the trained masked model.

    Args:
        model_path: Path to the model checkpoint
        device: Device to load model on

    Returns:
        Loaded model in eval mode
    """
    model = torch.load(model_path, map_location=device)
    model = model.to(device)
    model.eval()

    print(f"Model loaded with {utils.count_parameters(model):,} parameters")

    return model


def test(args):
    """
    Run model evaluation with masking.

    Args:
        args: Parsed command line arguments
    """
    device = config.get_device(args.device)
    print(f"Testing on device: {device}")

    # Load saved configuration
    saved_config = load_saved_config()

    # Use command line args if provided, otherwise use saved config
    top_percent = args.top_percent or saved_config["top_percent"]
    method = args.method or saved_config["method"]
    sub_method = args.sub_method or saved_config["sub_method"]

    print(f"Masking configuration:")
    print(f"  Top percent: {top_percent}%")
    print(f"  Method: {method}")
    if sub_method:
        print(f"  Sub-method: {sub_method}")

    # Load model
    model_path = args.model_path or config.get_model_path("best_masking")
    print(f"\nLoading model from: {model_path}")
    model = load_model(model_path, device)

    # Load masked test data
    print(f"\nLoading masked {args.split} data...")
    test_loader, test_data = utils.create_masked_dataloader(
        args.split,
        args.batch_size,
        top_percent,
        method=method,
        sub_method=sub_method,
        shuffle=False,
    )
    print(f"Samples: {test_data['n_samples']}")

    # Evaluate
    print("\nEvaluating...")
    results = utils.evaluate_model(model, test_loader, device, test_data["n_samples"])

    # Print results
    utils.print_metrics(results, split_name=f"{args.split.capitalize()} (masked)")


def main():
    """Main entry point."""
    args = parse_args()

    start_time = time.time()
    test(args)
    elapsed = time.time() - start_time

    print(f"\nTotal time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
