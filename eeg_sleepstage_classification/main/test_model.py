"""
Test the trained TimeSliver model on EEG sleep stage data.

Usage:
    python test_model.py --device cuda
    python test_model.py --split valid  # Test on validation set
"""
import argparse
import time
import torch

import config
import utils
from timesliver import TimeSliverNetwork


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test TimeSliver model for EEG sleep stage classification",
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
        "--model-path",
        type=str,
        default=None,
        help="Path to model checkpoint. Uses default best.pth if not specified.",
    )
    return parser.parse_args()


def load_model(model_path, device):
    """
    Load the trained model.

    Args:
        model_path: Path to the model checkpoint
        device: Device to load model on

    Returns:
        Loaded model in eval mode
    """
    # Load model configuration
    model_config = utils.load_model_config()

    # Create model architecture
    model = TimeSliverNetwork(
        num_classes=model_config.get("num_classes", config.NUM_CLASSES),
        d_model=model_config["q"],
        d_out=model_config["d"],
        max_m=model_config["max_m"],
        device=device,
    )

    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    print(f"Model loaded with {utils.count_parameters(model):,} parameters")

    return model


def test(args):
    """
    Run model evaluation.

    Args:
        args: Parsed command line arguments
    """
    device = config.get_device(args.device)
    print(f"Testing on device: {device}")

    # Determine model path
    model_path = args.model_path or config.get_model_path("best")
    print(f"Loading model from: {model_path}")

    # Load model
    model = load_model(model_path, device)

    # Load test data
    print(f"Loading {args.split} data...")
    test_loader, test_data = utils.create_dataloader(
        args.split, args.batch_size, shuffle=False
    )
    print(f"Samples: {test_data['n_samples']}")

    # Evaluate
    print("\nEvaluating...")
    results = utils.evaluate_model(model, test_loader, device, test_data["n_samples"])

    # Print results
    utils.print_metrics(results, split_name=args.split.capitalize())


def main():
    """Main entry point."""
    args = parse_args()

    start_time = time.time()
    test(args)
    elapsed = time.time() - start_time

    print(f"\nTotal time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
