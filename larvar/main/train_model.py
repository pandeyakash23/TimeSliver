"""
Train the TimeSliver model on Larvar dataset.

Usage:
    python train_model.py --epochs 2500 --lr 0.002 --device cuda
"""
import argparse
import time
import numpy as np
import torch
import torch.nn as nn

# TensorBoard is optional due to potential TensorFlow/numpy conflicts
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except (ImportError, AttributeError):
    HAS_TENSORBOARD = False
    print("Warning: TensorBoard not available, logging disabled")

import config
import utils
from timesliver import TimeSliverNetwork


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train TimeSliver model for Larvar classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.TRAIN_EPOCHS,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=config.TRAIN_LR,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.TRAIN_BATCH_SIZE,
        help="Batch size for training",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda, cuda:0, cpu). Auto-detected if not specified.",
    )
    parser.add_argument(
        "--d-out",
        type=int,
        default=config.D_OUT,
        help="Output dimension for CNN layers (d)",
    )
    parser.add_argument(
        "--max-m",
        type=int,
        default=config.MAX_M,
        help="Number of motif scales",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        default=True,
        help="Enable TensorBoard logging",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_false",
        dest="tensorboard",
        help="Disable TensorBoard logging",
    )
    return parser.parse_args()


def create_model(d_model, d_out, max_m, device):
    """Create and initialize the TimeSliver model."""
    model = TimeSliverNetwork(
        num_classes=config.NUM_CLASSES,
        d_model=d_model,
        d_out=d_out,
        max_m=max_m,
        device=device,
    ).to(device)

    print(f"Model created with {utils.count_parameters(model):,} trainable parameters")

    criterion = nn.CrossEntropyLoss()

    return model, criterion


def train(args):
    """Main training loop."""
    device = config.get_device(args.device)
    print(f"Training on device: {device}")

    # Load data
    print("Loading training data...")
    train_loader, train_data = utils.create_dataloader(
        "train", args.batch_size, shuffle=True
    )
    valid_loader, valid_data = utils.create_dataloader(
        "valid", args.batch_size, shuffle=False
    )

    print(f"Training samples: {train_data['n_samples']}")
    print(f"Validation samples: {valid_data['n_samples']}")

    # Get feature dimension from data
    d_model = train_data["feature_dim"]
    print(f"Feature dimension (d_model): {d_model}")

    # Save configuration
    utils.save_model_config(config.NUM_CLASSES, d_model, args.d_out, args.max_m)
    np.save(config.get_config_path("init_lr"), args.lr)
    np.save(config.get_config_path("categorical_size"), d_model)

    # Create model
    model, criterion = create_model(d_model, args.d_out, args.max_m, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # TensorBoard
    writer = None
    if args.tensorboard and HAS_TENSORBOARD:
        writer = SummaryWriter(comment=f"TimeSliver_Larvar_lr{args.lr}_epochs{args.epochs}")

    # Training loop
    best_valid_acc = 0.0
    print(f"\nStarting training for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        model.train()
        avg_loss = 0.0

        for i, (x, sax, classes, seq_len, labels) in enumerate(train_loader):
            x = x.to(device)
            sax = sax.to(device)
            seq_len = seq_len.to(device).float()
            labels = labels.to(device)

            # Forward pass
            logits = model(x, sax, seq_len)
            loss = criterion(logits, labels)
            avg_loss = (avg_loss * i + loss.item()) / (i + 1)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        results = utils.evaluate_model(
            model, valid_loader, device, valid_data["n_samples"]
        )
        valid_acc = results["accuracy"]

        # Logging
        if writer:
            writer.add_scalar("Loss/train", avg_loss, epoch + 1)
            writer.add_scalar("Accuracy/valid", valid_acc, epoch + 1)

        # Save best model
        if valid_acc >= best_valid_acc:
            torch.save(model, config.get_model_path("best"))
            best_valid_acc = valid_acc

        # Print progress every 100 epochs
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:5d}/{args.epochs} | "
                f"Loss: {avg_loss:.4f} | "
                f"Valid Acc: {valid_acc:.4f} | "
                f"Best: {best_valid_acc:.4f}"
            )

    if writer:
        writer.close()

    print(f"\nTraining complete! Best validation accuracy: {best_valid_acc:.4f}")
    print(f"Model saved to: {config.get_model_path('best')}")


def main():
    """Main entry point."""
    args = parse_args()

    start_time = time.time()
    train(args)
    elapsed = time.time() - start_time

    print(f"Total training time: {elapsed:.2f}s ({elapsed/60:.2f}min)")


if __name__ == "__main__":
    main()
