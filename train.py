#!/usr/bin/env python3
"""
NICE Flow Implementation with PyTorch
-------------------------------------
This module implements training and evaluation of NICE (Non-linear Independent Components Estimation)
flow-based generative models on image datasets.
"""

import argparse
import logging
import os
import pickle
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.utils.data
import torchvision
from torchvision import transforms

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the NICE model from the nice module
try:
    import nice
except ImportError:
    logger.error("Failed to import 'nice' module. Make sure it's installed or in your PYTHONPATH.")
    raise


def add_uniform_noise(x):
    """
    Add uniform noise for dequantization.
    This is defined outside as a function rather than a lambda for pickling compatibility.

    Args:
        x: Input tensor

    Returns:
        Tensor with added uniform noise
    """
    return x + torch.zeros_like(x).uniform_(0., 1. / 256.)


def train_epoch(
        flow: nice.NICE,
        trainloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device
) -> float:
    """
    Train the flow model for one epoch.

    Args:
        flow: The NICE flow model
        trainloader: DataLoader for training data
        optimizer: The optimizer to use for training
        device: The device to use for training

    Returns:
        Mean loss for the epoch
    """
    loss_epoch = 0.0
    flow.train()  # Set to training mode

    for inputs, _ in trainloader:
        optimizer.zero_grad()
        # Flatten inputs from BxCxHxW to Bx(C*H*W)
        inputs = inputs.view(inputs.shape[0], -1).to(device)

        # Calculate negative log-likelihood
        loss = -flow(inputs).mean()
        loss_epoch += loss.item()

        # Backpropagation and optimization step
        loss.backward()
        optimizer.step()

    # Calculate average loss
    loss_epoch /= len(trainloader)
    return loss_epoch


def evaluate(
        flow: nice.NICE,
        testloader: torch.utils.data.DataLoader,
        sample_shape: List[int],
        device: torch.device,
        output_dir: Path,
        filename_prefix: str,
        epoch: int,
        generate_samples: bool = False
) -> float:
    """
    Evaluate the flow model and optionally generate samples.

    Args:
        flow: The NICE flow model
        testloader: DataLoader for test data
        sample_shape: Shape of output samples [C,H,W]
        device: The device to use for evaluation
        output_dir: Directory to save generated samples
        filename_prefix: Prefix for output filenames
        epoch: Current epoch number
        generate_samples: Whether to generate samples

    Returns:
        Mean negative log-likelihood on test data
    """
    loss_inference = 0.0
    flow.eval()  # Set to inference mode

    with torch.no_grad():
        # Generate samples if requested
        if generate_samples:
            samples_dir = output_dir / "samples"
            samples_dir.mkdir(exist_ok=True, parents=True)

            samples = flow.sample(100).to(device)
            # Normalize samples to [0,1] range for visualization
            a, b = samples.min(), samples.max()
            samples = (samples - a) / (b - a + 1e-10)
            # Reshape samples to image format
            samples = samples.view(-1, *sample_shape)

            # Save generated samples
            sample_path = samples_dir / f"{filename_prefix}_epoch{epoch}.png"
            torchvision.utils.save_image(
                torchvision.utils.make_grid(samples),
                sample_path
            )
            logger.info(f"Generated samples saved to {sample_path}")

        # Calculate test loss
        for xs, _ in testloader:
            xs = xs.view(xs.shape[0], -1).to(device)
            loss = -flow(xs).mean()
            loss_inference += loss.item()

        loss_inference /= len(testloader)
        return loss_inference


def get_data_loaders(
        dataset_name: str,
        batch_size: int,
        data_root: str = "./data",
        num_workers: int = 0
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create data loaders for the specified dataset.

    Args:
        dataset_name: Name of the dataset ('mnist' or 'fashion-mnist')
        batch_size: Batch size for data loaders
        data_root: Root directory for dataset storage
        num_workers: Number of worker processes for data loading

    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Define transforms with dequantization
    # Using a named function instead of lambda for pickling compatibility
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (1.,)),
        transforms.Lambda(add_uniform_noise)  # Use the named function
    ])

    data_path = Path(data_root)
    data_path.mkdir(exist_ok=True, parents=True)

    if dataset_name.lower() == 'mnist':
        trainset = torchvision.datasets.MNIST(
            root=data_path / 'MNIST',
            train=True,
            download=True,
            transform=transform
        )
        testset = torchvision.datasets.MNIST(
            root=data_path / 'MNIST',
            train=False,
            download=True,
            transform=transform
        )
    elif dataset_name.lower() == 'fashion-mnist':
        trainset = torchvision.datasets.FashionMNIST(
            root=data_path / 'FashionMNIST',
            train=True,
            download=True,
            transform=transform
        )
        testset = torchvision.datasets.FashionMNIST(
            root=data_path / 'FashionMNIST',
            train=False,
            download=True,
            transform=transform
        )
    else:
        supported = ['mnist', 'fashion-mnist']
        raise ValueError(f"Dataset '{dataset_name}' not implemented. Supported datasets: {supported}")

    # Create data loaders
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return trainloader, testloader


def build_model_name(args: argparse.Namespace) -> str:
    """
    Construct a descriptive model name from the provided arguments.

    Args:
        args: Command line arguments

    Returns:
        A formatted model name string
    """
    return (
        f"{args.dataset}_"
        f"batch{args.batch_size}_"
        f"coupling{args.coupling}_"
        f"type{args.coupling_type}_"
        f"mid{args.mid_dim}_"
        f"hidden{args.hidden}"
    )


def save_metrics(
        metrics: List[float],
        filename: str,
        output_dir: Path
) -> None:
    """
    Save training/testing metrics to disk.

    Args:
        metrics: List of metric values
        filename: Filename to save metrics
        output_dir: Directory to save metrics
    """
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True, parents=True)

    metrics_path = metrics_dir / filename
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics, f)
    logger.info(f"Saved metrics to {metrics_path}")


def train_and_evaluate(args: argparse.Namespace) -> None:
    """
    Train and evaluate the NICE flow model with the specified parameters.

    Args:
        args: Command line arguments
    """
    # Set up device
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device(f"cuda:{args.gpu}")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    # Set up output directories
    output_dir = Path(args.output_dir)
    model_dir = output_dir / "models"
    model_dir.mkdir(exist_ok=True, parents=True)

    # Get data loaders
    trainloader, testloader = get_data_loaders(
        args.dataset,
        args.batch_size,
        args.data_root,
        args.num_workers
    )

    # Sample shape for reconstruction
    sample_shape = [1, 28, 28]  # [C, H, W] for MNIST datasets

    # Construct model name
    model_name = build_model_name(args)
    model_path = model_dir / f"{model_name}.pt"

    # Initialize model
    input_dim = sample_shape[0] * sample_shape[1] * sample_shape[2]
    logger.info(f"Initializing NICE model with input dimension {input_dim}")

    # Create NICE model
    flow = nice.NICE(
        prior=args.prior,
        coupling=args.coupling,
        coupling_type=args.coupling_type,
        in_out_dim=input_dim,
        mid_dim=args.mid_dim,
        hidden=args.hidden,
        device=device
    ).to(device)

    # Initialize optimizer
    optimizer = torch.optim.Adam(flow.parameters(), lr=args.lr)

    # Initialize learning rate scheduler if specified
    scheduler = None
    if args.lr_schedule:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

    # Initialize metrics lists
    train_losses = []
    test_losses = []

    # Training loop
    logger.info(f"Starting training for {args.epochs} epochs")
    for epoch in range(args.epochs):
        # Train for one epoch
        train_loss = train_epoch(flow, trainloader, optimizer, device)
        train_losses.append(train_loss)

        # Evaluate model
        test_loss = evaluate(
            flow,
            testloader,
            sample_shape,
            device,
            output_dir,
            model_name,
            epoch,
            generate_samples=(epoch % args.sample_interval == 0)
        )
        test_losses.append(test_loss)

        # Update learning rate if using scheduler
        if scheduler is not None:
            scheduler.step(test_loss)

        # Log progress
        logger.info(
            f"Epoch {epoch + 1}/{args.epochs} - "
            f"Train Loss: {train_loss:.4f}, "
            f"Test Loss: {test_loss:.4f}"
        )

        # Save model checkpoint
        if (epoch + 1) % args.checkpoint_interval == 0:
            checkpoint_path = model_dir / f"{model_name}_epoch{epoch + 1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': flow.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss,
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

    # Final evaluation with sample generation
    logger.info("Training complete, performing final evaluation with sample generation")
    final_test_loss = evaluate(
        flow,
        testloader,
        sample_shape,
        device,
        output_dir,
        model_name,
        args.epochs,
        generate_samples=True
    )

    # Save final model
    torch.save(flow.state_dict(), model_path)
    logger.info(f"Saved final model to {model_path}")

    # Save metrics
    save_metrics(
        train_losses,
        f"loss_train_{args.dataset}_{args.coupling_type}.pkl",
        output_dir
    )
    save_metrics(
        test_losses,
        f"loss_test_{args.dataset}_{args.coupling_type}.pkl",
        output_dir
    )

    logger.info(f"Final test loss: {final_test_loss:.4f}")
    logger.info("Training and evaluation complete!")


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description='NICE Flow Model Training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Dataset parameters
    parser.add_argument(
        '--dataset',
        type=str,
        default='mnist',
        choices=['mnist', 'fashion-mnist'],
        help='Dataset to model'
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default='./data',
        help='Root directory for dataset storage'
    )

    # Model parameters
    parser.add_argument(
        '--prior',
        type=str,
        default='logistic',
        choices=['logistic', 'gaussian'],
        help='Latent distribution'
    )
    parser.add_argument(
        '--coupling',
        type=int,
        default=4,
        help='Number of coupling layers'
    )
    parser.add_argument(
        '--coupling-type',
        type=str,
        default='additive',
        choices=['additive', 'affine'],
        help='Type of coupling layers'
    )
    parser.add_argument(
        '--mid-dim',
        type=int,
        default=1000,
        help='Dimension of hidden layers in coupling networks'
    )
    parser.add_argument(
        '--hidden',
        type=int,
        default=5,
        help='Number of hidden layers in coupling networks'
    )

    # Training parameters
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Batch size for training and evaluation'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Initial learning rate'
    )
    parser.add_argument(
        '--lr-schedule',
        action='store_true',
        help='Use learning rate scheduler'
    )

    # Hardware parameters
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        help='Disable CUDA even if available'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU device ID to use'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=0,
        help='Number of worker processes for data loading (0 for main process only)'
    )

    # Output parameters
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output',
        help='Directory for output files'
    )
    parser.add_argument(
        '--sample-interval',
        type=int,
        default=10,
        help='Interval (in epochs) for generating samples'
    )
    parser.add_argument(
        '--checkpoint-interval',
        type=int,
        default=10,
        help='Interval (in epochs) for saving model checkpoints'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=64,
        help='Number of samples to generate'
    )

    return parser.parse_args()


def main():
    """Main entry point of the program."""
    args = parse_arguments()

    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Configure file logging
    log_path = output_dir / f"{args.dataset}_{args.coupling_type}.log"
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    # Log configuration
    logger.info("NICE Flow training started with configuration:")
    for arg, value in sorted(vars(args).items()):
        logger.info(f"  {arg}: {value}")

    # Train and evaluate model
    try:
        train_and_evaluate(args)
    except Exception as e:
        logger.exception(f"Error during training: {e}")
        raise


if __name__ == '__main__':
    main()