"""
Finetune the Decima model.

This module contains the CLI for finetuning the Decima model.

`decima finetune` is the main command for finetuning the Decima model.

It includes subcommands for:
- Finetuning the Decima model. `finetune`
"""

import logging
import click
import anndata
import wandb
from decima.model.lightning import LightningModel
from decima.data.dataset import HDF5Dataset


@click.command()
@click.option("--name", required=True, help="Name of the run.")
@click.option(
    "--model",
    default="0",
    type=str,
    help="Model path or replication number. If a path is provided, the model will be loaded from the path. If a replication number is provided, the model will be loaded from the replication number.",
)
@click.option(
    "--device",
    type=str,
    default="0",
    help="Device to use. Default: 0",
)
@click.option("--matrix-file", required=True, help="Matrix file path.")
@click.option("--h5-file", required=True, help="H5 file path.")
@click.option("--outdir", required=True, help="Output directory path to save model checkpoints.")
@click.option("--learning-rate", default=0.001, type=float, help="Learning rate.")
@click.option("--loss-total-weight", required=True, type=float, help="Total weight parameter for the loss function.")
@click.option("--gradient-accumulation", required=True, type=int, help="Gradient accumulation steps.")
@click.option("--batch-size", default=1, type=int, help="Batch size.")
@click.option("--max-seq-shift", default=5000, type=int, help="Shift augmentation.")
@click.option("--gradient-clipping", default=0.0, type=float, help="Gradient clipping.")
@click.option("--save-top-k", default=1, type=int, help="Number of checkpoints to save.")
@click.option("--epochs", default=1, type=int, help="Number of epochs.")
@click.option("--logger", default="wandb", type=str, help="Logger.")
@click.option("--num-workers", default=16, type=int, help="Number of workers.")
@click.option("--seed", default=0, type=int, help="Random seed.")
def cli_finetune(
    name,
    model,
    device,
    matrix_file,
    h5_file,
    outdir,
    learning_rate,
    loss_total_weight,
    gradient_accumulation,
    batch_size,
    max_seq_shift,
    gradient_clipping,
    save_top_k,
    epochs,
    logger,
    num_workers,
    seed,
):
    """Finetune the Decima model.

    Args:
        name: Name of the run for logging and checkpointing
        model: Borzoi model path or replication number (0-3)
        device: Device to use for training. Default: "0"
        matrix_file: Path to the matrix file containing training data
        h5_file: Path to the H5 file containing sequences
        outdir: Output directory path to save model checkpoints
        learning_rate: Learning rate for training. Default: 0.001
        loss_total_weight: Total weight parameter for the loss function
        gradient_accumulation: Number of gradient accumulation steps
        batch_size: Batch size for training. Default: 1
        max_seq_shift: Maximum sequence shift for data augmentation. Default: 5000
        gradient_clipping: Gradient clipping value. Default: 0.0 (disabled)
        save_top_k: Number of best checkpoints to save. Default: 1
        epochs: Number of training epochs. Default: 1
        logger: Logger type to use. Default: "wandb"
        num_workers: Number of data loading workers. Default: 16
        seed: Random seed for reproducibility. Default: 0
    """
    train_logger = logger
    logger = logging.getLogger("decima")
    logger.info(f"Data paths: matrix_file={matrix_file}, h5_file={h5_file}")
    logger.info("Reading anndata")
    ad = anndata.read_h5ad(matrix_file)

    logger.info("Making dataset objects")
    train_dataset = HDF5Dataset(
        h5_file=h5_file,
        ad=ad,
        key="train",
        max_seq_shift=max_seq_shift,
        augment_mode="random",
        seed=seed,
    )
    val_dataset = HDF5Dataset(h5_file=h5_file, ad=ad, key="val", max_seq_shift=0)

    if isinstance(device, str) and device.isdigit():
        device = int(device)

    train_params = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "devices": device,
        "logger": train_logger,
        "save_dir": outdir,
        "max_epochs": epochs,
        "lr": learning_rate,
        "total_weight": loss_total_weight,
        "accumulate_grad_batches": gradient_accumulation,
        "loss": "poisson_multinomial",
        "clip": gradient_clipping,
        "save_top_k": save_top_k,
        "pin_memory": True,
    }
    model_params = {
        "n_tasks": ad.shape[0],
        "init_borzoi": True,
        "replicate": model,
    }
    logger.info(f"train_params: {train_params}")
    logger.info(f"model_params: {model_params}")

    logger.info("Initializing model")
    model = LightningModel(name=name, model_params=model_params, train_params=train_params)

    if train_logger == "wandb":
        logger.info("Connecting to wandb.")
        wandb.login(host="https://genentech.wandb.io", anonymous="never")
        run = wandb.init(project="decima", dir=name, name=name)

    logger.info("Training")
    model.train_on_dataset(train_dataset, val_dataset)
    train_dataset.close()
    val_dataset.close()
    if logger == "wandb":
        run.finish()
