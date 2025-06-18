"""Finetune the Decima model."""

import os
import click
import anndata
import wandb
from decima.model.lightning import LightningModel
from decima.data.dataset import HDF5Dataset


@click.command()
@click.option("--name", required=True, help="Project name")
@click.option("--dir", required=True, help="Data directory path")
@click.option("--lr", default=0.001, type=float, help="Learning rate")
@click.option("--weight", required=True, type=float, help="Weight parameter")
@click.option("--grad", required=True, type=int, help="Gradient accumulation steps")
@click.option("--replicate", default=0, type=int, help="Replication number")
@click.option("--bs", default=4, type=int, help="Batch size")
def cli_finetune(name, dir, lr, weight, grad, replicate, bs):
    """Finetune the Decima model."""
    wandb.login(host="https://genentech.wandb.io")
    run = wandb.init(project="decima", dir=name, name=name)

    matrix_file = os.path.join(dir, "aggregated.h5ad")
    h5_file = os.path.join(dir, "data.h5")
    print(f"Data paths: {matrix_file}, {h5_file}")

    print("Reading anndata")
    ad = anndata.read_h5ad(matrix_file)

    print("Making dataset objects")
    train_dataset = HDF5Dataset(
        h5_file=h5_file,
        ad=ad,
        key="train",
        max_seq_shift=5000,
        augment_mode="random",
        seed=0,
    )
    val_dataset = HDF5Dataset(h5_file=h5_file, ad=ad, key="val", max_seq_shift=0)

    train_params = {
        "optimizer": "adam",
        "batch_size": bs,
        "num_workers": 16,
        "devices": 0,
        "logger": "wandb",
        "save_dir": dir,
        "max_epochs": 15,
        "lr": lr,
        "total_weight": weight,
        "accumulate_grad_batches": grad,
        "loss": "poisson_multinomial",
        "pairs": ad.uns["disease_pairs"].values,
    }
    model_params = {
        "n_tasks": ad.shape[0],
        "replicate": replicate,
    }
    print(f"train_params: {train_params}")
    print(f"model_params: {model_params}")

    print("Initializing model")
    model = LightningModel(model_params=model_params, train_params=train_params)

    print("Training")
    model.train_on_dataset(train_dataset, val_dataset)
    train_dataset.close()
    val_dataset.close()
    run.finish()
