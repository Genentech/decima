"""Finetune the Decima model."""

import os
os.environ['WANDB_DISABLE_CODE_AND_METADATA_COLLECTION'] = 'true'
import click
import anndata
import wandb
from decima.model.lightning import LightningModel
from decima.data.dataset import HDF5Dataset
import wandb
wandb.login(host="https://genentech.wandb.io")

@click.command()
@click.option("--name", required=True, help="Project name")
@click.option("--datadir", required=True, help="Data directory path")
@click.option("--outdir", required=True, help="Output directory path")
@click.option("--lr", default=0.001, type=float, help="Learning rate")
@click.option("--weight", required=True, type=float, help="Weight parameter")
@click.option("--grad", required=True, type=int, help="Gradient accumulation steps")
@click.option("--replicate", default=0, type=int, help="Replication number")
@click.option("--bs", default=4, type=int, help="Batch size")
@click.option("--shift", default=5000, type=int, help="Shift augmentation")
@click.option("--optim", default='adam', type=str, help="Optimizer")
@click.option("--clip", default=0.0, type=float, help="Gradient clipping")
@click.option("--logger", default='wandb', type=str, help="Logger")

def cli_finetune(name, datadir, outdir, lr, weight, grad, replicate, bs, shift, 
optim, clip, logger):
    """Finetune the Decima model."""
    matrix_file = os.path.join(datadir, "aggregated.h5ad")
    h5_file = os.path.join(datadir, "data.h5")
    print(f"Data paths: {matrix_file}, {h5_file}")

    print("Reading anndata")
    ad = anndata.read_h5ad(matrix_file)

    print("Making dataset objects")
    train_dataset = HDF5Dataset(
        h5_file=h5_file,
        ad=ad,
        key="train",
        max_seq_shift=shift,
        augment_mode="random",
        seed=0,
    )
    val_dataset = HDF5Dataset(h5_file=h5_file, ad=ad, key="val", max_seq_shift=0)

    train_params = {
        "name": name,
        "optimizer": optim,
        "batch_size": bs,
        "num_workers": 16,
        "devices": 0,
        "logger": logger,
        "save_dir": outdir,
        "max_epochs": 50,
        "lr": lr,
        "total_weight": weight,
        "accumulate_grad_batches": grad,
        "loss": "poisson_multinomial",
        #"pairs": ad.uns["disease_pairs"].values,
        "clip": clip,
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
    if logger == "wandb":
        run = wandb.init(project="decima", dir=name, name=name)
    model.train_on_dataset(train_dataset, val_dataset)
    train_dataset.close()
    val_dataset.close()
    run.finish()
