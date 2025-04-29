"""Finetune the Decima model.

Usage:
  decima_train.py [options]

Options:
  --name=<name>         Project name.
  --dir=<dir>           Data directory path.
  --lr=<lr>             Learning rate [default: 0.001].
  --weight=<weight>     Weight parameter.
  --grad=<grad>         Gradient accumulation steps.
  --replicate=<rep>     Replication number [default: 0].
  --bs=<bs>             Batch size [default: 4].
  -h --help             Show this help message and exit.
"""

import os

import anndata
import wandb
from docopt import docopt
from lightning import LightningModel

from decima.data.read_hdf5 import HDF5Dataset


def main():
    args = docopt(__doc__)

    name = args["--name"]
    data_dir = args["--dir"]
    lr = float(args["--lr"])
    weight = float(args["--weight"])
    grad = int(args["--grad"])
    replicate = int(args["--replicate"])
    batch_size = int(args["--bs"])

    wandb.login(host="https://genentech.wandb.io")
    run = wandb.init(project="decima", dir=name, name=name)

    matrix_file = os.path.join(data_dir, "aggregated.h5ad")
    h5_file = os.path.join(data_dir, "data.h5")
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
        "batch_size": batch_size,
        "num_workers": 16,
        "devices": 0,
        "logger": "wandb",
        "save_dir": data_dir,
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


if __name__ == "__main__":
    main()
