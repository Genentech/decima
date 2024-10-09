import anndata
import os, sys
import argparse
import wandb

src_dir = f'{os.path.dirname(__file__)}/../src/decima/'
sys.path.append(src_dir)
from read_hdf5 import HDF5Dataset
from lightning import LightningModel

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str)
parser.add_argument("--dir", type=str)
parser.add_argument("--lr", type=float)
parser.add_argument("--weight", type=float)
parser.add_argument("--grad", type=int)
parser.add_argument("--replicate", type=int, default=0)
parser.add_argument("--bs", type=int, default=4)
args = parser.parse_args()


def main():
    wandb.login(host = "https://genentech.wandb.io")
    run = wandb.init(project="decima", dir=args.name, name=args.name)

    # Get paths
    data_dir = args.dir
    matrix_file = os.path.join(data_dir, "aggregated.h5ad")
    h5_file = os.path.join(data_dir, "data.h5")
    print(f"Data paths: {matrix_file}, {h5_file}")

    # Load data
    print("Reading anndata")
    ad = anndata.read_h5ad(matrix_file)

    # Make datasets
    print("Making dataset objects")
    train_dataset = HDF5Dataset(h5_file=h5_file, ad=ad, key="train", max_seq_shift=5000, augment_mode="random", seed=0)
    val_dataset = HDF5Dataset(h5_file=h5_file, ad=ad, key="val", max_seq_shift=0)

    # Make param dicts
    train_params = {
        "optimizer": "adam",
        "batch_size": args.bs,
        "num_workers": 16,
        "devices": 0,
        "logger": "wandb",
        "save_dir": data_dir,
        "max_epochs": 15,
        "lr":args.lr,
        "total_weight": args.weight,
        "accumulate_grad_batches": args.grad,
        "loss": 'poisson_multinomial',
        "pairs": ad.uns["disease_pairs"].values
    }
    model_params = {
        "n_tasks":ad.shape[0],
        "replicate":args.replicate,
    }

    print(f"train_params: {train_params}")
    print(f"model_params: {model_params}")

    # Make model
    print("Initializing model")
    model = LightningModel(model_params=model_params, train_params=train_params)
    
    # Fine-tune model
    print("Training")
    model.train_on_dataset(train_dataset, val_dataset)
    
    train_dataset.close()
    val_dataset.close()
    run.finish()

if __name__ == "__main__":
    main()