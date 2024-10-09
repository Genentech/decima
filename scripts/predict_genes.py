# Given an hdf5 file created by write_hdf5.py, make predictions for all the genes

import numpy as np
import pandas as pd
import anndata
import os, sys
import torch
from tqdm import tqdm
import argparse

src_dir = f'{os.path.dirname(__file__)}/../src/decima/'
sys.path.append(src_dir)

from read_hdf5 import HDF5Dataset, list_genes
from lightning import LightningModel


parser = argparse.ArgumentParser()
parser.add_argument("--device", 
                    help="which gpu to use",
                    type=int)
parser.add_argument("--ckpts", help="Path to the model checkpoint", nargs='+')
parser.add_argument("--h5_file", 
                    help="Path to h5 file indexed by genes")
parser.add_argument("--matrix_file", 
                    help="Path to h5ad file containing genes to predict")
parser.add_argument("--out_file", 
                    help="Output file path")
parser.add_argument("--max_seq_shift", 
                    help="Maximum jitter for augmentation", default=0, type=int)

args = parser.parse_args()


torch.set_float32_matmul_precision("medium")
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
device = torch.device(0)

print("Loading anndata")
ad = anndata.read_h5ad(args.matrix_file)
assert np.all(list_genes(args.h5_file, key=None) == ad.var_names.tolist())

print("Making dataset")
ds = HDF5Dataset(
    key=None,
    h5_file=args.h5_file,
    ad=ad,
    seq_len=524288,
    max_seq_shift=args.max_seq_shift,
)

print("Loading models from checkpoint")
models = [LightningModel.load_from_checkpoint(f).eval() for f in args.ckpts]

print("Computing predictions")
preds = np.stack([model.predict_on_dataset(ds, devices=0, batch_size=6, num_workers=16) for model in models]).mean(0).T
ad.layers['preds'] = preds

print("Computing correlations per gene")
ad.var["pearson"] = [np.corrcoef(ad.X[:, i], ad.layers["preds"][:, i])[0, 1] for i in range(ad.shape[1])]
ad.var["size_factor_pearson"] = [np.corrcoef(ad.X[:, i], ad.obs['size_factor'])[0, 1] for i in range(ad.shape[1])]
print(f"Mean Pearson Correlation per gene: True: {ad.var.pearson.mean().round(2)} Size Factor: {ad.var.size_factor_pearson.mean().round(2)}")

print("Computing correlation per track")
for dataset in ad.var.dataset.unique():
    key = f"{dataset}_pearson"
    ad.obs[key] = [np.corrcoef(ad[i, ad.var.dataset==dataset].X, ad[i, ad.var.dataset==dataset].layers["preds"])[0, 1] for i in range(ad.shape[0])]
    print(f"Mean Pearson Correlation per pseudobulk over {dataset} genes: {ad.obs[key].mean().round(2)}")


print("Saved")
ad.write_h5ad(args.out_file)
