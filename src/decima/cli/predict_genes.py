"""Make predictions for all genes using an HDF5 file created by Decima's ``write_hdf5.py``."""

import os
import click
import anndata
import numpy as np
import torch
from decima.constants import DECIMA_CONTEXT_SIZE
from decima.model.lightning import LightningModel
from decima.data.read_hdf5 import list_genes
from decima.data.dataset import HDF5Dataset

# TODO: input can be just a h5ad file rather than a combination of h5 and matrix file.


@click.command()
@click.option("--device", type=int, help="Which GPU to use.")
@click.option("--ckpts", multiple=True, required=True, help="Path to the model checkpoint(s).")
@click.option("--h5_file", required=True, help="Path to h5 file indexed by genes.")
@click.option("--matrix_file", required=True, help="Path to h5ad file containing genes to predict.")
@click.option("--out_file", required=True, help="Output file path.")
@click.option("--key", type=str, default=None, help="train, val or test. If None, all genes will be predicted.")
@click.option("--max_seq_shift", default=0, help="Maximum jitter for augmentation.")
@click.option("--bs", default=8, help="Batch size.")
@click.option("--num_workers", default=16, help="Number of workers.")
def cli_predict_genes(device, ckpts, h5_file, matrix_file, out_file, key, max_seq_shift, bs, num_workers):
    """Make predictions for all genes."""
    torch.set_float32_matmul_precision("medium")

    # TODO: device is unused, set the device appropriately
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    device = torch.device(0)

    print("Loading anndata")
    ad = anndata.read_h5ad(matrix_file)
    if key is not None:
        print(f"Subsetting anndata to {key} genes")
        ad = ad[:, ad.var.dataset == key]
    assert np.all(list_genes(h5_file, key=key) == ad.var_names.tolist())

    print("Making dataset")
    ds = HDF5Dataset(
        key=key,
        h5_file=h5_file,
        ad=ad,
        seq_len=DECIMA_CONTEXT_SIZE,
        max_seq_shift=max_seq_shift,
    )

    print("Loading models from checkpoint")
    models = [LightningModel.load_from_checkpoint(f).eval() for f in ckpts]

    print("Computing predictions")
    preds = (
        np.stack([model.predict_on_dataset(ds, devices=0, batch_size=bs, num_workers=num_workers) for model in models]).mean(0).T
    )
    ad.layers["preds"] = preds

    print("Computing correlations per gene")
    ad.var["pearson"] = [np.corrcoef(ad.X[:, i], ad.layers["preds"][:, i])[0, 1] for i in range(ad.shape[1])]
    ad.var["size_factor_pearson"] = [np.corrcoef(ad.X[:, i], ad.obs["size_factor"])[0, 1] for i in range(ad.shape[1])]
    print(
        f"Mean Pearson Correlation per gene: True: {ad.var.pearson.mean().round(2)} Size Factor: {ad.var.size_factor_pearson.mean().round(2)}"
    )

    print("Computing correlation per track")
    if key is None:
        for dataset in ad.var.dataset.unique():
            ad.obs[f"{dataset}_pearson"] = [
                np.corrcoef(
                    ad[i, ad.var.dataset == dataset].X,
                    ad[i, ad.var.dataset == dataset].layers["preds"],
                )[0, 1]
                for i in range(ad.shape[0])
            ]
            print(
                f"Mean Pearson Correlation per pseudobulk over {dataset} genes: {ad.obs[f'{dataset}_pearson'].mean().round(2)}"
            )
    else:
        ad.obs[f"{key}_pearson"] = [np.corrcoef(ad[i, :].X, ad[i, :].layers["preds"])[0, 1] for i in range(ad.shape[0])]
        print(f"Mean Pearson Correlation per pseudobulk over {key} genes: {ad.obs[f'{key}_pearson'].mean().round(2)}")

    print(f"Saving to {out_file}")
    ad.write_h5ad(out_file)
    print("Saved.")
