"""Make predictions for all genes using an HDF5 file created by Decima's ``write_hdf5.py``.

Usage:
  decima_predict.py [options]

Options:
  --device=<device>             Which GPU to use.
  --ckpts=<ckpts>...            Path to the model checkpoint(s).
  --h5_file=<file>              Path to h5 file indexed by genes.
  --matrix_file=<file>          Path to h5ad file containing genes to predict.
  --out_file=<file>             Output file path.
  --max_seq_shift=<shift>       Maximum jitter for augmentation [default: 0].
  -h --help                     Show this help message and exit.
"""

import os

import anndata
import numpy as np
import torch
from docopt import docopt
from lightning import LightningModel

from decima.data.read_hdf5 import HDF5Dataset, list_genes

# TODO: input can be just a h5ad file rather than a combination of h5 and matrix file.


def main():
    args = docopt(__doc__)

    device_id = int(args["--device"])
    ckpts = args["--ckpts"]
    h5_file = args["--h5_file"]
    matrix_file = args["--matrix_file"]
    out_file = args["--out_file"]
    max_seq_shift = int(args["--max_seq_shift"])

    torch.set_float32_matmul_precision("medium")

    # TODO: device is unused, set the device appropriately
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    device = torch.device(0)

    print("Loading anndata")
    ad = anndata.read_h5ad(matrix_file)
    assert np.all(list_genes(h5_file, key=None) == ad.var_names.tolist())

    print("Making dataset")
    ds = HDF5Dataset(
        key=None,
        h5_file=h5_file,
        ad=ad,
        seq_len=524288,
        max_seq_shift=max_seq_shift,
    )

    print("Loading models from checkpoint")
    models = [LightningModel.load_from_checkpoint(f).eval() for f in ckpts]

    print("Computing predictions")
    preds = (
        np.stack([model.predict_on_dataset(ds, devices=0, batch_size=6, num_workers=16) for model in models]).mean(0).T
    )
    ad.layers["preds"] = preds

    print("Computing correlations per gene")
    ad.var["pearson"] = [np.corrcoef(ad.X[:, i], ad.layers["preds"][:, i])[0, 1] for i in range(ad.shape[1])]
    ad.var["size_factor_pearson"] = [np.corrcoef(ad.X[:, i], ad.obs["size_factor"])[0, 1] for i in range(ad.shape[1])]
    print(
        f"Mean Pearson Correlation per gene: True: {ad.var.pearson.mean().round(2)} Size Factor: {ad.var.size_factor_pearson.mean().round(2)}"
    )

    print("Computing correlation per track")
    for dataset in ad.var.dataset.unique():
        key = f"{dataset}_pearson"
        ad.obs[key] = [
            np.corrcoef(
                ad[i, ad.var.dataset == dataset].X,
                ad[i, ad.var.dataset == dataset].layers["preds"],
            )[0, 1]
            for i in range(ad.shape[0])
        ]
        print(f"Mean Pearson Correlation per pseudobulk over {dataset} genes: {ad.obs[key].mean().round(2)}")

    print("Saved")
    ad.write_h5ad(out_file)


if __name__ == "__main__":
    main()
