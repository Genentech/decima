"""Make predictions for all genes using an HDF5 file created by Decima's ``write_hdf5.py``."""

import logging
import click
import anndata
import numpy as np
from decima.constants import DECIMA_CONTEXT_SIZE
from decima.data.dataset import GeneDataset
from decima.hub import load_decima_model
from decima.utils import get_compute_device


@click.command()
@click.option("-o", "--output", type=click.Path(), help="Path to the output h5ad file.")
@click.option(
    "-m",
    "--model",
    type=str,
    default="ensemble",
    help="Path to the model checkpoint: `0`, `1`, `2`, `3`, `ensemble` or `path/to/model.ckpt`.",
)
@click.option(
    "--metadata",
    type=click.Path(exists=True),
    default=None,
    help="Path to the metadata anndata file. Default: None.",
)
@click.option(
    "--device",
    type=str,
    default=None,
    help="Device to use. Default: None which automatically selects the best device.",
)
@click.option("--batch-size", type=int, default=8, help="Batch size for the model. Default: 8")
@click.option("--num-workers", type=int, default=4, help="Number of workers for the loader. Default: 4")
@click.option("--max_seq_shift", default=0, help="Maximum jitter for augmentation.")
@click.option("--genome", type=str, default="hg38", help="Genome build. Default: hg38.")
@click.option(
    "--save-replicates",
    is_flag=True,
    help="Save the replicates in the output parquet file. Default: False.",
)
def cli_predict_genes(output, model, metadata, device, batch_size, num_workers, max_seq_shift, genome, save_replicates):
    if model in ["0", "1", "2", "3"]:
        model = int(model)

    if isinstance(device, str) and device.isdigit():
        device = int(device)

    if save_replicates and (model != "ensemble"):
        raise ValueError("`--save-replicates` is only supported for ensemble model (`--model ensemble`).")

    logger = logging.getLogger("decima")
    device = get_compute_device(device)
    logger.info(f"Using device: {device} and genome: {genome} for prediction.")

    # TODO: move to new function and call it here
    # TODO: allow passing list of genes useful for testing
    model = "ensemble"
    model = load_decima_model(model, device=device)

    ds = GeneDataset(metadata_anndata=metadata, seq_len=DECIMA_CONTEXT_SIZE, max_seq_shift=max_seq_shift)
    preds = model.predict_on_dataset(ds, devices=device, batch_size=batch_size, num_workers=num_workers)

    ad = anndata.AnnData(
        X=ds.result.anndata.X.copy(),
        layers={"preds": preds},
        var=ds.result.anndata.var.copy(),
        obs=ds.result.anndata.obs.copy(),
    )

    if save_replicates:
        for model, pred in zip(model.models, preds["ensemble_preds"]):
            ad.layers[f"{model.name}"] = pred.T

    if ad.X is not None:
        ad.var["pearson"] = [np.corrcoef(ad.X[:, i], ad.layers["preds"][:, i])[0, 1] for i in range(ad.shape[1])]
        ad.var["size_factor_pearson"] = [
            np.corrcoef(ad.X[:, i], ad.obs["size_factor"])[0, 1] for i in range(ad.shape[1])
        ]
        print(
            f"Mean Pearson Correlation per gene: True: {ad.var.pearson.mean().round(2)} Size Factor: {ad.var.size_factor_pearson.mean().round(2)}"
        )

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

    ad.write_h5ad(output)
