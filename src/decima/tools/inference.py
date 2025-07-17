import anndata
import logging
import numpy as np
from decima.data.dataset import GeneDataset
from decima.hub import load_decima_model
from decima.utils import get_compute_device


def predict_gene_expression(
    genes=None,
    model="ensemble",
    metadata_anndata=None,
    device=None,
    batch_size=8,
    num_workers=4,
    max_seq_shift=0,
    genome="hg38",
    save_replicates=False,
    float_precision="32",
):
    """Predict gene expression for a list of genes

    Args:
        genes (list, optional): List of genes to predict. Defaults to None.
        model (str, optional): Model to use for prediction. Defaults to 'ensemble'.
        metadata_anndata (str, optional): Path to the metadata anndata file. Defaults to None.
        device (str, optional): Device to use for prediction. Defaults to None.
        batch_size (int, optional): Batch size for prediction. Defaults to 8.
        num_workers (int, optional): Number of workers for prediction. Defaults to 4.
        max_seq_shift (int, optional): Maximum sequence shift for prediction. Defaults to 0.
        genome (str, optional): Genome build for prediction. Defaults to 'hg38'.
        save_replicates (bool, optional): Save the replicates for prediction. Defaults to False.

    Raises:
        ValueError: If the model is not 'ensemble' and save_replicates is True.

    Returns:
        anndata.AnnData: AnnData object with the predicted gene expression.
    """
    logger = logging.getLogger("decima")
    device = get_compute_device(device)
    logger.info(f"Using device: {device} and genome: {genome} for prediction.")

    model = load_decima_model(model, device=device)

    ds = GeneDataset(genes=genes, metadata_anndata=metadata_anndata, max_seq_shift=max_seq_shift)
    preds = model.predict_on_dataset(
        ds, devices=device, batch_size=batch_size, num_workers=num_workers, float_precision=float_precision
    )

    X = None
    if ds.result.anndata.X is not None:
        X = ds.result.anndata.X.copy()
        X = X[:, ds.result.genes.isin(ds.genes)]

    ad = anndata.AnnData(
        X=X,
        layers={"preds": preds["expression"].T},
        var=ds.result.anndata.var.loc[ds.genes].copy(),
        obs=ds.result.anndata.obs.copy(),
    )

    if save_replicates:
        for model, pred in zip(model.models, preds["ensemble_preds"]):
            ad.layers[f"preds_{model.name}"] = pred.T

    if ad.X is not None:
        ad.var["pearson"] = [np.corrcoef(ad.X[:, i], ad.layers["preds"][:, i])[0, 1] for i in range(ad.shape[1])]
        ad.var["size_factor_pearson"] = [
            np.corrcoef(ad.X[:, i], ad.obs["size_factor"])[0, 1] for i in range(ad.shape[1])
        ]
        print(
            f"Mean Pearson Correlation per gene: True: {round(ad.var.pearson.mean(), 2)}. "
            f"Size Factor: {round(ad.var.size_factor_pearson.mean(), 2)}."
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
            print(f"Mean Pearson Correlation per pseudobulk over {dataset} genes: {round(ad.obs[key].mean(), 2)}")
    else:
        del ad.var["pearson"]
        del ad.var["size_factor_pearson"]

        for dataset in ad.var.dataset.unique():
            del ad.obs[f"{dataset}_pearson"]

    return ad
