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
    batch_size=1,
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

    logger.info("Making predictions")
    model = load_decima_model(model, device=device)

    ds = GeneDataset(genes=genes, metadata_anndata=metadata_anndata, max_seq_shift=max_seq_shift, genome=genome)
    preds = model.predict_on_dataset(
        ds, devices=device, batch_size=batch_size, num_workers=num_workers, float_precision=float_precision
    )

    logger.info("Creating anndata")
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
        for model_name, pred in zip(model.model_names, preds["ensemble_preds"]):
            ad.layers[model.name] = pred.T

    logger.info("Evaluating performance")
    evaluate_gene_expression_predictions(ad)
    return ad


def evaluate_gene_expression_predictions(ad):
    assert ad.X is not None, "ad.X is required for evaluation."
    assert ad.layers["preds"] is not None, "ad.layers['preds'] is required for evaluation."

    n_pbs = ad.shape[0]
    n_genes = ad.shape[1]
    truth = ad.X
    preds = ad.layers["preds"]

    # Compute Pearson correlation per gene
    ad.var["pearson"] = [np.corrcoef(truth[:, i], preds[:, i])[0, 1] for i in range(n_genes)]

    if "size_factor" not in ad.obs.columns:
        ad.obs["size_factor"] = ad.X.sum(1)

    ad.var["size_factor_pearson"] = [np.corrcoef(truth[:, i], ad.obs["size_factor"])[0, 1] for i in range(n_genes)]

    # compute correlations per pseudobulk
    for dataset in ad.var.dataset.unique():
        in_dataset = ad.var.dataset == dataset

        key = f"{dataset}_pearson"
        ad.obs[key] = [np.corrcoef(truth[i, in_dataset], preds[i, in_dataset])[0, 1] for i in range(n_pbs)]

        # Compute averages
        mean_per_gene = ad.var.loc[in_dataset, "pearson"].mean()
        mean_per_gene_sf = ad.var.loc[in_dataset, "size_factor_pearson"].mean()
        mean_per_pb = ad.obs[key].mean()

        # Report results
        print(f"Performance on genes in the {dataset} dataset.")
        print(f"Mean Pearson Correlation per gene: Mean: {mean_per_gene:.2f}.")
        print(f"Mean Pearson Correlation per gene using size factor (baseline): {mean_per_gene_sf:.2f}.")
        print(f"Mean Pearson Correlation per pseudobulk: {mean_per_pb: .2f}")
        print("")
