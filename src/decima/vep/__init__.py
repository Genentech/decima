import logging
import warnings
from pathlib import Path
from collections import Counter
from typing import Optional, Union, List

import pandas as pd
from grelu.transforms.prediction_transforms import Aggregate

from decima.constants import SUPPORTED_GENOMES
from decima.model.metrics import WarningType
from decima.utils import get_compute_device
from decima.utils.dataframe import chunk_df, ChunkDataFrameWriter
from decima.utils.io import read_vcf_chunks
from decima.data.dataset import VariantDataset
from decima.hub import load_decima_model


def _predict_variant_effect(
    df_variant: Union[pd.DataFrame, str],
    tasks: Optional[Union[str, List[str]]] = None,
    model: Union[int, str] = "ensemble",
    metadata_anndata: Optional[str] = None,
    batch_size: int = 8,
    num_workers: int = 16,
    device: Optional[str] = None,
    include_cols: Optional[List[str]] = None,
    gene_col: Optional[str] = None,
    distance_type: Optional[str] = "tss",
    min_distance: Optional[float] = 0,
    max_distance: Optional[float] = float("inf"),
    genome: str = "hg38",
    save_replicates: bool = False,
    reference_cache: bool = True,
    float_precision: str = "32",
) -> pd.DataFrame:
    """Predict variant effect on a set of variants

    Args:
        df_variant (pd.DataFrame): DataFrame with variant information
        tasks (str, optional): Tasks to predict. Defaults to None.
        model (int, optional): Model to use. Defaults to 0.
        metadata_anndata (str, optional): Path to anndata file. Defaults to None.
        batch_size (int, optional): Batch size. Defaults to 8.
        num_workers (int, optional): Number of workers. Defaults to 16.
        device (str, optional): Device to use. Defaults to "cpu".
        include_cols (list, optional): Columns to include in the output. Defaults to None.
        gene_col (str, optional): Column name for gene names. Defaults to None.
        distance_type (str, optional): Type of distance. Defaults to "tss".
        min_distance (float, optional): Minimum distance from the end of the gene. Defaults to 0 (inclusive).
        max_distance (float, optional): Maximum distance from the TSS. Defaults to inf (exclusive).

    Returns:
        pd.DataFrame: DataFrame with variant effect predictions
    """
    if (genome not in SUPPORTED_GENOMES) and (not Path(genome).exists()):
        raise ValueError(f"Genome {genome} not supported. Currently only hg38 is supported.")
    include_cols = include_cols or list()

    model = load_decima_model(model=model, device=device)

    try:
        dataset = VariantDataset(
            df_variant,
            include_cols=include_cols,
            metadata_anndata=metadata_anndata,
            gene_col=gene_col,
            distance_type=distance_type,
            min_distance=min_distance,
            max_distance=max_distance,
            model_name=model.name,
            reference_cache=reference_cache,
        )
    except ValueError as e:
        if str(e).startswith("NoOverlapError"):
            warnings.warn("No overlapping gene and variant found. Skipping this chunk...")
            return pd.DataFrame(columns=[*VariantDataset.DEFAULT_COLUMNS, *include_cols]), [], 0
        else:
            raise e

    model = load_decima_model(model=model)

    if tasks is not None:
        tasks = dataset.result.query_cells(tasks)

        model.reset_transform()
        agg_transform = Aggregate(tasks=tasks, model=model)
        model.add_transform(agg_transform)
    else:
        tasks = dataset.result.cells

    logging.getLogger("decima").info(f"Performing predictions on {dataset}")
    results = model.predict_on_dataset(
        dataset, devices=device, batch_size=batch_size, num_workers=num_workers, float_precision=float_precision
    )

    df = dataset.variants.reset_index(drop=True)
    df_pred = pd.DataFrame(results["expression"], columns=tasks)

    if save_replicates:
        df_pred = pd.concat(
            [
                df_pred,
                *[
                    pd.DataFrame(pred, columns=tasks).rename(columns=lambda x: f"{x}_{model.name}")
                    for i, (pred, model) in enumerate(zip(results["ensemble_preds"], model.models))
                ],
            ],
            axis=1,
        )
    return pd.concat([df, df_pred], axis=1), results["warnings"], results["expression"].shape[0]


def predict_variant_effect(
    df_variant: Union[pd.DataFrame, str],
    output_pq: Optional[str] = None,
    tasks: Optional[Union[str, List[str]]] = None,
    model: Union[int, str] = "ensemble",
    metadata_anndata: Optional[str] = None,
    chunksize: int = 10_000,
    batch_size: int = 8,
    num_workers: int = 16,
    device: Optional[str] = None,
    include_cols: Optional[List[str]] = None,
    gene_col: Optional[str] = None,
    distance_type: Optional[str] = "tss",
    min_distance: Optional[float] = 0,
    max_distance: Optional[float] = float("inf"),
    genome: str = "hg38",
    save_replicates: bool = False,
    reference_cache: bool = True,
    float_precision: str = "32",
) -> None:
    """Predict variant effect and save to parquet

    Args:
        df_variant (pd.DataFrame): DataFrame with variant information
        output_path (str): Path to save the parquet file
        tasks (str, optional): Tasks to predict. Defaults to None.
        model (int, optional): Model to use. Defaults to 0.
        metadata_anndata (str, optional): Path to anndata file. Defaults to None.
        chunksize (int, optional): Number of variants to predict in each chunk. Defaults to 10_000.
        batch_size (int, optional): Batch size. Defaults to 8.
        num_workers (int, optional): Number of workers. Defaults to 16.
        device (str, optional): Device to use. Defaults to "cpu".
        include_cols (list, optional): Columns to include in the output. Defaults to None.
        gene_col (str, optional): Column name for gene names. Defaults to None.
        distance_type (str, optional): Type of distance. Defaults to "tss".
        min_distance (float, optional): Minimum distance from the end of the gene. Defaults to 0 (inclusive).
        max_distance (float, optional): Maximum distance from the TSS. Defaults to inf (exclusive).
        genome (str, optional): Genome build. Defaults to "hg38".
    """
    logger = logging.getLogger("decima")
    device = get_compute_device(device)
    logger.info(f"Using device: {device} and genome: {genome}")

    if isinstance(df_variant, pd.DataFrame):
        chunks = chunk_df(df_variant, chunksize)
    elif isinstance(df_variant, str):
        if df_variant.endswith(".tsv"):
            chunks = pd.read_csv(df_variant, sep="\t", chunksize=chunksize)
        elif df_variant.endswith(".csv"):
            chunks = pd.read_csv(df_variant, sep=",", chunksize=chunksize)
        elif df_variant.endswith(".vcf") or df_variant.endswith(".vcf.gz"):
            chunks = read_vcf_chunks(df_variant, chunksize)
        else:
            raise ValueError(f"Unsupported file extension: {df_variant}. Must be .tsv or .vcf.")
    else:
        raise ValueError(
            f"Unsupported input type: {type(df_variant)}. Must be pd.DataFrame or str (path to .tsv or .vcf)."
        )

    model = load_decima_model(model=model, device=device)

    results = (
        _predict_variant_effect(
            df_variant=df_chunk,
            tasks=tasks,
            model=model,
            metadata_anndata=metadata_anndata,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            include_cols=include_cols,
            gene_col=gene_col,
            distance_type=distance_type,
            min_distance=min_distance,
            max_distance=max_distance,
            genome=genome,
            save_replicates=save_replicates,
            reference_cache=reference_cache,
            float_precision=float_precision,
        )
        for df_chunk in chunks
    )

    warning_counter = Counter()
    num_variants = 0

    if Path(genome).exists():
        genome_path = genome
    else:
        import genomepy

        genome_path = genomepy.Genome(genome).filename

    if output_pq is not None:
        metadata = {
            "genome": genome,
            "model": model.name,
            "min_distance": int(min_distance) if min_distance is not None else None,
            "max_distance": max_distance,
        }
        with ChunkDataFrameWriter(output_pq, metadata=metadata) as writer:
            for df, warnings, _num_variants in results:
                if df.shape[0] == 0:
                    warnings.append("no_overlap_found_for_chunk")
                else:
                    writer.write(df)
                num_variants += _num_variants
                warning_counter.update(warnings)
        if warning_counter.total():
            with open(output_pq + ".warnings.log", "w") as f:
                for warning, count in warning_counter.items():
                    f.write(f"{warning}: {count} / {num_variants} \n")
    else:
        _df = list()
        for df, warnings, _num_variants in results:
            num_variants += _num_variants
            warning_counter.update(warnings)
            _df.append(df)

    if warning_counter.total():
        logger.warning("Warnings:")

        for warning, count in warning_counter.items():
            if count == 0:
                continue
            if warning == WarningType.ALLELE_MISMATCH_WITH_REFERENCE_GENOME.value:
                logger.warning(
                    f"{warning}: {count} alleles out of {num_variants} predictions mismatched with the genome file {genome_path}."
                    "If this is not expected, please check if you are using the correct genome version."
                )
            elif warning == "no_overlap_found_for_chunk":
                logger.warning(f"{warning}: {count} chunks with no overlap found with genes.")
            else:
                logger.warning(f"{warning}: {count} out of {num_variants} variants")

    if output_pq is None:
        return pd.concat(_df)
