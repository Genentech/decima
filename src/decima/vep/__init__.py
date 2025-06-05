import logging
from collections import Counter
from typing import Iterator, Optional, Union, List
import genomepy
import pandas as pd
from grelu.transforms.prediction_transforms import Aggregate

from decima.model.metrics import WarningType
from decima.utils import get_compute_device
from decima.utils.dataframe import chunk_df, ChunkDataFrameWriter
from decima.utils.io import read_vcf_chunks
from decima.data.dataset import VariantDataset
from decima.hub import load_decima_model


def _predict_variant_effect(
    df_variant: Union[pd.DataFrame, str],
    tasks: Optional[Union[str, List[str]]] = None,
    model: Union[int, str] = 0,
    batch_size: int = 8,
    num_workers: int = 16,
    device: Optional[str] = None,
    include_cols: Optional[List[str]] = None,
    gene_col: Optional[str] = None,
    max_distance: Optional[float] = float("inf"),
    max_distance_type: Optional[str] = "tss",
    genome: str = "hg38",
) -> pd.DataFrame:
    """Predict variant effect on a set of variants

    Args:
        df_variant (pd.DataFrame): DataFrame with variant information
        tasks (str, optional): Tasks to predict. Defaults to None.
        model (int, optional): Model to use. Defaults to 0.
        batch_size (int, optional): Batch size. Defaults to 8.
        num_workers (int, optional): Number of workers. Defaults to 16.
        device (str, optional): Device to use. Defaults to "cpu".
        include_cols (list, optional): Columns to include in the output. Defaults to None.
        gene_col (str, optional): Column name for gene names. Defaults to None.
        min_from_end (int, optional): Minimum distance from the end of the gene. Defaults to 0.
        max_dist_tss (float, optional): Maximum distance from the TSS. Defaults to inf.

    Returns:
        pd.DataFrame: DataFrame with variant effect predictions
    """

    assert genome in ["hg38"], "Currently only hg38 genome is supported."

    dataset = VariantDataset(
        df_variant,
        include_cols=include_cols,
        gene_col=gene_col,
        max_distance=max_distance,
        max_distance_type=max_distance_type,
    )
    model = load_decima_model(model=model, device=device)

    if tasks is not None:
        tasks = dataset.result.query_cells(tasks)
        agg_transform = Aggregate(tasks=tasks, model=model)
        model.add_transform(agg_transform)
    else:
        tasks = dataset.result.cells

    results = model.predict_on_dataset(dataset, devices=device, batch_size=batch_size, num_workers=num_workers)

    df = dataset.variants.reset_index(drop=True)
    df_pred = pd.DataFrame(results["expression"], columns=tasks)

    return pd.concat([df, df_pred], axis=1), results["warnings"], results["expression"].shape[0]


def predict_variant_effect(
    df_variant: Union[pd.DataFrame, str],
    output_pq: Optional[str] = None,
    tasks: Optional[Union[str, List[str]]] = None,
    model: Union[int, str] = 0,
    chunksize: int = 10_000,
    batch_size: int = 8,
    num_workers: int = 16,
    device: Optional[str] = None,
    include_cols: Optional[List[str]] = None,
    gene_col: Optional[str] = None,
    max_distance: Optional[float] = float("inf"),
    max_distance_type: Optional[str] = "tss",
    genome: str = "hg38",
) -> None:
    """Predict variant effect and save to parquet

    Args:
        df_variant (pd.DataFrame): DataFrame with variant information
        output_path (str): Path to save the parquet file
        tasks (str, optional): Tasks to predict. Defaults to None.
        model (int, optional): Model to use. Defaults to 0.
        chunksize (int, optional): Number of variants to predict in each chunk. Defaults to 10_000.
        batch_size (int, optional): Batch size. Defaults to 8.
        num_workers (int, optional): Number of workers. Defaults to 16.
        device (str, optional): Device to use. Defaults to "cpu".
        include_cols (list, optional): Columns to include in the output. Defaults to None.
        gene_col (str, optional): Column name for gene names. Defaults to None.
        max_distance (float, optional): Maximum distance from the TSS. Defaults to inf.
        max_distance_type (str, optional): Type of maximum distance. Defaults to None.
        genome (str, optional): Genome build. Defaults to "hg38".
    """
    logger = logging.getLogger(__name__)
    device = get_compute_device(device)
    logger.info(f"Using device: {device} and genome: {genome}")

    if isinstance(df_variant, pd.DataFrame):
        chunks = chunk_df(df_variant, chunksize)
    elif isinstance(df_variant, str):
        if df_variant.endswith(".tsv"):
            chunks = pd.read_csv(df_variant, sep="\t", chunksize=chunksize)
        elif df_variant.endswith(".vcf") or df_variant.endswith(".vcf.gz"):
            chunks = read_vcf_chunks(df_variant, chunksize)
        else:
            raise ValueError(f"Unsupported file extension: {df_variant}. Must be .tsv or .vcf.")
    else:
        raise ValueError(
            f"Unsupported input type: {type(df_variant)}. Must be pd.DataFrame or str (path to .tsv or .vcf)."
        )

    results = (
        _predict_variant_effect(
            df_chunk,
            tasks,
            model,
            batch_size,
            num_workers,
            device,
            include_cols,
            gene_col,
            max_distance,
            max_distance_type,
            genome,
        )
        for df_chunk in chunks
    )

    warning_counter = Counter()
    num_variants = 0

    genome_path = genomepy.Genome(genome).filename

    if output_pq is not None:
        with ChunkDataFrameWriter(output_pq) as writer:
            for df, warnings, _num_variants in results:
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
        logger.info("Warnings:")

        for warning, count in warning_counter.items():
            if count == 0:
                continue
            if warning == WarningType.ALLELE_MISMATCH_WITH_REFERENCE_GENOME.value:
                logger.info(
                    f"{warning}: {count} alleles out of {num_variants} predictions mismatched with the genome file {genome_path}."
                    "If this is not expected, please check if you are using the correct genome version."
                )
            else:
                logger.info(f"{warning}: {count} out of {num_variants} variants")

    if output_pq is None:
        return pd.concat(_df)
