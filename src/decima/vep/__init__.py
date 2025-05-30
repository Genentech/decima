import logging
from typing import Iterator, Optional, Union, List
import pandas as pd
from grelu.transforms.prediction_transforms import Aggregate

from decima.utils import get_compute_device
from decima.utils.dataframe import chunk_df, write_df_chunks_to_parquet
from decima.utils.io import read_vcf_chunks
from decima.data.dataset import VariantDataset
from decima.hub import load_decima_model


def predict_variant_effect(
    df_variant: pd.DataFrame,
    tasks: Optional[Union[str, List[str]]] = None,
    model: Union[int, str] = 0,
    batch_size: int = 8,
    num_workers: int = 16,
    device: Optional[str] = None,
    include_cols: Optional[List[str]] = None,
    gene_col: Optional[str] = None,
    min_from_end: int = 0,
    max_dist_tss: float = float("inf"),
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
    device = get_compute_device(device)

    dataset = VariantDataset(
        df_variant,
        include_cols=include_cols,
        gene_col=gene_col,
        min_from_end=min_from_end,
        max_dist_tss=max_dist_tss,
    )
    model = load_decima_model(model=model, device=device)

    if tasks is not None:
        tasks = dataset.result.query_cells(tasks)
        agg_transform = Aggregate(tasks=tasks, model=model)
        model.add_transform(agg_transform)
    else:
        tasks = dataset.result.cells

    preds = model.predict_on_dataset(dataset, devices=device, batch_size=batch_size, num_workers=num_workers)

    df = dataset.variants.reset_index(drop=True)
    df_pred = pd.DataFrame(preds, columns=tasks)

    return pd.concat([df, df_pred], axis=1)


def _predict_variant_effect_save(
    chunks: Iterator[pd.DataFrame],
    output_pq: str,
    tasks: Optional[Union[str, List[str]]] = None,
    model: Union[int, str] = 0,
    batch_size: int = 8,
    num_workers: int = 16,
    device: Optional[str] = None,
    include_cols: Optional[List[str]] = None,
    gene_col: Optional[str] = None,
    min_from_end: int = 0,
    max_dist_tss: float = float("inf"),
) -> None:
    """Predict variant effect and save to parquet

    Args:
        chunks (Iterator[pd.DataFrame]): Iterator of chunks of variants
        output_pq (str): Path to save the parquet file
        tasks (str, optional): Tasks to predict. Defaults to None.
        model (int, optional): Model to use. Defaults to 0.
        batch_size (int, optional): Batch size. Defaults to 8.
        num_workers (int, optional): Number of workers. Defaults to 16.
        device (str, optional): Device to use. Defaults to "cpu".
        include_cols (list, optional): Columns to include in the output. Defaults to None.
        gene_col (str, optional): Column name for gene names. Defaults to None.
        min_from_end (int, optional): Minimum distance from the end of the gene. Defaults to 0.
        max_dist_tss (float, optional): Maximum distance from the TSS. Defaults to inf.
    """
    logger = logging.getLogger(__name__)
    device = get_compute_device(device)
    logger.info(f"Using device: {device}")

    write_df_chunks_to_parquet(
        (
            predict_variant_effect(
                df_chunk,
                tasks,
                model,
                batch_size,
                num_workers,
                device,
                include_cols,
                gene_col,
                min_from_end,
                max_dist_tss,
            )
            for df_chunk in chunks
        ),
        output_pq,
    )


def predict_variant_effect_save(
    df_variant: pd.DataFrame,
    output_pq: str,
    tasks: Optional[Union[str, List[str]]] = None,
    model: Union[int, str] = 0,
    chunksize: int = 10_000,
    batch_size: int = 8,
    num_workers: int = 16,
    device: Optional[str] = None,
    include_cols: Optional[List[str]] = None,
    gene_col: Optional[str] = None,
    min_from_end: int = 0,
    max_dist_tss: float = float("inf"),
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
        min_from_end (int, optional): Minimum distance from the end of the gene. Defaults to 0.
        max_dist_tss (float, optional): Maximum distance from the TSS. Defaults to inf.
    """
    _predict_variant_effect_save(
        chunk_df(df_variant, chunksize),
        output_pq,
        tasks,
        model,
        batch_size,
        num_workers,
        device,
        include_cols,
        gene_col,
        min_from_end,
        max_dist_tss,
    )


def predict_vcf_variant_effect_save(
    vcf_file: str,
    output_pq: str,
    tasks: Optional[Union[str, List[str]]] = None,
    model: Union[int, str] = 0,
    chunksize: int = 10_000,
    batch_size: int = 8,
    num_workers: int = 16,
    device: Optional[str] = None,
    include_cols: Optional[List[str]] = None,
    gene_col: Optional[str] = None,
    min_from_end: int = 0,
    max_dist_tss: float = float("inf"),
) -> None:
    """Predict variant effect from VCF file and save to parquet

    Args:
        vcf_file (str): Path to the VCF file
        output_pq (str): Path to save the parquet file
        tasks (str, optional): Tasks to predict. Defaults to None.
        model (int, optional): Model to use. Defaults to 0.
        chunksize (int, optional): Number of variants to predict in each chunk. Defaults to 10_000.
        batch_size (int, optional): Batch size. Defaults to 8.
        num_workers (int, optional): Number of workers. Defaults to 16.
        device (str, optional): Device to use. Defaults to "cpu".
        include_cols (list, optional): Columns to include in the output. Defaults to None.
        gene_col (str, optional): Column name for gene names. Defaults to None.
        min_from_end (int, optional): Minimum distance from the end of the gene. Defaults to 0.
        max_dist_tss (float, optional): Maximum distance from the TSS. Defaults to inf.
    """
    _predict_variant_effect_save(
        read_vcf_chunks(vcf_file, chunksize),
        output_pq,
        tasks,
        model,
        batch_size,
        num_workers,
        device,
        include_cols,
        gene_col,
        min_from_end,
        max_dist_tss,
    )
