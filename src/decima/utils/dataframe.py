import glob
import warnings
from pathlib import Path
from typing import Iterator, Generator, List, Optional, Union

from tqdm import tqdm
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from more_itertools import flatten


def chunk_df(df: pd.DataFrame, chunksize: int) -> Generator[pd.DataFrame, None, None]:
    """Chunk dataframe into chunks of size chunksize

    Args:
        df (pd.DataFrame): Input dataframe
        chunksize (int): Size of each chunk

    Returns:
        Generator[pd.DataFrame, None, None]: Generator of dataframe chunks
    """
    for i in range(0, len(df), chunksize):
        yield df.iloc[i : i + chunksize]


class ChunkDataFrameWriter:
    def __init__(self, output_path: str, metadata: Optional[dict] = None):
        """Initialize ParquetWriter

        Args:
            output_path (str): Path to the output parquet file
            metadata (dict): Metadata to write to the parquet file. Keys and values must be string-like / coercible to bytes.
        """
        self.output_path = output_path
        self.writer = None
        self.schema = None
        self.first_chunk = True
        self.metadata = metadata

    def __enter__(self):
        self.first_chunk = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.writer is not None:
            if self.metadata is not None:
                self.writer.add_key_value_metadata({str(k): str(v) for k, v in self.metadata.items()})
            self.writer.close()
        else:
            warnings.warn("NoDataFrameWrittenError: No dataframe was written to the parquet file.")
        self.first_chunk = True

    def write(self, chunk: pd.DataFrame) -> None:
        """Write dataframe chunk to parquet file

        Args:
            chunk (pd.DataFrame): DataFrame chunk to write
        """
        if self.first_chunk:
            self.schema = pa.Table.from_pandas(chunk).schema
            self.writer = pq.ParquetWriter(self.output_path, self.schema)
            self.first_chunk = False

        self.writer.write_table(pa.Table.from_pandas(chunk))


def write_df_chunks_to_parquet(
    chunks: Iterator[pd.DataFrame], output_path: str, metadata: Optional[dict] = None
) -> None:
    """Write dataframe chunks to parquet file

    Args:
        chunks (Iterator[pd.DataFrame]): Iterator of dataframe chunks
        output_path (str): Path to the output parquet file
        metadata (dict): Metadata to write to the parquet file. If None, no metadata is written.
    """
    with ChunkDataFrameWriter(output_path, metadata) as writer:
        for chunk in chunks:
            writer.write(chunk)


def read_metadata_from_replicate_parquets(files: Union[str, List[str]]) -> pd.DataFrame:
    """Read metadata from multiple parquet files and return as a DataFrame.

    This function reads key-value metadata from each parquet file and extracts
    model, distance parameters and other metadata into a structured DataFrame.
    All files must contain the required metadata fields.

    Args:
        files (List[str]): List of parquet file paths to read metadata from

    Returns:
        pd.DataFrame: DataFrame containing metadata with columns:
            - model: Model identifier
            - max_distance: Maximum distance used for predictions
            - min_distance: Minimum distance used for predictions
            - file: Source file path

    Raises:
        KeyError: If any required metadata field is missing from a file
    """
    df = list()
    if isinstance(files, str) or isinstance(files, Path):
        files = [files]

    files = list(flatten([glob.glob(str(f)) for f in files]))

    for file in tqdm(files, total=len(files), desc="Reading metadata from parquet files"):
        parquet_file = pq.ParquetFile(file)
        metadata = parquet_file.metadata.metadata

        if not metadata:
            raise KeyError(f"No metadata found in file: {file}")

        df.append(
            {
                "model": metadata[b"model"].decode("utf-8"),
                "max_distance": int(float(metadata[b"max_distance"].decode("utf-8"))),
                "min_distance": int(float(metadata[b"min_distance"].decode("utf-8"))),
                "file": file,
            }
        )

    return pd.DataFrame(df)


def _ensemble_predictions(files: List[str], save_replicates: bool = False) -> Iterator[pd.DataFrame]:
    """Aggregate replicates from parquet files

    Args:
        files (List[str]): List of parquet files to aggregate
        output_pq (Optional[str]): Path to the output parquet file
    """
    df_metadata = read_metadata_from_replicate_parquets(files)

    model_names = set(df_metadata["model"])
    df_metadata = df_metadata.groupby(["max_distance", "min_distance"]).agg(list)

    assert all(df_metadata["model"].map(len) == len(model_names)), (
        "All groups must have the same number of models "
        "however found different numbers of models in the following groups: "
        + f"{df_metadata[df_metadata['model'].map(len) != len(model_names)]}"
    )

    for _, rows in tqdm(df_metadata.iterrows(), total=len(df_metadata), desc="Ensembling predictions"):
        df_variants = pd.read_parquet(rows["file"][0])
        ann_cols = df_variants.columns.tolist()[:14]
        cell_cols = df_variants.columns.tolist()[14:]
        df_variants = df_variants[ann_cols]

        preds = list()

        # TO CONSIDER: batch read if needed or switch to polars if too slow
        for model_name, file_path in zip(rows["model"], rows["file"]):
            _df = pd.read_parquet(file_path)
            assert (
                (_df.shape[0] == df_variants.shape[0])
                and (_df["chrom"] == df_variants["chrom"]).all()
                and (_df["pos"] == df_variants["pos"]).all()
                and (_df["ref"] == df_variants["ref"]).all()
                and (_df["alt"] == df_variants["alt"]).all()
                and (_df["gene"] == df_variants["gene"]).all()
            ), (
                f"Variant data mismatch between files {rows['file']}."
                " Are you sure you are aggregating the correct files containing the same variants?"
            )
            preds.append(_df[cell_cols].rename(columns=lambda x: f"{x}_{model_name}"))

        df_pred = pd.DataFrame(np.mean(preds, axis=0), columns=cell_cols)

        if save_replicates:
            df_pred = pd.concat([df_pred, *preds], axis=1)

        yield pd.concat([df_variants, df_pred], axis=1)


def ensemble_predictions(
    files: Union[str, List[str]], output_pq: Optional[str] = None, save_replicates: bool = False
) -> None:
    """Aggregate replicates from parquet files

    Args:
        files (List[str]): List of parquet files to aggregate
        output_pq (Optional[str]): Path to the output parquet file
    """
    if output_pq is not None:
        write_df_chunks_to_parquet(_ensemble_predictions(files, save_replicates), output_pq)
    else:
        return pd.concat(_ensemble_predictions(files, save_replicates), axis=0)
