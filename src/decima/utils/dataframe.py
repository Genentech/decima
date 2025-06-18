import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Iterator, Generator


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
    def __init__(self, output_path: str):
        """Initialize ParquetWriter

        Args:
            output_path (str): Path to the output parquet file
        """
        self.output_path = output_path
        self.writer = None
        self.schema = None
        self.first_chunk = True

    def __enter__(self):
        self.first_chunk = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.writer is not None:
            self.writer.close()
        else:
            raise RuntimeError("NoDataFrameWrittenError: No dataframe was written to the parquet file.")
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


def write_df_chunks_to_parquet(chunks: Iterator[pd.DataFrame], output_path: str) -> None:
    """Write dataframe chunks to parquet file

    Args:
        chunks (Iterator[pd.DataFrame]): Iterator of dataframe chunks
        output_path (str): Path to the output parquet file
    """
    with ChunkDataFrameWriter(output_path) as writer:
        for chunk in chunks:
            writer.write(chunk)
