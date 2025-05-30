import pandas as pd
from typing import Iterator, Generator


def chunk_df(df: pd.DataFrame, chunksize: int) -> Generator[pd.DataFrame, None, None]:
    for i in range(0, len(df), chunksize):
        yield df.iloc[i : i + chunksize]


def write_df_chunks_to_parquet(chunks: Iterator[pd.DataFrame], output_path: str) -> None:
    df_chunk = next(chunks)
    df_chunk.to_parquet(output_path)

    for df_chunk in chunks:
        df_chunk.to_parquet(output_path, append=True)
