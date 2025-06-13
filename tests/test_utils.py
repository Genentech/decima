import pandas as pd
from decima.utils.dataframe import chunk_df, write_df_chunks_to_parquet
from decima.utils.io import read_vcf_chunks


def test_chunk_df():
    chunks = list(chunk_df(pd.read_csv("tests/data/variants.tsv", sep="\t"), chunksize=2))
    assert len(chunks) == 3
    assert len(chunks[0]) == 2
    assert len(chunks[1]) == 2
    assert len(chunks[2]) == 1


def test_read_vcf_chunks():
    chunks = list(read_vcf_chunks("tests/data/test.vcf", chunksize=2))
    assert len(chunks) == 3
    assert len(chunks[0]) == 2
    assert len(chunks[1]) == 2
    assert len(chunks[2]) == 1


def test_write_df_chunks_to_parquet(tmp_path):
    output_path = tmp_path / "variants_test.parquet"
    df = pd.read_csv("tests/data/variants.tsv", sep="\t")
    chunks = chunk_df(df, chunksize=2)
    write_df_chunks_to_parquet(chunks, output_path)
    pd.testing.assert_frame_equal(pd.read_parquet(output_path), df)
