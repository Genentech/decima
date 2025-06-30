import pandas as pd
import pytest
import pyarrow as pa
import pyarrow.parquet as pq
from decima.utils.io import read_vcf_chunks
from decima.utils.dataframe import chunk_df, write_df_chunks_to_parquet, ensemble_predictions


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

    chunks = chunk_df(df, chunksize=2)
    write_df_chunks_to_parquet(
        chunks, output_path,
        metadata={"model": "decima_rep0", "min_distance": 0, "max_distance": 10000}
    )

    parquet_file = pq.ParquetFile(output_path)
    assert parquet_file.metadata.metadata[b'model'] == b'decima_rep0'
    assert parquet_file.metadata.metadata[b'min_distance'] == b'0'
    assert parquet_file.metadata.metadata[b'max_distance'] == b'10000'


def test_ensemble_predictions(tmp_path):
    pred_paths = [
        tmp_path / "model_rep0.parquet",
        tmp_path / "model_rep1.parquet",
        tmp_path / "model_rep2.parquet",
        tmp_path / "model_rep3.parquet",
    ]

    df = pd.DataFrame({
        "chrom": ["1", "1", "1"],
        "pos": [100000000, 100000001, 100000002],
        "ref": ["A", "C", "G"],
        "alt": ["G", "T", "A"],
        "gene": ["GENE1", "GENE1", "GENE1"],
        "start": [100000000, 100000001, 100000002],
        "end": [100000001, 100000002, 100000003],
        "strand": ["+", "+", "+"],
        "gene_mask_start": [100000000, 100000001, 100000002],
        "gene_mask_end": [100000001, 100000002, 100000003],
        "rel_pos": [0, 1, 2],
        "ref_tx": ["A", "G", "T"],
        "alt_tx": ["T", "C", "G"],
        "tss_dist": [0, 1, 2],
        "agg_0": [0.1, 0.2, 0.3],
        "agg_1": [0.4, 0.5, 0.6],
        "agg_2": [0.7, 0.8, 0.9],
        "agg_3": [1.0, 1.1, 1.2],
    })
    for i in range(4):
        with pq.ParquetWriter(pred_paths[i], pa.Table.from_pandas(df).schema) as writer:
            writer.add_key_value_metadata({"model": f"decima_rep{i}", "min_distance": "0", "max_distance": "10000"})
            writer.write_table(pa.Table.from_pandas(df))

    ensemble_predictions(pred_paths, output_pq=tmp_path / "ensemble_predictions.parquet")

    df_ensemble = pd.read_parquet(tmp_path / "ensemble_predictions.parquet")
    assert df_ensemble.columns.tolist() == [
        "chrom", "pos", "ref", "alt", "gene", "start", "end", "strand",
        "gene_mask_start", "gene_mask_end", "rel_pos", "ref_tx", "alt_tx", "tss_dist",
        "agg_0", "agg_1", "agg_2", "agg_3"
    ]

    ensemble_predictions(pred_paths, output_pq=tmp_path / "ensemble_predictions.parquet", save_replicates=True)
    df_ensemble = pd.read_parquet(tmp_path / "ensemble_predictions.parquet")
    assert df_ensemble.columns.tolist() == [
        "chrom", "pos", "ref", "alt", "gene", "start", "end", "strand",
        "gene_mask_start", "gene_mask_end", "rel_pos", "ref_tx", "alt_tx", "tss_dist",
        "agg_0", "agg_1", "agg_2", "agg_3",
        "agg_0_decima_rep0", "agg_1_decima_rep0", "agg_2_decima_rep0", "agg_3_decima_rep0",
        "agg_0_decima_rep1", "agg_1_decima_rep1", "agg_2_decima_rep1", "agg_3_decima_rep1",
        "agg_0_decima_rep2", "agg_1_decima_rep2", "agg_2_decima_rep2", "agg_3_decima_rep2",
        "agg_0_decima_rep3", "agg_1_decima_rep3", "agg_2_decima_rep3", "agg_3_decima_rep3",
    ]

    for i in range(3):
        with pq.ParquetWriter(pred_paths[i], pa.Table.from_pandas(df).schema) as writer:
            writer.add_key_value_metadata({"model": f"decima_rep{i}", "min_distance": "0", "max_distance": "10000"})
            writer.write_table(pa.Table.from_pandas(df))

    pred_path = tmp_path / "model_*.parquet"
    ensemble_predictions(pred_path, output_pq=tmp_path / "ensemble_predictions.parquet")
    df_ensemble = pd.read_parquet(tmp_path / "ensemble_predictions.parquet")
    assert df_ensemble.shape == (3, 18)

    with pq.ParquetWriter(pred_paths[i], pa.Table.from_pandas(df).schema) as writer:
        writer.add_key_value_metadata({"model": f"decima_rep{i}", "min_distance": "0", "max_distance": "10000"})
        _df = pd.DataFrame({
            "chrom": ["1"],
            "pos": [100000000],
            "ref": ["A"],
            "alt": ["G"],
            "gene": ["GENE1"],
            "start": [100000000],
            "end": [100000001],
            "strand": ["+"],
            "gene_mask_start": [100000000],
            "gene_mask_end": [100000001],
            "rel_pos": [0],
            "ref_tx": ["A"],
            "alt_tx": ["T"],
            "tss_dist": [0],
            "agg_0": [0.1],
            "agg_1": [0.4],
            "agg_2": [0.7],
            "agg_3": [1.],
        })
        writer.write_table(pa.Table.from_pandas(_df))

    with pytest.raises(AssertionError):
        ensemble_predictions(pred_paths, output_pq=tmp_path / "ensemble_predictions.parquet")
