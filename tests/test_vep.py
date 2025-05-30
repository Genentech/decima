import numpy as np
import pandas as pd
import pytest
from decima.core.result import DecimaResult
from decima.data.dataset import VariantDataset
from decima.vep import predict_variant_effect, predict_variant_effect_save, predict_vcf_variant_effect_save
from decima.utils.io import read_vcf_chunks

from conftest import device


@pytest.fixture
def df_variant():
    return pd.DataFrame({
        "chrom": ["chr1", "chr1", "chr1", "chr1", "chr1"],
        "pos": [1000018, 1002308, 109727471, 109728286, 109728807],
        "ref": ["G", "T", "A", "T", "T"],
        "alt": ["A", "C", "C", "G", "G"],
    })


def test_read_vcf_chunks():
    chunks = list(read_vcf_chunks("tests/data/test.vcf", 10))
    assert len(chunks) == 1
    assert chunks[0].shape == (5, 4)
    assert chunks[0].columns.tolist() == ["chrom", "pos", "ref", "alt"]


def test_VariantDataset_overlap_genes(df_variant):
    df_genes = pd.DataFrame({
        "chrom": ["chr1", "chr1"],
        "start": [837298, 109380590],
        "end": [1361586, 109904878],
        "strand": ["+", "-"],
        "gene": ["ISG15", "GSTM3"],
        "gene_mask_start": [163840, 163840],
        "gene_mask_end": [164840, 164840],
    })

    df = VariantDataset.overlap_genes(df_variant, df_genes)
    pd.testing.assert_frame_equal(df, pd.DataFrame({
        "chrom": ["chr1", "chr1", "chr1", "chr1", "chr1"],
        "pos": [1000018, 1002308, 109727471, 109728286, 109728807],
        "ref": ["G", "T", "A", "T", "T"],
        "alt": ["A", "C", "C", "G", "G"],
        "gene": ["ISG15", "ISG15", "GSTM3", "GSTM3", "GSTM3"],
        "start": [837298, 837298, 109380590, 109380590, 109380590],
        "end": [1361586, 1361586, 109904878, 109904878, 109904878],
        "strand": ["+", "+", "-", "-", "-"],
        "gene_mask_start": [163840, 163840, 163840, 163840, 163840],
        "gene_mask_end": [164840, 164840, 164840, 164840, 164840],
        "rel_pos": [162720, 165010, 177407, 176592, 176071],
        "ref_tx": ["G", "T", "T", "A", "A"],
        "alt_tx": ["A", "C", "G", "C", "C"],
        "tss_dist": [-1120, 1170, 13567, 12752, 12231],
    }))

    df_variant['gene'] = ['ISG15', 'ISG15', 'ISG15', 'ISG15', 'ISG15']
    df = VariantDataset.overlap_genes(df_variant, df_genes, gene_col="gene")
    pd.testing.assert_frame_equal(df, pd.DataFrame({
        "chrom": ["chr1", "chr1"],
        "pos": [1000018, 1002308],
        "ref": ["G", "T"],
        "alt": ["A", "C"],
        "gene": ["ISG15", "ISG15"],
        "start": [837298, 837298],
        "end": [1361586, 1361586],
        "strand": ["+", "+"],
        "gene_mask_start": [163840, 163840],
        "gene_mask_end": [164840, 164840],
        "rel_pos": [162720, 165010],
        "ref_tx": ["G", "T"],
        "alt_tx": ["A", "C"],
        "tss_dist": [-1120, 1170],
    }))


def test_VariantDataset(df_variant):

    dataset = VariantDataset(df_variant)

    assert isinstance(dataset.result, DecimaResult)

    assert dataset.variants.columns.tolist() == [
        "chrom", "pos", "ref", "alt", "gene", "start", "end", "strand",
        "gene_mask_start", "gene_mask_end", "rel_pos", "ref_tx", "alt_tx", "tss_dist"
    ]

    assert len(dataset) == 82 * 2
    assert dataset[0].shape == (5, 524288)

    rows, cols = np.where(dataset[0] != dataset[1])
    assert rows.tolist() == [1, 3] # C > T
    assert cols.tolist() == [40725, 40725] # should be the same for both


@pytest.mark.long_running
def test_predict_variant_effect(df_variant):

    query = "cell_type == 'CD8-positive, alpha-beta T cell'"
    cells = DecimaResult.load().query_cells(query)

    df = predict_variant_effect(df_variant, tasks=query, device=device, max_dist_tss=5000)

    assert df.shape == (4, 273)
    assert df.columns.tolist() == [
        'chrom', 'pos', 'ref', 'alt', 'gene', 'start', 'end', 'strand',
        'gene_mask_start', 'gene_mask_end', 'rel_pos', 'ref_tx', 'alt_tx', 'tss_dist',
        *cells
    ]
    row = df.iloc[0]
    assert row['gene'] == 'HES4'
    assert row['tss_dist'] == 154


@pytest.mark.long_running
def test_predict_variant_effect_save(df_variant, tmp_path):
    output_file = tmp_path / "test_predictions.parquet"

    query = "cell_type == 'CD8-positive, alpha-beta T cell'"
    cells = DecimaResult.load().query_cells(query)

    predict_variant_effect_save(
        df_variant,
        str(output_file),
        tasks=query,
        device=device,
        max_dist_tss=5000,
        chunksize=5
    )

    assert output_file.exists()

    df_saved = pd.read_parquet(output_file)

    assert df_saved.shape == (4, 273)
    assert df_saved.columns.tolist() == [
        'chrom', 'pos', 'ref', 'alt', 'gene', 'start', 'end', 'strand',
        'gene_mask_start', 'gene_mask_end', 'rel_pos', 'ref_tx', 'alt_tx', 'tss_dist',
        *cells
    ]
    row = df_saved.iloc[0]
    assert row['gene'] == 'HES4'
    assert row['tss_dist'] == 154

    preds_cols = df_saved.columns[14:].tolist()
    assert len(preds_cols) == len(cells)


@pytest.mark.long_running
def test_predict_vcf_variant_effect_save(tmp_path):

    output_file = tmp_path / "test_predictions.parquet"
    predict_vcf_variant_effect_save(
        "tests/data/test.vcf",
        str(output_file),
        device=device,
        max_dist_tss=20000,
    )

    assert output_file.exists()

    df_saved = pd.read_parquet(output_file)
    assert df_saved.shape == (12, 8870)
