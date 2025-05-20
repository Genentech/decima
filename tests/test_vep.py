import numpy as np
import pandas as pd
import pytest
from decima.core.result import DecimaResult
from decima.data.dataset import VariantDataset
from decima.vep import predict_variant_effect

from conftest import device


@pytest.fixture
def variant_df():
    return pd.DataFrame({
        "chrom": ["chr1", "chr1", "chr1", "chr1", "chr1"],
        "pos": [1000018, 1002308, 109727471, 109728286, 109728807],
        "ref": ["G", "T", "A", "T", "T"],
        "alt": ["A", "C", "C", "G", "G"],
    })

def test_VariantDataset_overlap_genes(variant_df):
    df_genes = pd.DataFrame({
        "chrom": ["chr1", "chr1"],
        "start": [837298, 109380590],
        "end": [1361586, 109904878],
        "strand": ["+", "-"],
        "gene": ["ISG15", "GSTM3"],
        "gene_mask_start": [163840, 163840],
    })

    df = VariantDataset.overlap_genes(variant_df, df_genes)

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
        "rel_pos": [162720, 165010, 177407, 176592, 176071],
        "ref_tx": ["G", "T", "T", "A", "A"],
        "alt_tx": ["A", "C", "G", "C", "C"],
        "tss_dist": [-1120, 1170, 13567, 12752, 12231],
    }))


def test_VariantDataset(variant_df):

    dataset = VariantDataset(variant_df)

    assert isinstance(dataset.result, DecimaResult)

    assert dataset.variants.columns.tolist() == [
        "chrom", "pos", "ref", "alt", "gene", "start", "end", "strand",
        "gene_mask_start", "rel_pos", "ref_tx", "alt_tx", "tss_dist"
    ]

    assert len(dataset) == 82 * 2
    assert dataset[0].shape == (5, 524288)

    rows, cols = np.where(dataset[0] != dataset[1])
    assert rows.tolist() == [1, 3] # C > T
    assert cols.tolist() == [40725, 40725] # should be the same for both


def test_predict_variant_effect(variant_df):

    df_variant = variant_df.iloc[:1]
    result = DecimaResult.load()

    query = "cell_type == 'CD8-positive, alpha-beta T cell'"
    df = predict_variant_effect(df_variant, tasks=query, device=device, max_dist_tss=250)

    assert df.shape == (1, 272)
    assert df.columns.tolist() == [
        'chrom', 'pos', 'ref', 'alt', 'gene', 'start', 'end', 'strand', 'gene_mask_start', 'rel_pos', 'ref_tx', 'alt_tx', 'tss_dist',
        *result.query_cells(query)
    ]
    row = df.iloc[0]
    assert row['gene'] == 'HES4'
    assert row['tss_dist'] == 154
