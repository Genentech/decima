import pytest
import torch
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import pearsonr

from decima.core.result import DecimaResult
from decima.data.dataset import VariantDataset
from decima.model.metrics import WarningType
from decima.vep import _predict_variant_effect, predict_variant_effect
from decima.utils.io import read_vcf_chunks

from conftest import device


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
        "ref": ["G", "T", "A", "TTT", "T"],
        "alt": ["A", "C", "C", "G", "GG"],
        "gene": ["ISG15", "ISG15", "GSTM3", "GSTM3", "GSTM3"],
        "start": [837298, 837298, 109380590, 109380590, 109380590],
        "end": [1361586, 1361586, 109904878, 109904878, 109904878],
        "strand": ["+", "+", "-", "-", "-"],
        "gene_mask_start": [163840, 163840, 163840, 163840, 163840],
        "gene_mask_end": [164840, 164840, 164840, 164840, 164840],
        "rel_pos": [162719, 165009, 177407, 176592, 176071],
        "ref_tx": ["G", "T", "T", "AAA", "A"],
        "alt_tx": ["A", "C", "G", "C", "CC"],
        "tss_dist": [-1121, 1169, 13567, 12752, 12231],
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
        "rel_pos": [162719, 165009],
        "ref_tx": ["G", "T"],
        "alt_tx": ["A", "C"],
        "tss_dist": [-1121, 1169],
    }))

    with pytest.raises(ValueError):
        df_variant = pd.DataFrame({
            "chrom": ["chr13"],
            "pos": [30154905],
            "ref": ["G"],
            "alt": ["A"],
            "gene": ["TEX26-AS1"],
        })
        df = VariantDataset.overlap_genes(df_variant, df_genes)

def test_VariantDataset(df_variant):

    dataset = VariantDataset(df_variant, model_name="v1_rep0")

    assert isinstance(dataset.result, DecimaResult)
    assert dataset.variants.columns.tolist() == [
        "chrom", "pos", "ref", "alt", "gene", "start", "end", "strand",
        "gene_mask_start", "gene_mask_end", "rel_pos", "ref_tx", "alt_tx", "tss_dist"
    ]

    assert len(dataset) == 82 * 2
    assert dataset[0]['seq'].shape == (5, 524288)

    assert dataset[0]['pred_expr']['v1_rep0'].shape == (8856,)
    assert not dataset[0]['pred_expr']['v1_rep0'].isnan().any()
    assert dataset[1]['pred_expr']['v1_rep0'].isnan().all()
    assert not dataset[2]['pred_expr']['v1_rep0'].isnan().any()
    assert dataset[3]['pred_expr']['v1_rep0'].isnan().all()
    assert dataset[88]['pred_expr']['v1_rep0'].isnan().all()

    assert dataset[2]["warning"] == []
    assert dataset[3]["warning"] == []
    assert dataset[2]["warning"] == []
    assert dataset[10]["warning"] == []
    assert dataset[30]["warning"] == []
    assert dataset[88]["warning"] == [WarningType.ALLELE_MISMATCH_WITH_REFERENCE_GENOME]

    rows, cols = np.where(dataset[2]['seq'] != dataset[3]['seq'])
    assert rows.tolist() == [0, 2] # A > G
    assert cols.tolist() == [38435, 38435] # should be the same for both

    for i in range(len(dataset)):
        assert dataset[i]['seq'].shape == (5, 524288)

    rows, cols = np.where(dataset[162]['seq'] != dataset[163]['seq'])
    assert cols.min() == 505705 # the positions before should not be effected.
    assert cols.shape[0] > 10_000 # remaining most bp should be different due to shifting.

    dataset = VariantDataset(df_variant, max_seq_shift=100)

    assert len(dataset) == 82 * 2 * 201
    assert dataset[0]['seq'].shape == (5, 524288)

    for i in range(20):
        assert dataset[i]["warning"] == []
        assert dataset[i]['seq'].shape == (5, 524288)

    assert dataset[44 * 2 * 201]["warning"] == [WarningType.ALLELE_MISMATCH_WITH_REFERENCE_GENOME]

    rows, cols = np.where(dataset[201 * 2]['seq'] != dataset[201 * 2 + 1]['seq'])
    assert rows.tolist() == [0, 2] # A > G
    assert cols.tolist() == [38435 + 100, 38435 + 100] # should be the same for both


@pytest.mark.long_running
def test_VariantDataset_dataloader(df_variant):

    dataset = VariantDataset(df_variant, model_name="ensemble")
    dl = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=0, collate_fn=dataset.collate_fn)
    batches = iter(dl)

    batch = next(batches)
    assert batch["seq"].shape == (64, 5, 524288)
    assert batch["warning"] == []
    assert batch["pred_expr"]["v1_rep0"].shape == (64, 8856)
    assert batch["pred_expr"]["v1_rep1"].shape == (64, 8856)
    assert batch["pred_expr"]["v1_rep2"].shape == (64, 8856)
    assert batch["pred_expr"]["v1_rep3"].shape == (64, 8856)

    batch = next(batches)
    assert batch["seq"].shape == (64, 5, 524288)
    assert len(batch["warning"]) > 0
    assert WarningType.ALLELE_MISMATCH_WITH_REFERENCE_GENOME in batch["warning"]
    assert batch["pred_expr"]["v1_rep0"].shape == (64, 8856)
    assert batch["pred_expr"]["v1_rep1"].shape == (64, 8856)
    assert batch["pred_expr"]["v1_rep2"].shape == (64, 8856)
    assert batch["pred_expr"]["v1_rep3"].shape == (64, 8856)

@pytest.mark.long_running
def test_VariantDataset_dataloader_vcf():

    df_variant = next(read_vcf_chunks("tests/data/test.vcf", 10000))
    dataset = VariantDataset(df_variant, model_name="ensemble", max_distance=20000)
    dl = torch.utils.data.DataLoader(dataset, batch_size=8, num_workers=0, collate_fn=dataset.collate_fn)
    batches = iter(dl)

    batch = next(batches)
    assert batch["seq"].shape == (8, 5, 524288)
    assert batch["warning"] == []
    assert batch["pred_expr"]['v1_rep0'].shape == (8, 8856)
    assert batch["pred_expr"]['v1_rep1'].shape == (8, 8856)
    assert batch["pred_expr"]['v1_rep2'].shape == (8, 8856)
    assert batch["pred_expr"]['v1_rep3'].shape == (8, 8856)

    batch = next(batches)
    assert batch["seq"].shape == (8, 5, 524288)
    assert batch["warning"] == []
    assert batch["pred_expr"]['v1_rep0'].shape == (8, 8856)
    assert batch["pred_expr"]['v1_rep1'].shape == (8, 8856)
    assert batch["pred_expr"]['v1_rep2'].shape == (8, 8856)
    assert batch["pred_expr"]['v1_rep3'].shape == (8, 8856)

    batch = next(batches)
    assert batch["seq"].shape == (8, 5, 524288)
    assert len(batch["warning"]) > 0
    assert batch["pred_expr"]['v1_rep0'].shape == (8, 8856)
    assert batch["pred_expr"]['v1_rep1'].shape == (8, 8856)
    assert batch["pred_expr"]['v1_rep2'].shape == (8, 8856)
    assert batch["pred_expr"]['v1_rep3'].shape == (8, 8856)


@pytest.mark.long_running
def test_predict_variant_effect(df_variant):

    query = "cell_type == 'CD8-positive, alpha-beta T cell'"
    cells = DecimaResult.load().query_cells(query)

    df, warnings, num_variants = _predict_variant_effect(df_variant, model=0, tasks=query, device=device, max_distance=5000)
    assert num_variants == 4

    assert df.shape == (4, 273)
    assert warnings['unknown'] == 0
    assert warnings['allele_mismatch_with_reference_genome'] == 0
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
    warnings_file = tmp_path / "test_predictions.parquet.warnings.log"

    query = "cell_type == 'CD8-positive, alpha-beta T cell'"
    cells = DecimaResult.load().query_cells(query)

    predict_variant_effect(
        df_variant,
        output_pq=str(output_file),
        model="ensemble",
        tasks=query,
        device=device,
        max_distance=5000,
        chunksize=5,
        float_precision="16-true"
    )

    assert output_file.exists()
    assert not warnings_file.exists()

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
def test_predict_variant_effect_vcf(tmp_path):

    output_file = tmp_path / "test_predictions.parquet"
    warnings_file = tmp_path / "test_predictions.parquet.warnings.log"

    predict_variant_effect(
        "tests/data/test.vcf",
        output_pq=str(output_file),
        model=0,
        device=device,
        max_distance=20000,
    )
    assert output_file.exists()

    parquet_file = pq.ParquetFile(output_file)
    metadata = parquet_file.metadata.metadata

    assert metadata[b"genome"] == b"hg38"
    assert metadata[b"model"] == b"v1_rep0"
    assert metadata[b"min_distance"] == b"0"
    assert metadata[b"max_distance"] == b"20000"

    df_saved = pd.read_parquet(output_file)
    assert df_saved.shape == (12, 8870)

    with open(warnings_file, 'r') as f:
        warnings = f.read()
        assert "allele_mismatch_with_reference_genome: 1 / 12" in warnings


@pytest.mark.long_running
def test_predict_variant_effect_check_results():
    df_orig = pd.read_csv("tests/data/test_preds.csv.gz")
    df_variants = df_orig[df_orig.columns[:5]]
    df_preds = predict_variant_effect(
        df_variants, model=0, device=device, gene_col="gene"
    )

    for i in range(df_orig.shape[0]):
        orig = df_orig.iloc[i][df_orig.columns[5:]].values.tolist()
        pred = df_preds.iloc[i][df_preds.columns[14:]].values.tolist()
        assert pearsonr(orig, pred).statistic > 0.99
        assert orig == pytest.approx(pred, abs=2e-2)
        # np.where(np.abs(np.array(orig) - np.array(pred)) < 2e-2)

@pytest.mark.long_running
def test_predict_variant_effect_vcf_ensemble(tmp_path):

    output_file = tmp_path / "test_predictions.parquet"
    predict_variant_effect(
        "tests/data/test.vcf",
        output_pq=str(output_file),
        model="ensemble",
        device=device,
        max_distance=20000,
    )
    assert output_file.exists()

    df_saved = pd.read_parquet(output_file)
    assert df_saved.shape == (12, 8870)


@pytest.mark.long_running
def test_predict_variant_effect_vcf_ensemble_replicates(tmp_path):
    output_file = tmp_path / "test_predictions.parquet"
    predict_variant_effect(
        "tests/data/test.vcf",
        output_pq=str(output_file),
        model="ensemble",
        device=device,
        max_distance=20000,
        save_replicates=True,
    )
    assert output_file.exists()

    df_saved = pd.read_parquet(output_file)
    assert df_saved.shape == (12, 44294)

    cells = list(df_saved.columns[14:8870])
    average_preds = np.mean([
        df_saved[[f"{cell}_v1_rep{i}" for cell in cells]].values
        for i in range(4)
    ], axis=0)
    np.testing.assert_allclose(df_saved[cells].values, average_preds, rtol=1e-5)
