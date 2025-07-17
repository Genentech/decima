import pytest
import anndata
import pandas as pd
import torch

from decima.core.result import DecimaResult
from decima.core.metadata import GeneMetadata, CellMetadata
from decima.model.decima_model import DecimaModel

from decima.constants import DECIMA_CONTEXT_SIZE

from conftest import device


def test_load_result():
    result = DecimaResult.load()
    assert isinstance(result.anndata, anndata.AnnData)
    assert result.anndata.shape == (8856, 18457)


@pytest.mark.long_running
def test_model_loading():
    result = DecimaResult.load()

    result.load_model()
    assert result.model is not None
    assert isinstance(result.model.model, DecimaModel)

    result.load_model(model=1)
    assert result.model is not None

def test_metadata_access():
    result = DecimaResult.load()
    assert len(result.genes) == 18457
    assert len(result.cells) == 8856

    assert isinstance(result.cell_metadata, pd.DataFrame)
    assert isinstance(result.gene_metadata, pd.DataFrame)

    cell, gene = 'agg_0', 'STRADA'

    gene_meta = result.get_gene_metadata(gene)
    assert gene_meta.chrom == 'chr17'
    assert gene_meta.strand == '-'
    assert gene_meta.gene_type == 'protein_coding'
    assert gene_meta.start == 63381538
    assert gene_meta.end == 63905826
    assert gene_meta.gene_start == 63682336
    assert gene_meta.gene_end == 63741986
    assert gene_meta.gene_length == 59650
    assert gene_meta.gene_mask_start == 163840
    assert gene_meta.gene_mask_end == 223490
    assert gene_meta.frac_N == 0.0
    assert gene_meta.fold == ['fold1']
    assert gene_meta.dataset == 'train'

    assert isinstance(gene_meta, GeneMetadata)

    cell_meta = result.get_cell_metadata(cell)
    assert isinstance(cell_meta, CellMetadata)


def test_predictions():
    result = DecimaResult.load()

    preds = result.predicted_expression_matrix()
    assert isinstance(preds, pd.DataFrame)
    assert preds.shape == result.shape

    genes = result.genes[:5]
    preds = result.predicted_expression_matrix(genes)
    assert isinstance(preds, pd.DataFrame)
    assert preds.shape == (result.shape[0], 5)
    assert all(g in preds.columns for g in genes)

    genes = result.genes[:5]
    preds = result.predicted_expression_matrix(genes, model_name="v1_rep0")
    assert isinstance(preds, pd.DataFrame)
    assert preds.shape == (result.shape[0], 5)
    assert all(g in preds.columns for g in genes)

    genes = result.genes[:5]
    preds = result.predicted_expression_matrix(genes, model_name="v1_rep3")
    assert isinstance(preds, pd.DataFrame)
    assert preds.shape == (result.shape[0], 5)
    assert all(g in preds.columns for g in genes)

    assert isinstance(result.predicted_gene_expression("STRADA", "v1_rep3"), torch.Tensor)
    assert result.predicted_gene_expression("STRADA", model_name="v1_rep3").shape == (result.shape[0],)


@pytest.mark.long_running
def test_attributions():
    result = DecimaResult.load()
    result.load_model(device=device)

    gene = 'SPI1'
    cell_type = 'classical monocyte'
    attr = result.attributions(gene, f"cell_type == '{cell_type}'")
    assert attr.gene == gene
    assert attr.chrom == 'chr11'


def test_one_hot_preparation():
    result = DecimaResult.load()
    gene = 'SPI1'

    one_hot, gene_mask = result.prepare_one_hot(gene)
    assert isinstance(one_hot, torch.Tensor)
    assert one_hot.shape == (4, DECIMA_CONTEXT_SIZE)
    assert gene_mask.shape == (1, DECIMA_CONTEXT_SIZE)


def test_DecimaResult_prepare_one_hot_indel():
    result = DecimaResult.load()
    gene = 'STRADA'

    one_hot, gene_mask = result.prepare_one_hot(gene, variants=[{"chrom": "chr17", "pos": 63682336, "ref": "CCCCC", "alt": "T"}])
    assert isinstance(one_hot, torch.Tensor)
    assert one_hot.shape == (4, DECIMA_CONTEXT_SIZE)
    assert gene_mask.shape == (1, DECIMA_CONTEXT_SIZE)
