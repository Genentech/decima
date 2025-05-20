import pytest
import anndata
import pandas as pd
import torch

from decima.core.result import DecimaResult
from decima.core.metadata import GeneMetadata, CellMetadata
from decima.model.decima_model import DecimaModel

from conftest import device


def test_load_result():
    result = DecimaResult.load()
    assert isinstance(result.anndata, anndata.AnnData)
    assert result.anndata.shape == (8856, 18457)


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
    assert one_hot.shape == (4, 524288)
    assert gene_mask.shape == (1, 524288)
