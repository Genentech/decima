import pytest
import anndata
from decima.model.lightning import LightningModel
from decima.utils.resources import load_decima_model, load_decima_metadata


def test_load_decima_model():
    model_0 = load_decima_model()
    assert model_0 is not None
    assert isinstance(model_0, LightningModel)

    model_2 = load_decima_model(rep=2)
    assert model_2 is not None


def test_load_decima_metadata():
    metadata = load_decima_metadata()
    assert isinstance(metadata, anndata.AnnData)
    assert metadata.shape == (8856, 18457)
