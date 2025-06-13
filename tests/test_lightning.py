import pytest
import torch
from decima.constants import DECIMA_CONTEXT_SIZE
from decima.data.dataset import VariantDataset
from decima.model.lightning import LightningModel
from decima.model.metrics import WarningType

from conftest import device


@pytest.fixture
def lightning_model():
    return LightningModel(model_params={'n_tasks': 1, 'init_borzoi': False}).to(device)


@pytest.mark.long_running
def test_LightningModel_predict_step(lightning_model):

    batch = torch.randn(1, 5, DECIMA_CONTEXT_SIZE).to(device)

    preds = lightning_model.predict_step(batch.to(device), 0)
    assert preds.shape == (1, 1, 1)

    batch = {"seq": batch.to(device), "warning": [WarningType.ALLELE_MISMATCH_WITH_REFERENCE_GENOME]}
    results = lightning_model.predict_step(batch, 1)
    assert results["expression"].shape == (1, 1, 1)
    assert results["warnings"] == [WarningType.ALLELE_MISMATCH_WITH_REFERENCE_GENOME]


@pytest.mark.long_running
def test_LightningModel_predict_on_dataset(lightning_model, df_variant):

    dataset = VariantDataset(df_variant)
    results = lightning_model.predict_on_dataset(dataset)
    assert results["expression"].shape == (82, 1)
    assert results["warnings"]['unknown'] == 0
    assert results["warnings"]['allele_mismatch_with_reference_genome'] == 13
