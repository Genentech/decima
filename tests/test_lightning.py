import pytest
import torch
from decima.constants import DECIMA_CONTEXT_SIZE, NUM_CELLS
from decima.data.dataset import VariantDataset
from decima.model.lightning import LightningModel
from decima.model.metrics import WarningType

from conftest import device


@pytest.fixture
def lightning_model():
    model = LightningModel(model_params={'n_tasks': NUM_CELLS, 'init_borzoi': False}, name='v1_rep0').to(device)
    return model


@pytest.mark.long_running
def test_LightningModel_predict_step(lightning_model):

    seq = torch.randn(1, 5, DECIMA_CONTEXT_SIZE).to(device)

    preds = lightning_model.predict_step(seq, 0)
    assert preds.shape == (1, NUM_CELLS, 1)

    batch = {"seq": seq, "warning": [WarningType.ALLELE_MISMATCH_WITH_REFERENCE_GENOME]}
    results = lightning_model.predict_step(batch, 1)
    assert results["expression"].shape == (1, NUM_CELLS, 1)
    assert results["warnings"] == [WarningType.ALLELE_MISMATCH_WITH_REFERENCE_GENOME]

    batch = {
        "seq": seq.to(device),
        "warning": [WarningType.ALLELE_MISMATCH_WITH_REFERENCE_GENOME],
        "pred_expr": {"v1_rep0": torch.ones((1, NUM_CELLS), device=device)}
    }
    results = lightning_model.predict_step(batch, 1)
    assert results["expression"].shape == (1, NUM_CELLS, 1)
    assert (results["expression"] == torch.ones((1, NUM_CELLS, 1), device=device)).all()
    assert results["warnings"] == [WarningType.ALLELE_MISMATCH_WITH_REFERENCE_GENOME]


@pytest.mark.long_running
def test_LightningModel_predict_on_dataset(lightning_model, df_variant):
    dataset = VariantDataset(df_variant, model_name="v1_rep0")
    results = lightning_model.predict_on_dataset(dataset)
    assert results["expression"].shape == (82, NUM_CELLS)
    assert results["warnings"]['unknown'] == 0
    assert results["warnings"]['allele_mismatch_with_reference_genome'] == 13


@pytest.mark.long_running
def test_LightningModel_predict_on_dataset_ensemble(lightning_model, df_variant):
    dataset = VariantDataset(df_variant)
    results = lightning_model.predict_on_dataset(dataset)
    assert results["expression"].shape == (82, NUM_CELLS)
    assert results["warnings"]['unknown'] == 0
    assert results["warnings"]['allele_mismatch_with_reference_genome'] == 13
