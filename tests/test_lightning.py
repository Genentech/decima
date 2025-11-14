import pytest
import torch
from decima.constants import DECIMA_CONTEXT_SIZE, MODEL_METADATA, DEFAULT_ENSEMBLE
from decima.data.dataset import VariantDataset
from decima.model.lightning import LightningModel, GeneMaskLightningModel
from decima.model.metrics import WarningType

from conftest import device


@pytest.fixture
def lightning_model():
    model_name = "v1_rep0"
    metadata = MODEL_METADATA[model_name]
    model = LightningModel(model_params={'n_tasks': metadata['num_tasks'], 'init_borzoi': False}, name=model_name).to(device)
    return model


@pytest.mark.long_running
def test_LightningModel_predict_step(lightning_model):
    metadata = MODEL_METADATA[MODEL_METADATA[DEFAULT_ENSEMBLE][0]]

    seq = torch.randn(1, 5, DECIMA_CONTEXT_SIZE).to(device)

    preds = lightning_model.predict_step(seq, 0)
    assert preds.shape == (1, metadata['num_tasks'], 1)

    batch = {"seq": seq, "warning": [WarningType.ALLELE_MISMATCH_WITH_REFERENCE_GENOME]}
    results = lightning_model.predict_step(batch, 1)
    assert results["expression"].shape == (1, metadata['num_tasks'], 1)
    assert results["warnings"] == [WarningType.ALLELE_MISMATCH_WITH_REFERENCE_GENOME]

    batch = {
        "seq": seq.to(device),
        "warning": [WarningType.ALLELE_MISMATCH_WITH_REFERENCE_GENOME],
        "pred_expr": {"v1_rep0": torch.ones((1, metadata['num_tasks']), device=device)}
    }
    results = lightning_model.predict_step(batch, 1)
    assert results["expression"].shape == (1, metadata['num_tasks'], 1)
    assert (results["expression"] == torch.ones((1, metadata['num_tasks'], 1), device=device)).all()
    assert results["warnings"] == [WarningType.ALLELE_MISMATCH_WITH_REFERENCE_GENOME]


@pytest.mark.long_running
def test_LightningModel_predict_on_dataset(lightning_model, df_variant):
    dataset = VariantDataset(df_variant, model_name="v1_rep0")
    results = lightning_model.predict_on_dataset(dataset)

    metadata = MODEL_METADATA[MODEL_METADATA[DEFAULT_ENSEMBLE][0]]

    assert results["expression"].shape == (82, metadata["num_tasks"])
    assert results["warnings"]['unknown'] == 0
    assert results["warnings"]['allele_mismatch_with_reference_genome'] == 13


@pytest.mark.long_running
def test_LightningModel_predict_on_dataset_ensemble(lightning_model, df_variant):
    dataset = VariantDataset(df_variant)
    results = lightning_model.predict_on_dataset(dataset)
    metadata = MODEL_METADATA[MODEL_METADATA[DEFAULT_ENSEMBLE][0]]
    assert results["expression"].shape == (82, metadata["num_tasks"])
    assert results["warnings"]['unknown'] == 0
    assert results["warnings"]['allele_mismatch_with_reference_genome'] == 13


@pytest.mark.long_running
def test_GeneMaskLightningModel_forward():
    seq = torch.randn(1, 4, DECIMA_CONTEXT_SIZE).to(device)
    metadata = MODEL_METADATA[MODEL_METADATA[DEFAULT_ENSEMBLE][0]]
    model = GeneMaskLightningModel(
        gene_mask_start=200_000, gene_mask_end=300_000,
        model_params={"n_tasks": metadata["num_tasks"], "init_borzoi": False}, name=metadata["name"]
    ).to(device)
    preds = model(seq)
    assert preds.shape == (1, metadata["num_tasks"], 1)
