import pytest
import torch
import numpy as np
from decima.constants import DECIMA_CONTEXT_SIZE, MODEL_METADATA, DEFAULT_ENSEMBLE
from decima.data.dataset import VariantDataset
from decima.model.lightning import LightningModel, EnsembleLightningModel, GeneMaskLightningModel
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


def test_EnsembleLightningModel_predict_on_dataset_matches_individual_replicates():
    """Ensemble mean must equal the mean of per-model predictions for any n_seqs.

    Two bugs caused wrong results when n_seqs > 1:
    - Bug 1 (batch_size=1): allele dimension scrambled by 4-model concat in predict_step.
    - Bug 2 (n_seqs > 1): rearrange '(e b) -> e b' treated model as outer dim, but the
      actual ordering from predict_step is variant-outer, so models and variants were mixed.

    We mock predict_on_dataset on each constituent model to isolate and test the ensemble
    averaging logic without running expensive forward passes.
    """
    from unittest.mock import patch, MagicMock

    n_variants, n_tasks = 5, 10  # n_variants > 1 exercises the n_seqs > 1 bug
    warnings = {"allele_mismatch_with_reference_genome": 0, "unknown": 0}

    np.random.seed(0)
    preds_m0 = np.random.randn(n_variants, n_tasks).astype(np.float32)
    np.random.seed(1)
    preds_m1 = np.random.randn(n_variants, n_tasks).astype(np.float32)

    m0 = LightningModel(model_params={"n_tasks": n_tasks, "init_borzoi": False}, name="v1_rep0")
    m1 = LightningModel(model_params={"n_tasks": n_tasks, "init_borzoi": False}, name="v1_rep1")
    ensemble = EnsembleLightningModel([m0, m1])

    sentinel_dataset = MagicMock()

    with patch.object(m0, "predict_on_dataset", return_value={"expression": preds_m0, "warnings": warnings}) as mock0, \
         patch.object(m1, "predict_on_dataset", return_value={"expression": preds_m1, "warnings": warnings}) as mock1:
        result = ensemble.predict_on_dataset(sentinel_dataset, device="cpu", batch_size=4)

    # Both constituent models must have been called with the forwarded arguments
    mock0.assert_called_once_with(dataset=sentinel_dataset, device="cpu", num_workers=1, batch_size=4,
                                  augment_aggfunc="mean", compare_func=None, float_precision="32")
    mock1.assert_called_once_with(dataset=sentinel_dataset, device="cpu", num_workers=1, batch_size=4,
                                  augment_aggfunc="mean", compare_func=None, float_precision="32")

    expected_mean = (preds_m0 + preds_m1) / 2
    np.testing.assert_allclose(result["expression"], expected_mean, rtol=1e-6)

    # Per-replicate outputs must match the corresponding individual model predictions
    np.testing.assert_allclose(result["ensemble_preds"][0], preds_m0, rtol=1e-6)
    np.testing.assert_allclose(result["ensemble_preds"][1], preds_m1, rtol=1e-6)
