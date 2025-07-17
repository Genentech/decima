import pytest
from decima.tools.inference import predict_gene_expression

from conftest import device


@pytest.mark.long_running
def test_predict_gene_expression():
    ad = predict_gene_expression(
        genes=["SPI1", "GATA1"],
        metadata_anndata=None,
        model=0,
        device=device,
    )

    assert ad.layers["preds"].shape == (8856, 2)
    assert ad.var.shape[0] == 2
    assert ad.obs.shape[0] == 8856

    ad = predict_gene_expression(
        genes=["SPI1", "GATA1"],
        model="ensemble", device=device,
        save_replicates=True,
    )

    assert ad.layers["preds"].shape == (8856, 2)
    assert ad.var.shape[0] == 2
    assert ad.obs.shape[0] == 8856
    assert ad.layers["preds_v1_rep0"].shape == (8856, 2)
    assert ad.layers["preds_v1_rep1"].shape == (8856, 2)
    assert ad.layers["preds_v1_rep2"].shape == (8856, 2)
    assert ad.layers["preds_v1_rep3"].shape == (8856, 2)
