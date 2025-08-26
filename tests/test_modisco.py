import pytest
import h5py
from pathlib import Path

from decima.interpret.modisco import predict_save_modisco_attributions, modisco_patterns, modisco_reports, modisco
from decima.constants import DECIMA_CONTEXT_SIZE

from conftest import device


@pytest.mark.long_running
def test_predict_save_modisco_attributions(tmp_path):
    output_prefix = tmp_path / "test_modisco"
    predict_save_modisco_attributions(
        output_prefix=output_prefix,
        tasks="(cell_type == 'classical monocyte') and (tissue == 'blood')",
        off_tasks="(cell_type != 'classical monocyte') and (tissue == 'blood')",
        top_n_markers=5,
        model=0,
        device=device,
    )
    attribution_file = tmp_path / "test_modisco.attributions.h5"
    with h5py.File(attribution_file, "r") as f:
        assert f["attribution"].shape == (5, 4, DECIMA_CONTEXT_SIZE)
        assert f["sequence"].shape == (5, 4, DECIMA_CONTEXT_SIZE)
        assert list(f["genes"][:]) == [b'MEFV', b'AQP9', b'CLEC5A', b'CLEC4D', b'PLA2G7']
        assert f.attrs['model_name'] == 'v1_rep0'


@pytest.mark.long_running
def test_modisco(tmp_path):
    output_prefix = tmp_path / "test_modisco"
    modisco(
        output_prefix=output_prefix,
        tasks="cell_type == 'classical monocyte' and tissue == 'blood'",
        off_tasks="cell_type != 'classical monocyte' and tissue == 'blood'",
        top_n_markers=5,
        model='ensemble',
        max_seqlets_per_metacluster=10_000,
        device=device,
        tss_distance=5_000,
        n_leiden_runs=1,
    )
    assert Path(str(output_prefix) + "_0.attributions.h5").exists()
    assert Path(str(output_prefix) + "_1.attributions.h5").exists()
    assert Path(str(output_prefix) + "_2.attributions.h5").exists()
    assert Path(str(output_prefix) + "_3.attributions.h5").exists()
    assert output_prefix.with_suffix(".modisco.h5").exists()
    assert Path(str(output_prefix) + "_report").exists()
