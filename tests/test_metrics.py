import torch
from decima.model.metrics import WarningCounter, WarningType


def test_warning_counter_init():
    counter = WarningCounter()

    assert len(counter.warning_types) == 2
    assert WarningType.UNKNOWN in counter.warning_types
    assert WarningType.ALLELE_MISMATCH_WITH_REFERENCE_GENOME in counter.warning_types

    assert torch.all(counter.counts == 0)


def test_warning_counter_update():
    counter = WarningCounter()

    warnings = [
        WarningType.ALLELE_MISMATCH_WITH_REFERENCE_GENOME,
        WarningType.UNKNOWN,
        WarningType.ALLELE_MISMATCH_WITH_REFERENCE_GENOME
    ]
    counter.update(warnings)

    result = counter.compute()
    assert result["allele_mismatch_with_reference_genome"] == 2
    assert result["unknown"] == 1

    counter = WarningCounter()

    counter.update([])

    result = counter.compute()
    assert result["unknown"] == 0
    assert result["allele_mismatch_with_reference_genome"] == 0


def test_warning_counter_reset():
    counter = WarningCounter()

    counter.update([WarningType.UNKNOWN, WarningType.UNKNOWN])

    result = counter.compute()
    assert result["unknown"] == 2

    counter.reset()
    result = counter.compute()
    assert result["unknown"] == 0
    assert result["allele_mismatch_with_reference_genome"] == 0
