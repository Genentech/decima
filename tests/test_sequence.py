
from decima.utils.sequence import prepare_mask_gene


def test_mask_gene():
    mask = prepare_mask_gene(100, 200)
    assert mask.shape == (1, 524288)
    assert mask[0, 150].item() == 1.0
