from decima.constants import DECIMA_CONTEXT_SIZE
from decima.utils.sequence import prepare_mask_gene


def test_mask_gene():
    mask = prepare_mask_gene(100, 200)
    assert mask.shape == (1, DECIMA_CONTEXT_SIZE)
    assert mask[0, 150].item() == 1.0
