import numpy as np


def motif_start_end(attributions: np.ndarray, motif: np.ndarray):
    """
    Get the start and end of the motif attributions and motif matrix and returns start and end positions for the window with maximum score.

    Args:
        attributions: Attribution scores with shape (batch_size, seqlet_len, 4)
        motif: Motif matrix with shape (motif_len, 4)

    Returns:
        [np.ndarray, np.ndarray]: Start and end positions of the motif with shape (batch_size, 2)
    """

    assert len(attributions.shape) == 3, "`attributions` must be a 3D array where (batch_size, seqlet_len, 4)"
    assert len(motif.shape) == 2, "`motif` must be a 2D array where (motif_len, 4)"

    attrs = np.concatenate([attributions, np.zeros((attributions.shape[0], motif.shape[0], 4))], axis=1)
    attrs = np.lib.stride_tricks.sliding_window_view(attrs, (attributions.shape[0], motif.shape[0], 4)).squeeze(
        axis=(0, 2)
    )
    start = (
        np.einsum(
            "ibjk,jk->bi",
            attrs,
            motif,
        )
        .argmax(axis=1)
        .astype(int)
    )
    end = start + motif.shape[0]
    return start, end
