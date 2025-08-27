import numpy as np


def information_content(ppm, background=None, pseudocount=1e-5):
    """Compute the information content per position of a position weight matrix (PPM).

    Args:
        ppm: Position weight matrix (PPM) with shape (4, seq_len)
        background: Background frequency of the alphabet. Defaults to uniform background.
        pseudocount: Pseudocount to avoid log(0). Defaults to 1e-5.

    Returns:
        np.ndarray: Information content per position of the PPM.
    """
    if background is None:
        background = np.array([[0.25], [0.25], [0.25], [0.25]])

    alphabet_len = len(background)
    return (np.log((ppm + pseudocount) / (1 + pseudocount * alphabet_len)) / np.log(2)) * ppm - (
        np.log(background) * background / np.log(2)
    )


def information_content_per_position(ppm, background=None, pseudocount=1e-5):
    """Compute the information content per position of a position weight matrix (PPM).

    Args:
        ppm: Position weight matrix (PPM) with shape (4, seq_len)
        background: Background frequency of the alphabet. Defaults to uniform background.
        pseudocount: Pseudocount to avoid log(0). Defaults to 1e-5.

    Returns:
        np.ndarray: Information content per position of the PPM with shape (seq_len,).
    """
    return information_content(ppm, background, pseudocount).sum(axis=0)


def trim_ppm(ppm, background=None, trim_threshold=0.2, pseudocount=1e-5):
    """Trim the PPM to the positions with the highest information content.

    Args:
        ppm: Position weight matrix (PPM) with shape (4, seq_len)
        background: Background frequency of the alphabet. Defaults to uniform background.
        trim_threshold: Threshold for trimming the PPM. Defaults to 0.2.
        pseudocount: Pseudocount to avoid log(0). Defaults to 1e-5.

    Returns:
        np.ndarray: Trimmed PPM with shape (4, seq_len_trimmed).
    """
    ic = information_content_per_position(ppm, background, pseudocount)
    return ppm[:, ic > trim_threshold * ic.max()]


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
