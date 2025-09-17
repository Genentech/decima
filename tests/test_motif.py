import numpy as np
from decima.utils.motifs import motif_start_end


def test_motif_start_end():
    attributions = np.array([
        [[0.1, 0.0, 0.0, 0.0],
         [0.2, 0.1, 0.0, 0.0],
         [0.8, 0.1, 0.05, 0.05],
         [0.9, 0.05, 0.025, 0.025],
         [0.1, 0.0, 0.0, 0.0]]
    ])
    motif = np.array([[0.9, 0.05, 0.025, 0.025], [0.05, 0.9, 0.025, 0.025]])

    start, end = motif_start_end(attributions, motif)
    assert start == np.array([2])
    assert end == np.array([4])
