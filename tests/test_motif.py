import numpy as np
from decima.utils.motifs import information_content, information_content_per_position, trim_ppm, trim_attributions, motif_start_end


def test_information_content():
    ppm = np.array([[0.9, 0.25], [0.05, 0.25], [0.025, 0.25], [0.025, 0.25]])

    ic = information_content(ppm)
    assert ic.shape == (4, 2)


def test_information_content_per_position():
    ppm = np.array([[0.8, 0.25, 0.1], [0.1, 0.25, 0.8], [0.05, 0.25, 0.05], [0.05, 0.25, 0.05]])

    ic_per_pos = information_content_per_position(ppm)
    assert ic_per_pos.shape == (3,)
    assert ic_per_pos[0] > ic_per_pos[1]
    assert ic_per_pos[2] > ic_per_pos[1]


def test_trim_ppm():
    ppm = np.array([
        [0.25, 0.8, 0.9, 0.25],
        [0.25, 0.1, 0.05, 0.25],
        [0.25, 0.05, 0.025, 0.25],
        [0.25, 0.05, 0.025, 0.25]
    ])

    trimmed = trim_ppm(ppm, trim_threshold=0.3)
    assert trimmed.shape == (4, 2)


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
