from decima.model.decima_model import DecimaModel


def test_DecimaModel_init():
    model = DecimaModel(n_tasks=7611, mask=True)
    assert model.embedding is not None
    assert model.head is not None

    model = DecimaModel(n_tasks=7611, mask=True, init_borzoi=False, replicate=0)
    assert model.embedding is not None
    assert model.head is not None
