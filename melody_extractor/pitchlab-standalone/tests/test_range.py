import numpy as np

from pitchlab.models.base import coerce_frequency_range, f0_dataframe
from pitchlab.registry import available_models, get_model


def test_range_aliases():
    assert coerce_frequency_range(range=(100, 500)).as_tuple() == (100.0, 500.0)
    assert coerce_frequency_range(freq_range="100 500").as_tuple() == (100.0, 500.0)
    assert coerce_frequency_range(fmin=100, fmax=500).as_tuple() == (100.0, 500.0)
    assert coerce_frequency_range(min_frequency=100, max_frequency=500).as_tuple() == (100.0, 500.0)


def test_f0_filter():
    rng = coerce_frequency_range((100, 500))
    df = f0_dataframe(
        times=[0, 1, 2, 3],
        frequency_hz=[50, 110, 499, 800],
        confidence=[1, 1, 1, 1],
        model="test",
        freq_range=rng,
    )
    assert list(df["frequency_hz"]) == [0.0, 110.0, 499.0, 0.0]


def test_registry_has_core_models():
    models = set(available_models())
    assert "torchcrepe" in models
    assert "crepe" in models
    assert "basic-pitch" in models
    assert "librosa-pyin" in models
    assert get_model("pyin").name == "librosa-pyin"
