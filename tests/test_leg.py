"""Tests for :mod:`flyplotlib.leg`."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.cm import ScalarMappable

from flyplotlib import Segment, add_muscles, mushow
from flyplotlib.leg import _to_svg_id, svg_ids


@pytest.fixture
def ax():
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


def test_every_segment_has_svg_id():
    assert set(svg_ids) == set(Segment)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Tibia extensor MN", Segment.TIBIA_EXTENSOR),
        ("tibia_extensor", Segment.TIBIA_EXTENSOR),
        ("Pleural remotor/abductor MN", Segment.PLEURAL_REMOTOR_AND_ABDUCTOR),
        ("LTM1", Segment.LONG_TENDON_MUSCLE_1),
        ("Tergotr.", Segment.TERGOTROCHANTER_EXTENSOR),
        (Segment.FEMUR_REDUCTOR, Segment.FEMUR_REDUCTOR),
    ],
)
def test_from_str(text, expected):
    assert Segment.from_str(text) == expected


def test_from_str_unknown_warns():
    with pytest.warns(UserWarning):
        assert Segment.from_str("not a real muscle") is None


def test_to_svg_id_accepts_segment_name_and_id():
    assert _to_svg_id(Segment.TIBIA_EXTENSOR) == "Ti-extensor"
    assert _to_svg_id("tibia_extensor") == "Ti-extensor"
    assert _to_svg_id("Ti-extensor") == "Ti-extensor"  # already an SVG id


def test_to_svg_id_unknown_raises():
    with pytest.raises(KeyError):
        _to_svg_id("totally unknown")


def test_add_muscles_force_patches(ax):
    patches = add_muscles(ax=ax)
    assert isinstance(patches, dict)
    assert "Ti-extensor" in patches


def test_add_muscles_exclude_by_segment(ax):
    full = add_muscles(ax=ax)
    pruned = add_muscles(ax=ax, exclude=[Segment.TIBIA_EXTENSOR])
    assert "Ti-extensor" in full
    assert "Ti-extensor" not in pruned


def test_add_muscles_per_path_kwargs_segment_key(ax):
    patches = add_muscles(
        ax=ax, per_path_kwargs={"facecolor": {Segment.TIBIA_EXTENSOR: "red"}}
    )
    assert patches["Ti-extensor"].get_facecolor()[:3] == (1.0, 0.0, 0.0)


def test_mushow_dict(ax):
    result, mappable = mushow(
        {Segment.TIBIA_EXTENSOR: 0.7, Segment.TARSUS_LEVATOR: -1.0}, ax=ax
    )
    assert isinstance(mappable, ScalarMappable)
    assert isinstance(result, dict)


def test_mushow_iterable(ax):
    values = [0.1] * len(Segment)
    result, mappable = mushow(values, ax=ax)
    assert isinstance(mappable, ScalarMappable)


def test_mushow_all_nan_does_not_crash(ax):
    # No finite values -> default vmin/vmax, every muscle uncolored.
    _result, mappable = mushow({}, ax=ax)
    assert mappable.norm.vmin == 0.0
    assert mappable.norm.vmax == 1.0
