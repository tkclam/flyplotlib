"""Tests for :mod:`flyplotlib.core`."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PathCollection
from matplotlib.patches import PathPatch
from matplotlib.transforms import Bbox

from flyplotlib import add_fly, add_paths
from flyplotlib.core import (
    get_affine_matrix,
    get_bbox,
    get_data_dir,
    get_is_included,
    get_path_dicts,
    parse_svg,
    vectorize_dicts,
)


@pytest.fixture
def ax():
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


def test_get_data_dir():
    data_dir = get_data_dir()
    assert data_dir.name == "data"
    assert (data_dir / "fly.svg").exists()


def test_get_bbox():
    paths = [
        type("P", (), {"get_extents": lambda self: Bbox([[0, 1], [2, 3]])})(),
        type("P", (), {"get_extents": lambda self: Bbox([[-1, 0], [1, 2]])})(),
    ]
    bbox = get_bbox(paths)
    assert bbox.bounds == (-1, 0, 3, 3)  # (x0, y0, width, height)


def test_get_bbox_empty_raises():
    with pytest.raises(ValueError):
        get_bbox([])


def test_get_affine_matrix_identity():
    bbox = Bbox([[-1, -1], [1, 1]])
    m = get_affine_matrix(
        bbox, source_xy=(0.5, 0.5), target_xy=(0, 0), rotation=0, width=2, height=2
    ).get_matrix()
    assert np.allclose(m, np.eye(3))


def test_get_affine_matrix_rotation():
    bbox = Bbox([[-1, -1], [1, 1]])
    m = get_affine_matrix(
        bbox, source_xy=(0.5, 0.5), target_xy=(0, 0), rotation=90, width=2, height=2
    ).get_matrix()
    assert np.allclose(m, [[0, -1, 0], [1, 0, 0], [0, 0, 1]])


def test_get_affine_matrix_scalar_scale():
    bbox = Bbox([[0, 0], [1, 1]])
    m = get_affine_matrix(
        bbox,
        source_xy=(0, 0),
        target_xy=(0, 0),
        rotation=0,
        width=None,
        height=None,
        scale=2,
    ).get_matrix()
    assert np.allclose(m, [[2, 0, 0], [0, 2, 0], [0, 0, 1]])


def test_get_is_included():
    assert get_is_included("leg_1", exclude=())
    assert not get_is_included("leg_1", exclude="leg")
    assert not get_is_included("leg_1", exclude=["wing", "leg"])
    assert get_is_included("wing_1", exclude=["leg"])


def test_parse_svg_is_cached():
    a = parse_svg(get_data_dir() / "fly.svg")
    b = parse_svg(get_data_dir() / "fly.svg")
    assert a is b  # identical object -> served from cache
    assert len(a) > 0
    _id, props = a[0]
    assert set(props) == {"path", "facecolor", "edgecolor", "linewidth", "alpha"}


def test_get_path_dicts_grayscale():
    for d in get_path_dicts(grayscale=True).values():
        fc = d["facecolor"]
        if fc != "none":
            assert np.allclose(fc[0], fc[1]) and np.allclose(fc[1], fc[2])


def test_get_path_dicts_exclude():
    full = get_path_dicts()
    pruned = get_path_dicts(exclude="leg")
    assert len(pruned) < len(full)
    assert not any(k.startswith("leg") for k in pruned)


def test_vectorize_dicts_uniform():
    out = vectorize_dicts({"a": {"x": 1, "y": 2}, "b": {"x": 3, "y": 4}})
    assert out == {"x": [1, 3], "y": [2, 4]}


def test_vectorize_dicts_nonuniform_returns_none():
    assert vectorize_dicts({"a": {"x": 1}, "b": {"x": 1, "y": 2}}) is None


def test_vectorize_dicts_empty_returns_none():
    assert vectorize_dicts({}) is None


def test_add_fly_returns_collection(ax):
    result = add_fly(ax=ax)
    assert isinstance(result, PathCollection)
    assert result in ax.collections


def test_add_fly_force_patches(ax):
    result = add_fly(ax=ax, force_patches=True)
    assert isinstance(result, dict)
    assert all(isinstance(p, PathPatch) for p in result.values())


def test_add_fly_grayscale(ax):
    # Regression: grayscale must not crash on paths with fill="none".
    add_fly(ax=ax, grayscale=True)


def test_add_fly_per_path_facecolor_stays_a_collection(ax):
    # Per-path values for an existing property keep the key set uniform, so the
    # fast PathCollection path is still used (each path gets its own color).
    result = add_fly(ax=ax, per_path_kwargs={"facecolor": {"body": "red"}})
    assert isinstance(result, PathCollection)


def test_add_fly_per_path_new_key_falls_back_to_patches(ax):
    # A per-path property present on only some paths cannot be vectorized.
    result = add_fly(ax=ax, per_path_kwargs={"visible": {"body": False}})
    assert isinstance(result, dict)
    assert result["body"].get_visible() is False


def test_add_paths_missing_absolute_file_raises():
    with pytest.raises(FileNotFoundError):
        add_paths("/nonexistent/path/to/file.svg")


def test_add_paths_relative_resolves_to_data_dir(ax):
    # "fly.svg" does not exist relative to cwd but lives in the data dir.
    result = add_paths("fly.svg", ax=ax)
    assert isinstance(result, PathCollection | dict)
