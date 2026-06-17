"""Tests for :mod:`flyplotlib.core` (the fly-specific layer over svg2mpl).

Generic SVG-rendering behavior (bounding boxes, affine placement, the path
caching, vectorization, ...) is tested in the svg2mpl package; here we only
exercise the fly defaults and the bundled-asset path resolution.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PathCollection
from matplotlib.patches import PathPatch

from flyplotlib import add_fly, add_paths
from flyplotlib.core import get_data_dir


@pytest.fixture
def ax():
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


def test_get_data_dir():
    data_dir = get_data_dir()
    assert data_dir.name == "data"
    assert (data_dir / "fly.svg").exists()


def test_add_fly_returns_collection(ax):
    result = add_fly(ax=ax)
    assert isinstance(result, PathCollection)
    assert result in ax.collections


def test_add_fly_force_patches(ax):
    result = add_fly(ax=ax, force_patches=True)
    assert isinstance(result, dict)
    assert all(isinstance(p, PathPatch) for p in result.values())


def test_add_fly_grayscale_does_not_crash(ax):
    # Regression: grayscale must not crash on paths with fill="none".
    result = add_fly(ax=ax, grayscale=True, force_patches=True)
    for patch in result.values():
        fc = patch.get_facecolor()
        assert np.isclose(fc[0], fc[1]) and np.isclose(fc[1], fc[2])


def test_add_fly_exclude(ax):
    full = add_fly(ax=ax, force_patches=True)
    pruned = add_fly(ax=ax, force_patches=True, exclude="leg")
    assert len(pruned) < len(full)
    assert not any(k.startswith("leg") for k in pruned)


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
