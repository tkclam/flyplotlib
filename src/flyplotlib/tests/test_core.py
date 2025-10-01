"""Tests for flyplotlib.core module."""

import numpy as np
from unittest.mock import MagicMock
from matplotlib.transforms import Bbox
from pathlib import Path
from flyplotlib.core import (
    get_data_dir,
    get_bbox,
    get_affine_matrix,
    add_fly,
)


def test_get_data_dir():
    data_dir = get_data_dir()
    assert isinstance(data_dir, Path)
    assert data_dir.name == "data"


def test_get_collection_bbox():
    mock_path = MagicMock()
    mock_path.get_extents.return_value = Bbox([[0, 0], [1, 1]])
    mock_collection = MagicMock()
    mock_collection.get_paths.return_value = [mock_path]
    bbox = get_bbox(mock_collection)
    assert bbox.bounds == (0, 0, 1, 1)


def test_get_affine_matrix():
    bbox = Bbox([[-1, -1], [1, 1]])

    affine_matrix = get_affine_matrix(
        bbox, source_xy=(0.5, 0.5), target_xy=(0, 0), rotation=0, width=2, height=2
    ).get_matrix()

    assert np.allclose(affine_matrix, np.eye(3))

    affine_matrix = get_affine_matrix(
        bbox, source_xy=(0.5, 0.5), target_xy=(0, 0), rotation=90, width=2, height=2
    ).get_matrix()

    assert np.allclose(affine_matrix, np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]))


def test_add_fly():
    from matplotlib.collections import PathCollection

    mock_ax = MagicMock()
    fly = add_fly(ax=mock_ax)
    # The result should be either a PathCollection or a dict of patches
    assert isinstance(fly, (PathCollection, dict))
    mock_ax.add_collection.assert_called_with(fly)
