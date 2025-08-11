import numpy as np
import pytest
from unittest.mock import MagicMock
from matplotlib.transforms import Bbox
from pathlib import Path
from flyplotlib.core import (
    get_data_dir,
    get_collection_bbox,
    get_affine_matrix,
    FlyPathCollection,
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
    bbox = get_collection_bbox(mock_collection)
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


def test_flypathcollection_initialization():
    try:
        FlyPathCollection()
    except Exception as e:
        pytest.fail(f"Initialization failed: {e}")


def test_add_fly():
    mock_ax = MagicMock()
    fly = add_fly(ax=mock_ax)
    assert isinstance(fly, FlyPathCollection)
    mock_ax.add_collection.assert_called_with(fly)
