"""Tests for :mod:`flyplotlib.neuromechfly`."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba
from matplotlib.patches import PathPatch

from flyplotlib import (
    NEUROMECHFLY_SEGMENTS,
    add_neuromechfly,
    neuromechfly_segments,
    neutral_pose,
)
from flyplotlib.neuromechfly import (
    _data_dir,
    _load_left_triangles,
    _segment_triangles,
    _silhouette_path,
)


@pytest.fixture
def ax():
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


def test_assets_present():
    data = _data_dir()
    assert (data / "rigging.json").exists()
    assert (data / "neutral_pose.json").exists()
    assert (data / "colors.json").exists()
    assert (data / "meshes" / "c_thorax.stl").exists()


def test_segments_count_and_membership():
    segs = neuromechfly_segments()
    assert segs is NEUROMECHFLY_SEGMENTS
    assert len(segs) == 69  # 6 legs x 8 + head/thorax/proboscis/abdomen/eyes/...
    assert segs[0] == "c_thorax"
    assert {"c_head", "lf_tibia", "rf_tibia", "l_wing", "c_abdomen6"} <= set(segs)


def test_returns_patches_added_to_axes(ax):
    patches = add_neuromechfly(ax=ax)
    assert isinstance(patches, dict)
    assert len(patches) == len(NEUROMECHFLY_SEGMENTS)
    assert all(isinstance(p, PathPatch) for p in patches.values())
    assert all(p in ax.patches for p in patches.values())


def test_neutral_pose_default(ax):
    # joint_angles=None draws the neutral standing pose (all segments present).
    default = add_neuromechfly(ax=ax)
    explicit = add_neuromechfly(ax=ax, joint_angles=neutral_pose())
    for seg in default:
        np.testing.assert_allclose(
            default[seg].get_path().vertices, explicit[seg].get_path().vertices
        )


def test_neutral_pose_mirrors_roll_yaw():
    pose = neutral_pose()
    # Pitch is preserved across sides; roll/yaw are negated for the right side.
    assert pose["c_thorax-rf_coxa-pitch"] == pose["c_thorax-lf_coxa-pitch"]
    assert pose["c_thorax-rf_coxa-roll"] == -pose["c_thorax-lf_coxa-roll"]
    assert pose["c_thorax-rf_coxa-yaw"] == -pose["c_thorax-lf_coxa-yaw"]


def test_exclude_patterns(ax):
    full = add_neuromechfly(ax=ax)
    pruned = add_neuromechfly(ax=ax, exclude=["wing", "haltere", "r._"])
    assert len(pruned) < len(full)
    assert not any("wing" in s or "haltere" in s for s in pruned)
    assert not any(s[:2] in ("rf", "rm", "rh") for s in pruned)  # right legs gone
    assert "lf_tibia" in pruned  # left legs kept


def test_color_override_and_default(ax):
    patches = add_neuromechfly(ax=ax, colors={"c_head": "red"})
    assert tuple(patches["c_head"].get_facecolor()) == to_rgba("red")
    # Default color comes from the model (head/thorax base brown ~ (.59,.39,.12)).
    fc = patches["c_thorax"].get_facecolor()
    np.testing.assert_allclose(fc[:3], (0.59, 0.39, 0.12), atol=1e-6)


def test_black_edges_on_by_default_and_disable(ax):
    default = add_neuromechfly(ax=ax)
    assert all(tuple(p.get_edgecolor()) == to_rgba("black") for p in default.values())
    assert all(p.get_linewidth() > 0 for p in default.values())
    # Disable via edgecolor="none" -> fully transparent edge.
    none_edges = add_neuromechfly(ax=ax, edgecolor="none")
    assert all(p.get_edgecolor()[3] == 0 for p in none_edges.values())


def test_smoothing_reduces_vertices(ax):
    raw = add_neuromechfly(ax=ax, smoothing=0)
    smoothed = add_neuromechfly(ax=ax, smoothing=0.03)

    def total_vertices(patches):
        return sum(len(p.get_path().vertices) for p in patches.values())

    assert total_vertices(smoothed) < total_vertices(raw)


def test_grayscale(ax):
    patches = add_neuromechfly(ax=ax, grayscale=True)
    for patch in patches.values():
        r, g, b, _ = patch.get_facecolor()
        assert np.isclose(r, g) and np.isclose(g, b)


def test_degrees_flag_equivalence(ax):
    deg = add_neuromechfly(ax=ax, joint_angles=neutral_pose(degrees=True))
    rad = add_neuromechfly(
        ax=ax, joint_angles=neutral_pose(degrees=False), degrees=False
    )
    np.testing.assert_allclose(
        deg["lf_tibia"].get_path().vertices, rad["lf_tibia"].get_path().vertices
    )


@pytest.mark.parametrize("view", ["left", "right", "front", "back", "top", "bottom"])
def test_views_produce_finite_silhouettes(ax, view):
    patches = add_neuromechfly(ax=ax, view=view)
    assert len(patches) > 50
    for patch in patches.values():
        verts = patch.get_path().vertices
        assert len(verts) >= 4  # at least one triangle (4 path vertices)
        assert np.isfinite(verts).all()


def test_unknown_view_raises(ax):
    with pytest.raises(ValueError, match="Unknown view"):
        add_neuromechfly(ax=ax, view="sideways")


def test_fk_right_orientation(ax):
    # right view, neutral pose: head is anterior (+screen x) of the thorax and the
    # front tibia hangs ventrally (-screen y) below it. Guards the FK conventions
    # (DOF axis order and right-side sign flip).
    patches = add_neuromechfly(ax=ax, view="right")

    def centroid(seg):
        return patches[seg].get_path().vertices.mean(axis=0)

    thorax, head, tibia = centroid("c_thorax"), centroid("c_head"), centroid("lf_tibia")
    assert head[0] > thorax[0]
    assert tibia[1] < thorax[1]


def test_placement_length_and_xy(ax):
    patches = add_neuromechfly(ax=ax, xy=(10, 20), length=4)
    verts = np.vstack([p.get_path().vertices for p in patches.values()])
    # The fly is centered near xy and roughly `length` units across.
    assert 8 < verts[:, 0].mean() < 12
    assert 18 < verts[:, 1].mean() < 22
    span = verts.max(axis=0) - verts.min(axis=0)
    assert 3 < span.max() < 6  # body ~length, legs extend a bit beyond


def test_segment_triangles_shape_and_mirror():
    left = _load_left_triangles("lf_coxa")
    assert left.ndim == 3 and left.shape[1:] == (3, 3)  # (M, 3 verts, xyz)
    right = _segment_triangles("rf_coxa")
    # The right side reuses the left mesh, mirrored across the sagittal (y) plane.
    np.testing.assert_allclose(right[..., 0], left[..., 0])
    np.testing.assert_allclose(right[..., 1], -left[..., 1])
    np.testing.assert_allclose(right[..., 2], left[..., 2])


def test_silhouette_follows_mesh_not_hull():
    # Two disjoint triangles: the silhouette is their union (a concave region), so
    # a point in the gap between them is *outside* it -- unlike a convex hull, which
    # would bridge the gap and contain that point.
    triangles = np.array(
        [
            [[0, 0], [1, 0], [0, 1]],
            [[3, 0], [4, 0], [3, 1]],
        ],
        dtype=float,
    )
    path = _silhouette_path(triangles)
    assert path.contains_point((0.2, 0.2))  # inside the first triangle
    assert path.contains_point((3.3, 0.2))  # inside the second triangle
    assert not path.contains_point((2.0, 0.2))  # in the gap (a hull would include it)
