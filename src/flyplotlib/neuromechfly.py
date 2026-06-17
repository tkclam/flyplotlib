"""Draw a posed NeuroMechFly cartoon from the flygym v2 biomechanical model.

:func:`add_neuromechfly` renders a fly in an arbitrary posture: given a set of
joint angles and a view, it computes each body segment's 3D placement via forward
kinematics, projects it orthographically to 2D, and fills the silhouette of each
segment's projected mesh as a matplotlib patch.

The bundled meshes and model metadata (under ``data/neuromechfly/``) are vendored
from `flygym <https://github.com/NeLy-EPFL/flygym>`_ (Apache-2.0); see the
accompanying ``NOTICE`` file. Forward kinematics and STL parsing are implemented
here in numpy; segment silhouettes are dissolved with ``shapely``. No
``flygym``/``mujoco`` dependency is required at runtime.

Conventions (fly-local frame): ``+x`` anterior (forward), ``+y`` left, ``+z``
top (up). Joint DOF names follow flygym's ``"{parent}-{child}-{axis}"`` pattern
with ``axis`` one of ``pitch``/``roll``/``yaw`` (local axes ``y``/``z``/``x``).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from functools import cache, lru_cache
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "add_neuromechfly",
    "neuromechfly_segments",
    "neutral_pose",
    "NEUROMECHFLY_SEGMENTS",
]

# flygym scales mesh vertices by 1000 (m -> mm); rigging offsets are in the scaled
# units, so the meshes must be scaled to match. The absolute scale is irrelevant
# (the output is renormalized to ``length``); only this ratio matters.
_MESH_SCALE = 1000.0

# DOF rotation axes in the body-local (post-quat) frame, and the order in which the
# DOFs at a joint are composed (flygym default AxisOrder PITCH_ROLL_YAW).
_AXIS_VECTOR = {
    "pitch": (0.0, 1.0, 0.0),
    "roll": (0.0, 0.0, 1.0),
    "yaw": (1.0, 0.0, 0.0),
}
_AXIS_ORDER = ("pitch", "roll", "yaw")


# --------------------------------------------------------------------------- #
# Anatomy (kept in sync with flygym.anatomy; see scripts/vendor_*.py)
# --------------------------------------------------------------------------- #
_SIDES = ["l", "r"]
_LEGS = [f"{s}{p}" for s in _SIDES for p in "fmh"]
_LEG_LINKS = ["coxa", "trochanterfemur", "tibia", *(f"tarsus{i}" for i in "12345")]
_ANTENNA_LINKS = ["pedicel", "funiculus", "arista"]
_PROBOSCIS_LINKS = ["rostrum", "haustellum"]
_ABDOMEN_LINKS = ["abdomen12", *(f"abdomen{i}" for i in "3456")]
_ROOT = "c_thorax"


def _chain(*names: str) -> list[tuple[str, str]]:
    return [(names[i], names[i + 1]) for i in range(len(names) - 1)]


def _connected_pairs() -> list[tuple[str, str]]:
    """Parent-child segment pairs, ordered so every parent precedes its children."""
    pairs: list[tuple[str, str]] = [("c_thorax", "c_head")]
    pairs += _chain("c_head", *(f"c_{lk}" for lk in _PROBOSCIS_LINKS))
    pairs += _chain("c_thorax", *(f"c_{lk}" for lk in _ABDOMEN_LINKS))
    pairs += [("c_head", f"{s}_eye") for s in _SIDES]
    for s in _SIDES:
        pairs += _chain("c_head", *(f"{s}_{lk}" for lk in _ANTENNA_LINKS))
    pairs += [("c_thorax", f"{s}_wing") for s in _SIDES]
    pairs += [("c_thorax", f"{s}_haltere") for s in _SIDES]
    for leg in _LEGS:
        pairs += _chain("c_thorax", *(f"{leg}_{lk}" for lk in _LEG_LINKS))
    return pairs


_JOINTS = _connected_pairs()
# Root first, then every child in topological order.
NEUROMECHFLY_SEGMENTS: tuple[str, ...] = (_ROOT, *(child for _, child in _JOINTS))

# Default segments that align by body (excluded when normalizing placement) and
# default segments hidden for a cleaner cartoon, mirroring add_fly's conventions.
_BODY_BBOX_EXCLUDE = (
    r"_(coxa|trochanterfemur|tibia|tarsus\d)$",
    r"_wing$",
    r"_haltere$",
)


def neuromechfly_segments() -> tuple[str, ...]:
    """Return the names of all NeuroMechFly body segments, in kinematic-tree order."""
    return NEUROMECHFLY_SEGMENTS


# --------------------------------------------------------------------------- #
# Bundled data
# --------------------------------------------------------------------------- #
def _data_dir() -> Path:
    return Path(__file__).parent / "data" / "neuromechfly"


@cache
def _rigging() -> dict[str, dict[str, list[float]]]:
    return json.loads((_data_dir() / "rigging.json").read_text())


@cache
def _colors() -> dict[str, list[float]]:
    return json.loads((_data_dir() / "colors.json").read_text())


@cache
def _neutral_pose_deg() -> dict[str, float]:
    """Neutral standing pose in degrees (left/center DOFs only, as in flygym)."""
    return json.loads((_data_dir() / "neutral_pose.json").read_text())


def neutral_pose(degrees: bool = True) -> dict[str, float]:
    """Return the default standing pose as ``{dof_name: angle}`` for all DOFs.

    Left/center angles come from flygym's neutral pose; right-side angles are
    derived by mirroring (roll and yaw negated). Useful for discovering valid DOF
    names or as a base to modify.

    Parameters
    ----------
    degrees : bool, optional
        Return angles in degrees (default) or radians.
    """
    angles_deg = _full_neutral_deg()
    if degrees:
        return dict(angles_deg)
    return {k: np.deg2rad(v) for k, v in angles_deg.items()}


# --------------------------------------------------------------------------- #
# Joint-angle resolution
# --------------------------------------------------------------------------- #
def _mirror_segment(name: str) -> str:
    """Swap a segment to the other side (``lf_coxa`` <-> ``rf_coxa``; ``c_*`` stays)."""
    pos, _, link = name.partition("_")
    if pos[0] in "lr":
        pos = ("r" if pos[0] == "l" else "l") + pos[1:]
    return f"{pos}_{link}"


def _mirror_dof(dof: str) -> tuple[str, float]:
    """Mirror a left DOF name to the right, returning (name, sign for the angle)."""
    parent, child, axis = dof.split("-")
    sign = 1.0 if axis == "pitch" else -1.0
    return f"{_mirror_segment(parent)}-{_mirror_segment(child)}-{axis}", sign


@cache
def _full_neutral_deg() -> dict[str, float]:
    """Neutral pose in degrees for *all* DOFs (left/center + mirrored right)."""
    angles = dict(_neutral_pose_deg())
    for dof, value in _neutral_pose_deg().items():
        # Only DOFs whose child is a left segment have a right mirror image.
        child = dof.split("-")[1]
        if child.split("_")[0].startswith("l"):
            right_dof, sign = _mirror_dof(dof)
            angles[right_dof] = sign * value
    return angles


def _resolve_angles(
    joint_angles: Mapping[str, float] | None, degrees: bool
) -> dict[str, float]:
    """Merge user joint angles (radians out) over the neutral pose."""
    resolved = {k: np.deg2rad(v) for k, v in _full_neutral_deg().items()}
    if joint_angles:
        for dof, value in joint_angles.items():
            resolved[dof] = np.deg2rad(value) if degrees else float(value)
    return resolved


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def _quat_to_matrix(quat: Iterable[float]) -> np.ndarray:
    """Convert a MuJoCo ``(w, x, y, z)`` quaternion to a 3x3 rotation matrix."""
    w, x, y, z = np.asarray(quat, dtype=float)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n == 0:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _axis_angle_matrix(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    """Rotation matrix for a rotation of ``angle`` (rad) about a unit ``axis``."""
    a = np.asarray(axis, dtype=float)
    c, s = np.cos(angle), np.sin(angle)
    cross = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return c * np.eye(3) + s * cross + (1 - c) * np.outer(a, a)


def _homogeneous(rot: np.ndarray, pos: Iterable[float]) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = rot
    t[:3, 3] = pos
    return t


def _joint_rotation(parent: str, child: str, angles: Mapping[str, float]) -> np.ndarray:
    """Composed 3x3 rotation of the DOFs at the (parent, child) joint."""
    rot = np.eye(3)
    for axis in _AXIS_ORDER:
        angle = angles.get(f"{parent}-{child}-{axis}", 0.0)
        if angle:
            rot = rot @ _axis_angle_matrix(_AXIS_VECTOR[axis], angle)
    return rot


def _segment_transforms(angles: Mapping[str, float]) -> dict[str, np.ndarray]:
    """Forward kinematics: world (body-frame) 4x4 transform for every segment."""
    rigging = _rigging()
    transforms: dict[str, np.ndarray] = {}
    root = rigging[_ROOT]
    transforms[_ROOT] = _homogeneous(_quat_to_matrix(root["quat"]), root["pos"])
    for parent, child in _JOINTS:
        cfg = rigging[child]
        rest = _homogeneous(_quat_to_matrix(cfg["quat"]), cfg["pos"])
        joint = _homogeneous(_joint_rotation(parent, child, angles), (0, 0, 0))
        transforms[child] = transforms[parent] @ rest @ joint
    return transforms


# --------------------------------------------------------------------------- #
# Mesh loading (STL)
# --------------------------------------------------------------------------- #
@cache
def _load_left_triangles(mesh_name: str) -> np.ndarray:
    """Load a vendored STL's triangles ``(M, 3, 3)`` (scaled), cached per file.

    Triangle connectivity is kept (vertices are *not* deduplicated) so the
    segment's full projected silhouette can be filled, not just its hull.
    """
    path = _data_dir() / "meshes" / f"{mesh_name}.stl"
    return (_read_stl_vertices(path) * _MESH_SCALE).reshape(-1, 3, 3)


def _segment_triangles(segment: str) -> np.ndarray:
    """Triangles for ``segment`` in its local frame (right side mirrors the left)."""
    if segment[0] == "r":
        tris = _load_left_triangles(_mirror_segment(segment)).copy()
        tris[..., 1] *= -1.0  # mirror across the sagittal plane (flygym: scale y = -1)
        return tris
    return _load_left_triangles(segment)


def _read_stl_vertices(path: Path) -> np.ndarray:
    """Return an ``(N, 3)`` array of triangle vertices from a binary or ASCII STL."""
    data = path.read_bytes()
    if len(data) >= 84:
        n_tri = int(np.frombuffer(data, dtype="<u4", count=1, offset=80)[0])
        if len(data) == 84 + n_tri * 50:  # well-formed binary STL
            dtype = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
            tris = np.frombuffer(data, dtype=dtype, count=n_tri, offset=84)
            return tris["v"].reshape(-1, 3).astype(float)
    return _read_ascii_stl_vertices(data)


def _read_ascii_stl_vertices(data: bytes) -> np.ndarray:
    coords = [
        [float(c) for c in line.split()[1:4]]
        for line in data.decode("ascii", "replace").splitlines()
        if line.strip().startswith("vertex")
    ]
    if not coords:
        raise ValueError("Could not parse STL (no vertices found).")
    return np.asarray(coords, dtype=float)


# --------------------------------------------------------------------------- #
# Projection and 2D silhouette
# --------------------------------------------------------------------------- #
# Camera bases per view, as rows [screen-right, screen-up, toward-viewer] in the
# fly-local frame. Each is a proper rotation (det +1).
_VIEW_MATRICES = {
    "left": np.array([[-1.0, 0, 0], [0, 0, 1], [0, -1, 0]]),
    "right": np.array([[1.0, 0, 0], [0, 0, 1], [0, 1, 0]]),
    "top": np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]]),
    "bottom": np.array([[0.0, 1, 0], [1, 0, 0], [0, 0, -1]]),
    "front": np.array([[0.0, 1, 0], [0, 0, 1], [1, 0, 0]]),
    "back": np.array([[0.0, -1, 0], [0, 0, 1], [-1, 0, 0]]),
}


def _camera_matrix(view: str, rotation: float) -> np.ndarray:
    """3x3 camera matrix for a named view plus an in-plane rotation (degrees)."""
    try:
        view_matrix = _VIEW_MATRICES[view]
    except KeyError:
        raise ValueError(
            f"Unknown view {view!r}; choose from {sorted(_VIEW_MATRICES)}."
        ) from None
    theta = np.deg2rad(rotation)
    c, s = np.cos(theta), np.sin(theta)
    inplane = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])  # about the depth axis
    return inplane @ view_matrix


def _catmull_rom_path(points: np.ndarray):
    """A closed, smooth matplotlib path through ``points`` (``(N, 2)``, unclosed).

    Builds a cyclic Catmull-Rom spline and emits it as the equivalent chain of
    cubic Beziers (matplotlib ``CURVE4`` codes), so the outline is a genuinely
    smooth curve passing through every input vertex -- not a polygon. This uses
    only matplotlib's native path primitives, the same cubic-Bezier representation
    svg2mpl produces from SVG ``C`` commands, so SVG and mesh artwork round their
    corners identically.
    """
    from matplotlib.path import Path as MplPath

    p = np.asarray(points, dtype=float)
    prev, nxt, nxt2 = (np.roll(p, k, axis=0) for k in (1, -1, -2))
    # Uniform Catmull-Rom -> Bezier control points (tangent = (next - prev) / 6).
    c1 = p + (nxt - prev) / 6.0
    c2 = nxt - (nxt2 - p) / 6.0
    # Each segment p[i] -> p[i+1] contributes its two controls and its endpoint.
    segments = np.stack([c1, c2, nxt], axis=1).reshape(-1, 2)
    verts = np.vstack([p[:1], segments])
    codes = np.full(len(verts), MplPath.CURVE4, dtype=np.uint8)
    codes[0] = MplPath.MOVETO
    return MplPath(verts, codes)


def _silhouette_path(triangles_2d: np.ndarray, simplify: float = 0.0):
    """Build the outline path of projected ``(M, 3, 2)`` triangles.

    The triangles are dissolved into their union with shapely, giving the mesh's
    true flat 2D silhouette (its real, possibly concave outline -- not a convex
    hull). Assuming one mesh maps to one body part, only the largest connected
    blob is kept and its interior holes are filled, so the patch is a single solid
    contour with no stray slivers or gaps showing through.

    ``simplify`` is a Douglas-Peucker tolerance (in data units): a positive value
    first collapses the triangle-edge "staircase" along the boundary, then rounds
    the result into a smooth closed Bezier curve via :func:`_catmull_rom_path`.
    ``0`` keeps the raw polygonal outline. Returns ``None`` if the silhouette is
    empty.
    """
    import shapely

    tris = np.asarray(triangles_2d, dtype=float)
    # Keep only one facing half: for a closed mesh either the front- or the
    # back-facing triangles already tile the whole silhouette, so unioning the
    # positive-area (single-facing) subset gives the same outline at ~half the
    # cost. Degenerate (edge-on, ~zero-area) triangles are dropped too.
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    area = (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - (v2[:, 0] - v0[:, 0]) * (
        v1[:, 1] - v0[:, 1]
    )
    tris = tris[area > 1e-12]
    if len(tris) == 0:
        return None

    rings = np.concatenate([tris, tris[:, :1]], axis=1)  # (M, 4, 2), closed
    merged = shapely.union_all(shapely.polygons(rings))
    polygons = [g for g in shapely.get_parts(merged) if g.geom_type == "Polygon"]
    if not polygons:
        return None

    # One mesh -> one patch: keep the largest projected blob and drop its holes,
    # so internal triangle gaps don't show through as stray polygons.
    outline = max(polygons, key=lambda g: g.area).exterior
    if simplify > 0:
        outline = outline.simplify(simplify)
    points = np.asarray(outline.coords)[:-1]  # drop the duplicated closing vertex
    if len(points) < 3:
        return None
    if simplify > 0:
        return _catmull_rom_path(points)

    from matplotlib.path import Path as MplPath

    codes = np.full(len(points) + 1, MplPath.LINETO, dtype=np.uint8)
    codes[0], codes[-1] = MplPath.MOVETO, MplPath.CLOSEPOLY
    return MplPath(np.vstack([points, points[:1]]), codes)


# --------------------------------------------------------------------------- #
# Styling helpers
# --------------------------------------------------------------------------- #
def _matches_any(name: str, patterns: Iterable[str]) -> bool:
    return any(re.search(p, name) for p in patterns)


def _as_patterns(value: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def _segment_color(
    segment: str, overrides: Mapping[str, Any] | None, grayscale: bool
) -> Any:
    color: Any = _colors().get(segment, [0.7, 0.7, 0.7])
    if overrides:
        for pattern, override in overrides.items():
            if re.search(pattern, segment):
                color = override
    if grayscale:
        from matplotlib.colors import to_rgba

        r, g, b, a = to_rgba(color)
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        return (lum, lum, lum, a)
    return color


@lru_cache(maxsize=16)
def _base_silhouettes(angles_items: tuple, view: str, smoothing: float):
    """Per-segment silhouette outlines in the base (un-rotated, un-scaled) view.

    This is the expensive step (forward kinematics + per-segment shapely union),
    so it is cached. The in-plane rotation, scaling and translation requested by
    :func:`add_neuromechfly` are pure 2D affines applied afterwards to the returned
    paths, and so are deliberately excluded from the cache key. Returns
    ``(paths, depths, lo, span)`` where ``paths`` maps each segment to a
    :class:`~matplotlib.path.Path` (or ``None`` if empty), ``depths`` to its mean
    camera depth, and ``lo``/``span`` describe the body bounding box.
    """
    angles = dict(angles_items)
    transforms = _segment_transforms(angles)
    camera = _camera_matrix(view, 0)  # base orientation; rotation applied later

    projected: dict[str, np.ndarray] = {}
    depths: dict[str, float] = {}
    for segment in NEUROMECHFLY_SEGMENTS:
        tris = _segment_triangles(segment).reshape(-1, 3)
        cam = (np.c_[tris, np.ones(len(tris))] @ transforms[segment].T)[
            :, :3
        ] @ camera.T
        projected[segment] = cam[:, :2].reshape(-1, 3, 2)
        depths[segment] = float(cam[:, 2].mean())

    body_pts = np.vstack(
        [
            projected[s].reshape(-1, 2)
            for s in NEUROMECHFLY_SEGMENTS
            if not _matches_any(s, _BODY_BBOX_EXCLUDE)
        ]
    )
    lo = body_pts.min(axis=0)
    span = body_pts.max(axis=0) - lo
    tol = smoothing * span.max()
    paths = {s: _silhouette_path(projected[s], tol) for s in NEUROMECHFLY_SEGMENTS}
    return paths, depths, lo, span


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def add_neuromechfly(
    joint_angles: Mapping[str, float] | None = None,
    *,
    view: str = "right",
    xy: tuple[float, float] = (0, 0),
    rotation: float = 0,
    length: float | None = 3,
    origin: tuple[float, float] = (0.5, 0.5),
    degrees: bool = True,
    exclude: str | Iterable[str] = (),
    colors: Mapping[str, Any] | None = None,
    smoothing: float = 0.01,
    grayscale: bool = False,
    edgecolor: Any = "black",
    linewidth: float = 0.5,
    ax=None,
    **patch_kwargs: Any,
) -> dict[str, Any]:
    """Add a posed NeuroMechFly cartoon to a matplotlib axes.

    Each body segment is placed by forward kinematics from ``joint_angles``,
    projected orthographically for the chosen ``view``, and drawn as a filled
    :class:`~matplotlib.patches.PathPatch` outlining the silhouette of its projected
    mesh. Segments are z-ordered by depth so that nearer parts overlay farther
    ones.

    Parameters
    ----------
    joint_angles : Mapping[str, float], optional
        Mapping of DOF name (``"{parent}-{child}-{axis}"``, e.g.
        ``"c_thorax-lf_coxa-pitch"``) to angle. Unspecified DOFs fall back to the
        neutral standing pose, so ``None`` (the default) draws a neutral fly. See
        :func:`neutral_pose` for the full list of DOF names.
    view : str, optional
        Anatomical view: ``"left"``, ``"right"`` (default), ``"front"``, ``"back"``,
        ``"top"``, or ``"bottom"``.
    xy : tuple, optional
        Position of the fly in data coordinates. The default is ``(0, 0)``.
    rotation : float, optional
        In-plane rotation in degrees, applied after projection. The default is
        ``0``.
    length : float or None, optional
        Approximate on-screen body size in data units (largest dimension of the
        projected body bounding box). The default is ``3``. If ``None``, no
        rescaling is applied (raw model units, ~millimetres).
    origin : tuple, optional
        Anchor point within the projected body bounding box, in normalized
        coordinates. The default is ``(0.5, 0.5)`` (the body center).
    degrees : bool, optional
        Whether ``joint_angles`` are in degrees (default) or radians. Pass
        ``degrees=False`` to feed flygym simulation output directly.
    exclude : str or Iterable[str], optional
        Regular expressions matched against segment names; matching segments are
        hidden (e.g. ``"wing"``, ``"r._"`` to drop the right legs).
    colors : Mapping[str, Any], optional
        Per-segment color overrides. Keys are regular expressions matched against
        segment names; the last matching entry wins. Values are any matplotlib
        color.
    smoothing : float, optional
        Outline smoothing as a fraction of the body size. Each silhouette's
        triangle-edge "staircase" is collapsed at this tolerance and the boundary
        is then redrawn as a smooth closed Bezier curve, so segments read as
        rounded shapes rather than faceted polygons. The default is ``0.01``; set
        ``0`` to disable and keep the raw mesh outline.
    grayscale : bool, optional
        Convert all colors to grayscale if ``True``. The default is ``False``.
    edgecolor, linewidth : optional
        Outline color and width for every segment, drawn by default
        (``"black"``/``0.5``). Pass ``edgecolor="none"`` (or ``linewidth=0``) to
        disable the outlines.
    ax : Axes, optional
        The axes to draw on. Defaults to the current axes.
    **patch_kwargs
        Additional keyword arguments forwarded to every
        :class:`~matplotlib.patches.PathPatch` (e.g. ``alpha``).

    Returns
    -------
    dict
        Mapping of segment name to the added
        :class:`~matplotlib.patches.PathPatch`, in draw (depth) order.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import PathPatch
    from matplotlib.transforms import Affine2D

    if ax is None:
        ax = plt.gca()

    exclude = _as_patterns(exclude)
    angles = _resolve_angles(joint_angles, degrees)
    # Heavy step (FK + per-segment silhouette union), cached and computed in the
    # base orientation; placement below is a cheap 2D affine applied to the result.
    base_paths, depths, lo, span = _base_silhouettes(
        tuple(sorted(angles.items())), view, float(smoothing)
    )

    segments = [
        s
        for s in NEUROMECHFLY_SEGMENTS
        if base_paths[s] is not None and not _matches_any(s, exclude)
    ]
    if not segments:
        return {}

    # Place/scale/rotate the fly: align by the body bounding box, like add_fly.
    scale = 1.0 if length is None else length / max(span.max(), 1e-9)
    anchor = lo + np.asarray(origin) * span
    affine = (
        Affine2D()
        .translate(-anchor[0], -anchor[1])
        .rotate_deg(rotation)
        .scale(scale)
        .translate(*xy)
    )

    # Build a patch per segment, drawn back-to-front by mean depth.
    dmin = min(depths[s] for s in segments)
    dmax = max(depths[s] for s in segments)
    drange = dmax - dmin or 1.0

    patches: dict[str, Any] = {}
    for segment in sorted(segments, key=lambda s: depths[s]):
        patch = PathPatch(
            base_paths[segment].transformed(affine),
            facecolor=_segment_color(segment, colors, grayscale),
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=2 + (depths[segment] - dmin) / drange,
            **patch_kwargs,
        )
        # add_artist (not add_patch) skips matplotlib's per-segment Bezier limit
        # scan; we update the data limits once from the overall bounding box below.
        ax.add_artist(patch)
        patches[segment] = patch

    drawn = np.vstack([p.get_path().vertices for p in patches.values()])
    ax.update_datalim([drawn.min(axis=0), drawn.max(axis=0)])
    return patches
