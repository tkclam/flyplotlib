"""Core functionality for plotting fly anatomical structures from SVG files."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from functools import cache
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PathCollection
from matplotlib.colors import to_rgb
from matplotlib.patches import PathPatch
from matplotlib.transforms import Affine2D, Bbox
from svgpath2mpl import parse_path

_SVG_NS = "{http://www.w3.org/2000/svg}"

# Mapping of SVG attribute -> (matplotlib key, default, converter). Defined once
# at import time so parsing stays a tight loop.
_SVG_ATTRS = (
    ("d", "path", "", parse_path),
    ("fill", "facecolor", "none", None),
    ("stroke", "edgecolor", "none", None),
    ("stroke-width", "linewidth", "0", float),
    ("opacity", "alpha", "1", lambda x: min(max(float(x), 0.0), 1.0)),
)


def get_data_dir() -> Path:
    """Return the path to the bundled SVG data directory.

    Returns
    -------
    Path
        The path to the ``data`` directory shipped with the package.
    """
    return Path(__file__).parent / "data"


def get_bbox(paths: Iterable[Any]) -> Bbox:
    """Return the bounding box enclosing a collection of paths.

    Parameters
    ----------
    paths : Iterable
        Any iterable of objects exposing a ``get_extents()`` method (e.g.
        matplotlib ``Path`` objects).

    Returns
    -------
    Bbox
        The bounding box that contains every path.

    Raises
    ------
    ValueError
        If ``paths`` is empty.
    """
    extents = np.array([p.get_extents() for p in paths])
    if extents.size == 0:
        raise ValueError("Cannot compute a bounding box of an empty collection.")
    mins = extents[:, 0].min(axis=0)
    maxs = extents[:, 1].max(axis=0)
    return Bbox([mins, maxs])


def get_affine_matrix(
    bbox: Bbox,
    source_xy: tuple[float, float],
    target_xy: tuple[float, float],
    rotation: float,
    width: float | None,
    height: float | None,
    scale: tuple[float, float] | float = (1.0, 1.0),
) -> Affine2D:
    """Return an affine transformation that places ``bbox`` in data coordinates.

    Parameters
    ----------
    bbox : Bbox
        The bounding box of the object to transform.
    source_xy : tuple
        The anchor point within the bounding box, in normalized coordinates.
        ``(0, 0)`` is the lower-left corner, ``(1, 0)`` the lower-right corner
        and ``(0, 1)`` the upper-left corner.
    target_xy : tuple
        The point in data coordinates that ``source_xy`` is moved to.
    rotation : float
        The rotation angle in degrees.
    width : float or None
        The final width of the object. If ``None``, the object is scaled
        according to ``height`` to preserve its aspect ratio.
    height : float or None
        The final height of the object. If ``None``, the object is scaled
        according to ``width`` to preserve its aspect ratio.
    scale : float or tuple of float, optional
        An additional scaling factor. A single float is applied to both axes.
        The default is ``(1.0, 1.0)``.

    Returns
    -------
    Affine2D
        The affine transformation.
    """
    (x0, y0), (x1, y1) = np.asarray(bbox)
    w, h = x1 - x0, y1 - y0

    affine2d = Affine2D()
    affine2d.translate(-x0 - w * source_xy[0], -y0 - h * source_xy[1])

    if width is not None and height is not None:
        affine2d.scale(width / w, height / h)
    elif width is not None:
        affine2d.scale(width / w)
    elif height is not None:
        affine2d.scale(height / h)

    if np.ndim(scale) == 0:
        affine2d.scale(scale, scale)
    else:
        affine2d.scale(*scale)

    affine2d.rotate_deg(rotation)
    affine2d.translate(*target_xy)
    return affine2d


def get_is_included(id_: str, exclude: str | Iterable[str]) -> bool:
    """Return whether ``id_`` survives the ``exclude`` regular expressions.

    Parameters
    ----------
    id_ : str
        The path ID to test.
    exclude : str or Iterable[str]
        One or more regular expressions. ``id_`` is excluded if any pattern
        matches it (using :func:`re.match`).

    Returns
    -------
    bool
        ``True`` if the ID should be kept, ``False`` if it matches an
        exclusion pattern.
    """
    if isinstance(exclude, str):
        exclude = (exclude,)
    return not any(re.match(r, id_) for r in exclude)


def _to_grayscale(color: Any) -> tuple[float, float, float]:
    """Convert a matplotlib color to its luminance-preserving gray equivalent."""
    r, g, b = to_rgb(color)
    v = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return (v, v, v)


@cache
def _parse_svg(svg_path: str, _mtime: float) -> tuple[tuple[str, dict[str, Any]], ...]:
    """Parse an SVG file into immutable, reusable path records.

    The expensive work (file I/O, XML parsing and ``parse_path`` regex parsing)
    is done once per file and cached. ``_mtime`` participates in the cache key
    so the cache is invalidated automatically when the file changes on disk.
    """
    root = ElementTree.parse(svg_path).getroot()
    records = []
    for e in root.findall(f".//{_SVG_NS}path"):
        id_ = e.attrib.get("id", "")
        record: dict[str, Any] = {}
        for src_key, dst_key, default, convert in _SVG_ATTRS:
            value = e.attrib.get(src_key, default)
            record[dst_key] = convert(value) if convert is not None else value
        records.append((id_, record))
    return tuple(records)


def _as_tuple(x: str | Iterable[str]) -> tuple[str, ...]:
    """Normalize an exclude argument to a hashable, re-iterable tuple."""
    return (x,) if isinstance(x, str) else tuple(x)


@cache
def _cached_bbox(
    svg_path: str,
    _mtime: float,
    exclude: tuple[str, ...],
    bbox_exclude: tuple[str, ...],
) -> Bbox:
    """Return the (cached) bounding box of the drawn, non-excluded paths.

    Computing exact Bezier extents is by far the most expensive step, yet the
    result only depends on the file and the exclusion patterns -- so it is
    cached and reused across the many repeated ``add_fly`` / ``add_muscles``
    calls that are typical when laying out a figure.
    """
    paths = [
        record["path"]
        for id_, record in _parse_svg(svg_path, _mtime)
        if get_is_included(id_, exclude) and get_is_included(id_, bbox_exclude)
    ]
    return get_bbox(paths)


def parse_svg(svg_path: str | Path) -> tuple[tuple[str, dict[str, Any]], ...]:
    """Return cached ``(id, properties)`` records for an SVG file.

    Parameters
    ----------
    svg_path : str or Path
        Path to the SVG file.

    Returns
    -------
    tuple
        A tuple of ``(path_id, properties)`` pairs. The ``properties`` dict maps
        matplotlib keys (``path``, ``facecolor``, ``edgecolor``, ``linewidth``,
        ``alpha``) to values. Records are cached and shared, so callers must not
        mutate them in place.
    """
    svg_path = Path(svg_path)
    return _parse_svg(str(svg_path), svg_path.stat().st_mtime)


def get_path_dicts(
    svg_path: str | Path | None = None,
    exclude: str | Iterable[str] = (),
    grayscale: bool = False,
    per_path_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
    **global_kwargs: Any,
) -> dict[str, dict[str, Any]]:
    """Extract per-path matplotlib properties from an SVG file.

    Parameters
    ----------
    svg_path : str or Path, optional
        Path to the SVG file. Defaults to the bundled ``fly.svg``.
    exclude : str or Iterable[str], optional
        Regular expressions used to exclude paths by their ID.
    grayscale : bool, optional
        Convert fill colors to grayscale if ``True``. The default is ``False``.
    per_path_kwargs : Mapping, optional
        Per-path keyword arguments. Maps a property name to a mapping of
        ``{path_id: value}``.
    **global_kwargs
        Keyword arguments applied to every path.

    Returns
    -------
    dict
        A mapping of path ID to a dict of matplotlib path properties.
    """
    if svg_path is None:
        svg_path = get_data_dir() / "fly.svg"
    if per_path_kwargs is None:
        per_path_kwargs = {}

    dicts: dict[str, dict[str, Any]] = {}
    for id_, record in parse_svg(svg_path):
        if not get_is_included(id_, exclude):
            continue

        dict_e = dict(record)
        if grayscale and dict_e["facecolor"] != "none":
            dict_e["facecolor"] = _to_grayscale(dict_e["facecolor"])

        dict_e.update(global_kwargs)
        for k, v in per_path_kwargs.items():
            if id_ in v:
                dict_e[k] = v[id_]

        dicts[id_] = dict_e

    return dicts


def vectorize_dicts(dicts: Mapping[str, Mapping[str, Any]]) -> dict[str, list] | None:
    """Transpose a mapping of dicts into a dict of lists for ``PathCollection``.

    Parameters
    ----------
    dicts : Mapping
        A mapping of path ID to property dicts.

    Returns
    -------
    dict or None
        A dict mapping each property to a list of values, or ``None`` if the
        input is empty or the property dicts do not share the same keys (in
        which case a single ``PathCollection`` cannot represent them).
    """
    if not dicts:
        return None
    it = iter(dicts.values())
    k0 = set(next(it).keys())
    if any(set(d.keys()) != k0 for d in it):
        return None
    return {k: [d[k] for d in dicts.values()] for k in k0}


def add_paths(
    svg_path: str | Path,
    xy: tuple[float, float] = (0, 0),
    width: float | None = 1,
    height: float | None = None,
    scale: tuple[float, float] | float = (1, 1),
    origin: tuple[float, float] = (0.5, 0.5),
    rotation: float = 0,
    transform=None,
    exclude: str | Iterable[str] = (),
    bbox_exclude: str | Iterable[str] = (),
    grayscale: bool = False,
    force_patches: bool = False,
    ax=None,
    per_path_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
    **global_kwargs: Any,
):
    """Add the paths of an SVG file to a matplotlib axes.

    Parameters
    ----------
    svg_path : str or Path
        Path to the SVG file. Relative paths that do not exist are looked up in
        the bundled data directory.
    xy : tuple, optional
        The position of the SVG in data coordinates. The default is ``(0, 0)``.
    width : float or None, optional
        The width of the bounding box. The default is ``1``. If ``None``, the
        width is derived from ``height`` to keep the aspect ratio.
    height : float or None, optional
        The height of the bounding box. The default is ``None``, deriving the
        height from ``width``.
    scale : float or tuple of float, optional
        An additional scaling factor. The default is ``(1, 1)``.
    origin : tuple, optional
        The anchor point of the bounding box, in normalized coordinates. The
        default is ``(0.5, 0.5)`` (the center).
    rotation : float, optional
        The rotation angle in degrees. The default is ``0``.
    transform : Transform, optional
        The transform to compose with. The default is ``ax.transData``.
    exclude : str or Iterable[str], optional
        Regular expressions used to exclude paths by their ID.
    bbox_exclude : str or Iterable[str], optional
        Regular expressions for paths to ignore when computing the bounding box
        used for placement.
    grayscale : bool, optional
        Convert fill colors to grayscale if ``True``. The default is ``False``.
    force_patches : bool, optional
        Return individual :class:`~matplotlib.patches.PathPatch` objects instead
        of a single :class:`~matplotlib.collections.PathCollection`. This is
        required if you need to address paths individually after drawing.
    ax : Axes, optional
        The axes to draw on. Defaults to the current axes.
    per_path_kwargs : Mapping of Mapping, optional
        Per-path keyword arguments. The outer mapping is keyed by property name
        and the inner mapping by path ID.
    **global_kwargs
        Keyword arguments applied to every path.

    Returns
    -------
    PathCollection or dict
        A :class:`~matplotlib.collections.PathCollection` when the paths can be
        batched (the fast path), otherwise a dict mapping path ID to
        :class:`~matplotlib.patches.PathPatch`.
    """
    svg_path = Path(svg_path)
    if not svg_path.exists():
        if svg_path.is_absolute():
            raise FileNotFoundError(f"SVG file not found: {svg_path}")
        candidate = get_data_dir() / svg_path
        if not candidate.exists():
            raise FileNotFoundError(
                f"SVG file not found: {svg_path} (also looked in {get_data_dir()})"
            )
        svg_path = candidate

    exclude = _as_tuple(exclude)
    bbox_exclude = _as_tuple(bbox_exclude)

    path_dicts = get_path_dicts(
        svg_path=svg_path,
        exclude=exclude,
        grayscale=grayscale,
        per_path_kwargs=per_path_kwargs,
        **global_kwargs,
    )
    if not path_dicts:
        raise ValueError(f"No paths to draw from {svg_path} (all were excluded?).")

    bbox = _cached_bbox(str(svg_path), svg_path.stat().st_mtime, exclude, bbox_exclude)
    affine2d = get_affine_matrix(
        bbox=bbox,
        source_xy=origin,
        target_xy=xy,
        rotation=rotation,
        width=width,
        height=height,
        scale=scale,
    )

    if ax is None:
        ax = plt.gca()
    if transform is None:
        transform = ax.transData
    transform = affine2d + transform

    vectorized_dicts = vectorize_dicts(path_dicts)
    if not force_patches and vectorized_dicts is not None:
        vectorized_dicts["paths"] = vectorized_dicts.pop("path")
        collection = PathCollection(**vectorized_dicts)
        collection.set_transform(transform)
        ax.add_collection(collection)
        ax.autoscale_view()
        return collection

    patches = {k: PathPatch(**v) for k, v in path_dicts.items()}
    for patch in patches.values():
        patch.set_transform(transform)
        ax.add_patch(patch)
    ax.autoscale_view()
    return patches


def add_fly(
    xy: tuple[float, float] = (0, 0),
    rotation: float = 90,
    length: float | None = 3,
    width: float | None = None,
    scale: tuple[float, float] | float = (1, 1),
    origin: tuple[float, float] = (0.65, 0.5),
    transform=None,
    svg_path: str | Path | None = None,
    exclude: str | Iterable[str] = (),
    bbox_exclude: str | Iterable[str] = ("leg", "wing"),
    grayscale: bool = False,
    force_patches: bool = False,
    per_path_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
    ax=None,
    **global_kwargs: Any,
):
    """Add a fly to a matplotlib axes.

    A thin convenience wrapper around :func:`add_paths` with fly-specific
    defaults.

    Parameters
    ----------
    xy : tuple, optional
        The position of the fly in data coordinates. The default is ``(0, 0)``.
    rotation : float, optional
        The rotation angle in degrees. The default is ``90`` (head up).
    length : float or None, optional
        The length of the fly (head to abdomen). The default is ``3``.
    width : float or None, optional
        The width of the fly. If ``None``, derived from ``length`` to keep the
        aspect ratio.
    scale : float or tuple of float, optional
        An additional scaling factor. The default is ``(1, 1)``.
    origin : tuple, optional
        The anchor point, in normalized coordinates. The default is
        ``(0.65, 0.5)`` (roughly the body center of mass).
    transform : Transform, optional
        The transform to compose with. The default is ``ax.transData``.
    svg_path : str or Path, optional
        Path to the SVG file. Defaults to the bundled ``fly.svg``.
    exclude : str or Iterable[str], optional
        Regular expressions used to exclude paths by their ID.
    bbox_exclude : str or Iterable[str], optional
        Paths to ignore when computing the placement bounding box. Defaults to
        ``("leg", "wing")`` so flies align by their body.
    grayscale : bool, optional
        Convert fill colors to grayscale if ``True``. The default is ``False``.
    force_patches : bool, optional
        Return individual patches instead of a single ``PathCollection``.
    per_path_kwargs : Mapping of Mapping, optional
        Per-path keyword arguments (see :func:`add_paths`).
    ax : Axes, optional
        The axes to draw on. Defaults to the current axes.
    **global_kwargs
        Keyword arguments applied to every path.

    Returns
    -------
    PathCollection or dict
        See :func:`add_paths`.
    """
    if svg_path is None:
        svg_path = get_data_dir() / "fly.svg"
    return add_paths(
        svg_path=svg_path,
        xy=xy,
        width=length,
        height=width,
        scale=scale,
        origin=origin,
        rotation=rotation,
        transform=transform,
        exclude=exclude,
        bbox_exclude=bbox_exclude,
        grayscale=grayscale,
        force_patches=force_patches,
        per_path_kwargs=per_path_kwargs,
        ax=ax,
        **global_kwargs,
    )
