"""Core functionality for plotting fly anatomical structures from SVG files."""

from pathlib import Path
from typing import Any, Iterable
import re
import numpy as np
from matplotlib.collections import PathCollection
from matplotlib.patches import PathPatch
from matplotlib.transforms import Bbox
import matplotlib.pyplot as plt
from xml.etree import ElementTree
from svgpath2mpl import parse_path


def get_data_dir():
    """Return the path to the data directory.

    Returns
    -------
    Path
        The path to the data directory.
    """
    from pathlib import Path

    return Path(__file__).parent / "data"


def get_bbox(objs):
    """Return the bounding box of a collection of paths.

    Parameters
    ----------
    collection : Collection
        A collection of paths.

    Returns
    -------
    Bbox
        The bounding box of the collection.
    """
    extents = np.array([p.get_extents() for p in objs])
    return Bbox(np.array([extents[:, 0].min(0), extents[:, 1].max(0)]))


def get_affine_matrix(
    bbox: Bbox,
    source_xy,
    target_xy,
    rotation: float,
    width: float,
    height: float,
    scale: tuple[float, float] | float = (1, 1),
):
    """Return an affine transformation matrix for a collection of paths.

    Parameters
    ----------
    bbox : Bbox
        The bounding box of the object.
    source_xy : tuple
        The source point in the bounding box. (0, 0) is the lower-left corner,
        (1, 0) is the lower-right corner, and (0, 1) is the upper-left corner.
    target_xy : tuple
        The target point in the data coordinates.
    rotation : float
        The rotation angle in degrees.
    width : float
        The final width of the object. If None, it is scaled according to the
        height.
    height : float
        The final height of the object. If None, it is scaled according to the
        width.
    scale : float or tuple of float, optional
        The scaling factor. If a float is provided, it is applied to both x and
        y. The default is (1.0, 1.0).

    Returns
    -------
    Affine2D
        The affine transformation matrix.
    """
    from matplotlib.transforms import Affine2D

    (x0, y0), (x1, y1) = np.array(bbox)
    affine2d = Affine2D()
    affine2d.translate(-x0 - (x1 - x0) * source_xy[0], -y0 - (y1 - y0) * source_xy[1])

    if height is not None and width is not None:
        affine2d.scale(width / (x1 - x0), height / (y1 - y0))
    elif width is not None:
        affine2d.scale(width / (x1 - x0))
    elif height is not None:
        affine2d.scale(height / (y1 - y0))

    try:
        affine2d.scale(*scale)
    except TypeError:
        affine2d.scale(scale, scale)
    affine2d.rotate_deg(rotation)
    affine2d.translate(*target_xy)

    return affine2d


def get_is_included(id_: str, exclude):
    """Check if a path ID should be included based on exclusion patterns.

    Parameters
    ----------
    id_ : str
        The path ID to check.
    exclude : str or Iterable[str]
        Regular expression patterns to exclude.

    Returns
    -------
    bool
        True if the ID should be included, False otherwise.
    """
    if isinstance(exclude, str):
        exclude = (exclude,)
    return not any(re.match(r, id_) for r in exclude)


def get_path_dicts(
    svg_path=get_data_dir() / "fly.svg",
    exclude: Iterable[str] = (),
    grayscale: bool = False,
    per_path_kwargs: dict[str, dict[str, Any]] = None,
    **global_kwargs,
):
    """Extract path information from SVG file and convert to matplotlib format.

    Parameters
    ----------
    svg_path : Path, optional
        Path to the SVG file. Default is the fly.svg file in data directory.
    exclude : Iterable[str], optional
        Regular expressions to exclude paths by ID.
    grayscale : bool, optional
        Convert colors to grayscale if True. Default is False.
    per_path_kwargs : dict, optional
        Per-path keyword arguments indexed by path ID.
    **global_kwargs
        Global keyword arguments applied to all paths.

    Returns
    -------
    dict
        Dictionary mapping path IDs to dictionaries of matplotlib path properties.
    """

    def facecolor_func(x):
        if grayscale:
            from matplotlib.colors import to_rgb

            r, g, b = to_rgb(x)
            v = 0.2989 * r + 0.5870 * g + 0.1140 * b
            return (v, v, v)
        return x

    def opacity_func(x):
        return min(max(float(x), 0), 1)

    if per_path_kwargs is None:
        per_path_kwargs = {}

    def identity(x):
        return x

    params = [
        ("d", "path", "", parse_path),
        ("fill", "facecolor", "none", facecolor_func),
        ("stroke", "edgecolor", "none", identity),
        ("stroke-width", "linewidth", 0, float),
        ("opacity", "alpha", 1, opacity_func),
    ]

    root = ElementTree.parse(svg_path).getroot()
    dicts = {}

    for e in root.findall(".//{http://www.w3.org/2000/svg}path"):
        id_ = e.attrib.get("id", "")

        if not get_is_included(id_, exclude):
            continue

        dict_e = {}

        for src_key, dst_key, default_value, func in params:
            dict_e[dst_key] = func(e.attrib.get(src_key, default_value))

        dict_e = dict(dict_e, **global_kwargs)

        for k, v in per_path_kwargs.items():
            if id_ in v:
                dict_e[k] = v[id_]

        dicts[id_] = dict_e

    return dicts


def vectorize_dicts(dicts):
    """Convert a dictionary of dictionaries to vectorized format for PathCollection.

    Parameters
    ----------
    dicts : dict
        Dictionary of dictionaries with the same keys.

    Returns
    -------
    dict or None
        Vectorized dictionary suitable for PathCollection, or None if keys differ.
    """
    it = iter(dicts.values())
    k0 = set(next(it).keys())
    for d in it:
        if set(d.keys()) != k0:
            return None
    return {k: [d[k] for d in dicts.values()] for k in k0}


def add_paths(
    svg_path,
    xy=(0, 0),
    width=1,
    height=None,
    scale=(1, 1),
    origin=(0.5, 0.5),
    rotation=0,
    transform=None,
    exclude=(),
    bbox_exclude=(),
    force_patches=False,
    ax=None,
    per_path_kwargs: dict[str, dict[str, Any]] = None,
    **global_kwargs,
):
    """
    Add SVG paths to the current axes.

    Parameters
    ----------
    svg_path : Path, optional
        The path to the SVG file. The default is the fly.svg file in the data
        directory.
    xy : tuple, optional
        The position of the SVG in the data coordinates. The default is
        (0, 0).
    width : float, optional
        The width of the bounding box. The default is 1. If None, it is scaled
        according to the height.
    height : float, optional
        The height of the bounding box. The default is None. If None, it is
        scaled according to the width.
    origin : tuple, optional
        The origin of the bounding box.
    rotation : float, optional
        The rotation angle in degrees. The default is 0.
    transform : Transform, optional
        The transform to apply to the SVG. The default is None, which
        applies the ax.transData transform.
    exclude : Iterable[str], optional
        A list of regular expressions to exclude paths by their ID.
    bbox_exclude : Iterable[str], optional
        A list of regular expressions to exclude paths when calculating the
        bounding box.
    per_path_kwargs : dict of dict, optional
        A dictionary of dictionaries containing keyword arguments to apply to
        specific paths by their ID. The outer dictionary's keys are the keyword
        argument names, and the inner dictionaries map path IDs to values.
    global_kwargs : dict, optional
        Additional keyword arguments to pass to all paths.
    ax : Axes, optional
        The axes to add the SVG to. The default is the current axes.

    Returns
    -------
    PathCollection or dict
        Either a PathCollection or dictionary of PathPatch objects.
    """
    svg_path = Path(svg_path)
    if not svg_path.exists():
        if not svg_path.is_absolute():
            svg_path = get_data_dir() / svg_path
        else:
            raise FileNotFoundError(f"SVG file not found: {svg_path}")

    path_dicts = get_path_dicts(
        svg_path=svg_path,
        exclude=exclude,
        per_path_kwargs=per_path_kwargs,
        **global_kwargs,
    )
    paths = {k: v["path"] for k, v in path_dicts.items()}

    if bbox_exclude:
        bbox_paths = (v for k, v in paths.items() if get_is_included(k, bbox_exclude))
    else:
        bbox_paths = paths.values()

    bbox = get_bbox(bbox_paths)
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

    if not force_patches and vectorized_dicts:
        vectorized_dicts["paths"] = vectorized_dicts.pop("path")
        collection = PathCollection(**vectorized_dicts)
        collection.set_transform(transform)
        ax.add_collection(collection)
        ax.autoscale()
        return collection
    else:
        patches = {k: PathPatch(**v) for k, v in path_dicts.items()}
        for patch in patches.values():
            patch.set_transform(transform)
            ax.add_patch(patch)
        ax.autoscale()
        return patches


def add_fly(
    xy=(0, 0),
    rotation=90,
    length=3,
    width=None,
    scale=(1, 1),
    origin=(0.65, 0.5),
    transform=None,
    svg_path=get_data_dir() / "fly.svg",
    exclude=(),
    bbox_exclude=("leg", "wing"),
    per_path_kwargs: dict[str, dict[str, Any]] = None,
    ax=None,
    **global_kwargs: dict[str, Any],
):
    """Add a fly to the current axes.

    This is a convenience function that calls add_paths with fly-specific
    default parameters.

    Parameters
    ----------
    xy : tuple, optional
        The position of the fly in data coordinates. Default is (0, 0).
    rotation : float, optional
        The rotation angle in degrees. Default is 90.
    length : float, optional
        The length of the fly. Default is 3.
    width : float, optional
        The width of the fly. If None, scaled according to length.
    scale : tuple of float, optional
        The scaling factor for x and y axes. Default is (1, 1).
    origin : tuple, optional
        The origin point for positioning. Default is (0.65, 0.5).
    transform : Transform, optional
        Custom matplotlib transform to apply.
    svg_path : Path, optional
        Path to the SVG file containing fly paths.
    exclude : Iterable[str], optional
        Regular expressions to exclude paths by ID.
    bbox_exclude : Iterable[str], optional
        Regular expressions to exclude from bounding box calculation.
        Default excludes legs and wings.
    per_path_kwargs : dict, optional
        Per-path keyword arguments indexed by path ID.
    ax : Axes, optional
        The axes to add the fly to. Uses current axes if None.
    global_kwargs : dict, optional
        Global keyword arguments applied to all paths.

    Returns
    -------
    PathCollection or dict
        Either a PathCollection or dictionary of PathPatch objects.
    """
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
        per_path_kwargs=per_path_kwargs,
        ax=ax,
        **global_kwargs,
    )
