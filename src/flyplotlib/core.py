"""Core functionality for plotting fly anatomical structures from SVG files.

The generic SVG-to-matplotlib rendering now lives in the standalone
:mod:`svg2mpl` package. This module is a thin, fly-specific layer on top of it:
it locates the bundled SVG assets and applies fly-friendly defaults.

The bundled artwork was authored against matplotlib's y-up coordinate system, so
:func:`add_paths` renders with ``flip_y=False`` (raw SVG coordinates) to keep the
historical appearance. Pass a custom ``svg_path`` together with placement
arguments to draw your own artwork.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from svg2mpl import add_svg

__all__ = ["get_data_dir", "add_paths", "add_fly"]


def get_data_dir() -> Path:
    """Return the path to the bundled SVG data directory.

    Returns
    -------
    Path
        The path to the ``data`` directory shipped with the package.
    """
    return Path(__file__).parent / "data"


def _resolve_svg_path(svg_path: str | Path) -> Path:
    """Resolve ``svg_path``, falling back to the bundled data directory."""
    svg_path = Path(svg_path)
    if svg_path.exists():
        return svg_path
    if svg_path.is_absolute():
        raise FileNotFoundError(f"SVG file not found: {svg_path}")
    candidate = get_data_dir() / svg_path
    if not candidate.exists():
        raise FileNotFoundError(
            f"SVG file not found: {svg_path} (also looked in {get_data_dir()})"
        )
    return candidate


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

    A thin wrapper around :func:`svg2mpl.add_svg` that resolves relative paths
    against the bundled data directory and renders in the historical (un-flipped)
    coordinate system.

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
    return add_svg(
        _resolve_svg_path(svg_path),
        xy=xy,
        width=width,
        height=height,
        scale=scale,
        origin=origin,
        rotation=rotation,
        transform=transform,
        flip_y=False,
        exclude=exclude,
        bbox_exclude=bbox_exclude,
        grayscale=grayscale,
        force_patches=force_patches,
        ax=ax,
        per_path_kwargs=per_path_kwargs,
        **global_kwargs,
    )


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
