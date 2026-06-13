"""Fly leg muscle segments and their visualization."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping
from enum import Enum
from math import isfinite
from pathlib import Path
from typing import Any

from flyplotlib.core import add_paths, get_data_dir


class Segment(Enum):
    """Fly leg muscle segments and anatomical parts.

    Order and names follow those from the supplementary material of
    Lesser et al. (2024), *Nature*: https://doi.org/10.1038/s41586-024-07389-x
    """

    TERGOPLEURAL_OR_PLEURAL_PROMOTOR = "tergopleural_or_pleural_promotor"
    PLEURAL_REMOTOR_AND_ABDUCTOR = "pleural_remotor_and_abductor"
    STERNAL_ANTERIOR_ROTATOR = "sternal_anterior_rotator"
    STERNAL_POSTERIOR_ROTATOR = "sternal_posterior_rotator"
    STERNAL_ADDUCTOR = "sternal_adductor"
    TERGOTROCHANTER_EXTENSOR = "tergotrochanter_extensor"
    STERNOTROCHANTER_EXTENSOR = "sternotrochanter_extensor"
    TROCHANTER_EXTENSOR = "trochanter_extensor"
    TROCHANTER_FLEXOR = "trochanter_flexor"
    ACCESSORY_TROCHANTER_FLEXOR = "accessory_trochanter_flexor"
    FEMUR_REDUCTOR = "femur_reductor"
    TIBIA_EXTENSOR = "tibia_extensor"
    TIBIA_FLEXOR = "tibia_flexor"
    ACCESSORY_TIBIA_FLEXOR = "accessory_tibia_flexor"
    TARSUS_DEPRESSOR = "tarsus_depressor"
    TARSUS_LEVATOR = "tarsus_levator"
    LONG_TENDON_MUSCLE_2 = "long_tendon_muscle_2"  # in femur
    LONG_TENDON_MUSCLE_1 = "long_tendon_muscle_1"  # in tibia
    # non-muscle segments
    TIBIA_TARSUS = "tibia_tarsus"
    TARSUS_1 = "tarsus_1"
    TARSUS_2 = "tarsus_2"
    TARSUS_3 = "tarsus_3"
    TARSUS_4 = "tarsus_4"
    TARSUS_5 = "tarsus_5"

    @classmethod
    def from_str(cls, s: str | Segment) -> Segment | None:
        """Resolve a (possibly messy) string to a :class:`Segment`.

        Accepts enum members, exact enum values and human-written motor neuron
        names with arbitrary separators, abbreviations and word order.

        Parameters
        ----------
        s : str or Segment
            The value to resolve.

        Returns
        -------
        Segment or None
            The matching segment, or ``None`` (with a warning) if unknown.
        """
        if isinstance(s, cls):
            return s
        tuple_ = _mn_tuple_from_string(s)
        segment = _mn_tuple_to_segment.get(tuple_)
        if segment is None:
            warnings.warn(f"Unknown motor neuron / segment: {s!r}", stacklevel=2)
        return segment


def _mn_tuple_from_string(s: str | Segment) -> tuple[str, ...]:
    """Normalize a motor neuron / segment name into a canonical word tuple."""
    if isinstance(s, Segment):
        s = s.value
    s = s.lower()
    for sep in "/-_.,":
        s = s.replace(sep, " ")
    word_map = {
        "long tendon muscle": "ltm",
        "accessory": "acc",
        "tibia": "ti",
        "femur": "fe",
        "trochanter": "tr",
        "tarsus": "ta",
        "ltm1": "ltm 1",
        "ltm2": "ltm 2",
    }
    for k, v in word_map.items():
        s = s.replace(k, v)
    ignored_words = {"and", "or", "mn"}
    words = set(s.split()) - ignored_words
    tuple_ = tuple(sorted(words))
    tuple_map = {
        ("tergotr",): ("extensor", "tergotr"),
        ("sternotr",): ("extensor", "sternotr"),
        ("1", "ltm", "ti"): ("1", "ltm"),
        ("2", "fe", "ltm"): ("2", "ltm"),
        ("reductor", "tr"): ("acc", "flexor", "tr"),
        ("flexor", "ta"): ("levator", "ta"),
        ("extensor", "ta"): ("depressor", "ta"),
    }
    return tuple_map.get(tuple_, tuple_)


# Mapping from normalized tuple representation to Segment enum.
_mn_tuple_to_segment = {
    _mn_tuple_from_string(segment.value): segment for segment in Segment
}

# Mapping from Segment enum to SVG path IDs in ``leg.svg``.
svg_ids: dict[Segment, str] = {
    Segment.TERGOPLEURAL_OR_PLEURAL_PROMOTOR: "Tergopleural-Pleural-promotor",
    Segment.PLEURAL_REMOTOR_AND_ABDUCTOR: "Pleural-remotor-abductor",
    Segment.STERNAL_ANTERIOR_ROTATOR: "Sternal-anterior-rotator",
    Segment.STERNAL_POSTERIOR_ROTATOR: "Sternal-posterior-rotator",
    Segment.STERNAL_ADDUCTOR: "Sternal-adductor",
    Segment.TERGOTROCHANTER_EXTENSOR: "Tergotr.",
    Segment.STERNOTROCHANTER_EXTENSOR: "Sternotrochanter",
    Segment.TROCHANTER_EXTENSOR: "Tr-extensor",
    Segment.TROCHANTER_FLEXOR: "Tr-flexor",
    Segment.ACCESSORY_TROCHANTER_FLEXOR: "Tr-reductor",
    Segment.FEMUR_REDUCTOR: "Fe-reductor",
    Segment.TIBIA_EXTENSOR: "Ti-extensor",
    Segment.TIBIA_FLEXOR: "Ti-flexor",
    Segment.ACCESSORY_TIBIA_FLEXOR: "Ti-acc.-flexor",
    Segment.TARSUS_DEPRESSOR: "Ta-extensor",
    Segment.TARSUS_LEVATOR: "Ta-flexor",
    Segment.LONG_TENDON_MUSCLE_2: "LTM-2",
    Segment.LONG_TENDON_MUSCLE_1: "LTM-1",
    Segment.TIBIA_TARSUS: "Ti-Ta",
    Segment.TARSUS_1: "Ta-1",
    Segment.TARSUS_2: "Ta-2",
    Segment.TARSUS_3: "Ta-3",
    Segment.TARSUS_4: "Ta-4",
    Segment.TARSUS_5: "Ta-5",
}

_svg_id_set = frozenset(svg_ids.values())


def _to_svg_id(key: str | Segment) -> str:
    """Resolve a segment, segment name or raw SVG ID to an SVG path ID."""
    if key in _svg_id_set:
        return key  # already an SVG path ID
    segment = (
        key
        if isinstance(key, Segment)
        else _mn_tuple_to_segment.get(_mn_tuple_from_string(key))
    )
    if segment is None:
        raise KeyError(f"Cannot resolve {key!r} to a leg SVG path ID.")
    return svg_ids[segment]


def add_muscles(
    xy: tuple[float, float] = (0, 0),
    rotation: float = 0,
    width: float | None = None,
    height: float | None = 1,
    scale: tuple[float, float] | float = (1, -1),
    origin: tuple[float, float] = (0.5, 0.5),
    transform=None,
    svg_path: str | Path | None = None,
    exclude: Iterable[str | Segment] = (),
    bbox_exclude: Iterable[str | Segment] = (),
    force_patches: bool = True,
    ax=None,
    per_path_kwargs: Mapping[str, Mapping[str | Segment, Any]] | None = None,
    **global_kwargs: Any,
):
    """Add the fly leg muscles to a matplotlib axes.

    Parameters
    ----------
    xy : tuple, optional
        The position of the leg in data coordinates. The default is ``(0, 0)``.
    rotation : float, optional
        The rotation angle in degrees. The default is ``0``.
    width : float or None, optional
        The width of the leg. If ``None``, derived from ``height``.
    height : float or None, optional
        The height of the leg. The default is ``1``.
    scale : float or tuple of float, optional
        An additional scaling factor. The default is ``(1, -1)``.
    origin : tuple, optional
        The anchor point, in normalized coordinates. The default is
        ``(0.5, 0.5)``.
    transform : Transform, optional
        The transform to compose with. The default is ``ax.transData``.
    svg_path : str or Path, optional
        Path to the SVG file. Defaults to the bundled ``leg.svg``.
    exclude : Iterable[str or Segment], optional
        Segments (or regular expressions) to exclude.
    bbox_exclude : Iterable[str or Segment], optional
        Segments (or regular expressions) to ignore for placement.
    force_patches : bool, optional
        Return individual patches keyed by SVG ID instead of a single
        ``PathCollection``. The default is ``True`` so individual muscles can be
        styled after drawing.
    ax : Axes, optional
        The axes to draw on. Defaults to the current axes.
    per_path_kwargs : Mapping, optional
        Per-path keyword arguments. The inner mapping may be keyed by
        :class:`Segment`, a segment name or a raw SVG ID.
    **global_kwargs
        Keyword arguments applied to every path.

    Returns
    -------
    PathCollection or dict
        See :func:`flyplotlib.add_paths`.
    """
    if svg_path is None:
        svg_path = get_data_dir() / "leg.svg"

    if per_path_kwargs is not None:
        per_path_kwargs = {
            prop: {_to_svg_id(k): v for k, v in mapping.items()}
            for prop, mapping in per_path_kwargs.items()
        }

    exclude = [_segment_pattern(s) for s in exclude]
    bbox_exclude = [_segment_pattern(s) for s in bbox_exclude]

    return add_paths(
        svg_path=svg_path,
        xy=xy,
        width=width,
        height=height,
        scale=scale,
        origin=origin,
        rotation=rotation,
        transform=transform,
        exclude=exclude,
        bbox_exclude=bbox_exclude,
        force_patches=force_patches,
        ax=ax,
        per_path_kwargs=per_path_kwargs,
        **global_kwargs,
    )


def _segment_pattern(s: str | Segment) -> str:
    """Resolve a segment to its SVG ID, leaving unknown strings as regex."""
    if isinstance(s, str) and s in _svg_id_set:
        return s
    segment = Segment.from_str(s)
    return svg_ids[segment] if segment is not None else s


def mushow(
    x: Mapping[str | Segment, float] | Iterable[float],
    cmap="viridis",
    norm=None,
    vmin: float | None = None,
    vmax: float | None = None,
    **kwargs: Any,
):
    """Color the leg muscles by a per-muscle scalar value ("muscle imshow").

    Parameters
    ----------
    x : Mapping or Iterable
        Muscle values. Either a mapping of ``{segment: value}`` (segments may be
        :class:`Segment` members, names or SVG IDs) or an iterable of values in
        :class:`Segment` order. Non-finite values (e.g. ``nan``) leave the
        corresponding muscle uncolored.
    cmap : str or Colormap, optional
        The colormap. The default is ``"viridis"``.
    norm : Normalize, optional
        A normalization instance. If given, ``vmin``/``vmax`` are ignored.
    vmin, vmax : float, optional
        The color scale limits. Auto-computed from finite values if ``None``.
    **kwargs
        Additional keyword arguments forwarded to :func:`add_muscles`.

    Returns
    -------
    tuple
        ``(paths, mappable)`` where ``paths`` is the return value of
        :func:`add_muscles` and ``mappable`` is a
        :class:`~matplotlib.cm.ScalarMappable` suitable for ``colorbar``.
    """
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    values = {segment: float("nan") for segment in Segment}
    items = x.items() if isinstance(x, Mapping) else zip(Segment, x, strict=False)
    for k, v in items:
        segment = k if isinstance(k, Segment) else Segment.from_str(k)
        if segment is not None:
            values[segment] = v

    if norm is None:
        finite = [v for v in values.values() if isfinite(v)]
        if vmin is None:
            vmin = min(finite) if finite else 0.0
        if vmax is None:
            vmax = max(finite) if finite else 1.0
        norm = Normalize(vmin=vmin, vmax=vmax)

    mappable = ScalarMappable(norm=norm, cmap=cmap)
    facecolor = {
        svg_ids[k]: mappable.to_rgba(v) if isfinite(v) else "none"
        for k, v in values.items()
        if k in svg_ids
    }
    return add_muscles(per_path_kwargs={"facecolor": facecolor}, **kwargs), mappable
