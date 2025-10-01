"""Module for handling fly leg muscle segments and visualization."""

from enum import Enum
from flyplotlib.core import add_paths, get_data_dir
from typing import Any, Iterable


class Segment(Enum):
    """Enumeration of fly leg muscle segments and anatomical parts.

    Order and names follow those from:
    https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07389-x/MediaObjects/41586_2024_7389_MOESM1_ESM.pdf
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

    @staticmethod
    def from_str(s: str) -> "Segment | None":
        """Convert a string to a Segment enum value.

        Parameters
        ----------
        s : str
            String representation of the segment name.

        Returns
        -------
        Segment | None
            The corresponding Segment enum value, or None if not found.
        """
        return _segment_from_string(s)


def _mn_tuple_from_string(s: str) -> tuple[str, ...]:
    """Convert a string representation to a normalized tuple of words.

    Parameters
    ----------
    s : str
        Input string representing a motor neuron or segment name.

    Returns
    -------
    tuple[str, ...]
        Normalized tuple of words representing the segment.
    """
    if isinstance(s, Segment):
        s = s.value
    s = s.lower()
    separators = ["/", "-", "_", ".", ","]
    for i in separators:
        s = s.replace(i, " ")
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
    words = set(s.split())
    ignored_words = {"and", "or", "mn"}
    words = words - ignored_words
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


# Mapping from normalized tuple representation to Segment enum
_mn_tuple_to_segment = {
    _mn_tuple_from_string(segment.value): segment for segment in Segment
}


def _segment_from_string(s: str) -> Segment | None:
    """Map a string to a Segment enum value.

    Parameters
    ----------
    s : str
        String representation of the segment name.

    Returns
    -------
    Segment | None
        The corresponding Segment enum value, or None if unknown.
    """
    if isinstance(s, Segment):
        return s
    tuple_ = _mn_tuple_from_string(s)
    if tuple_ not in _mn_tuple_to_segment:
        import warnings

        warnings.warn(f"Unknown motor neuron: {s}")
        return None
    return _mn_tuple_to_segment[tuple_]


# Mapping from Segment enum to SVG path IDs
svg_ids = {
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


def add_muscles(
    xy=(0, 0),
    rotation=0,
    width=None,
    height=1,
    scale=(1, -1),
    origin=(0.5, 0.5),
    transform=None,
    svg_path=get_data_dir() / "leg.svg",
    exclude: Iterable[str | Segment] = (),
    bbox_exclude: Iterable[str | Segment] = (),
    force_patches=True,
    ax=None,
    per_path_kwargs: dict[str, dict[str, Any]] = None,
    **global_kwargs,
):
    """Add fly leg muscles to the current axes.

    Parameters
    ----------
    xy : tuple, optional
        The position of the leg in data coordinates. Default is (0, 0).
    rotation : float, optional
        The rotation angle in degrees. Default is 0.
    width : float, optional
        The width of the leg. If None, scaled according to height.
    height : float, optional
        The height of the leg. Default is 1.
    scale : tuple of float, optional
        The scaling factor for x and y axes. Default is (1, -1).
    origin : tuple, optional
        The origin point for positioning. Default is (0.5, 0.5).
    transform : Transform, optional
        Custom matplotlib transform to apply.
    svg_path : Path, optional
        Path to the SVG file containing leg muscle paths.
    exclude : Iterable[str | Segment], optional
        Segment or regular expressions to exclude paths by ID.
    bbox_exclude : Iterable[str | Segment], optional
        Segment or regular expressions to exclude paths from bounding
        box calculation.
    force_patches : bool, optional
        Force creation of individual patches instead of collection.
    ax : Axes, optional
        The axes to add muscles to. Uses current axes if None.
    per_path_kwargs : dict, optional
        Per-path keyword arguments indexed by path ID.
    **global_kwargs
        Additional keyword arguments passed to all paths.

    Returns
    -------
    Collection or dict
        Either a PathCollection or dictionary of PathPatch objects.
    """
    if per_path_kwargs is None:
        per_path_kwargs = {}
    else:
        per_path_kwargs = {
            k: {
                (k_ if k_ in svg_ids.values() else svg_ids[Segment.from_str(k_)]): v_
                for k_, v_ in v.items()
            }
            for k, v in per_path_kwargs.items()
        }
    bbox_exclude = [
        svg_ids[Segment.from_str(s)] if Segment.from_str(s) is not None else s
        for s in bbox_exclude
    ]
    exclude = [
        svg_ids[Segment.from_str(s)] if Segment.from_str(s) is not None else s
        for s in exclude
    ]

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


def mushow(
    x: dict[str | Segment, float] | Iterable[float],
    cmap="viridis",
    norm=None,
    vmin=None,
    vmax=None,
    **kwargs,
):
    """Display muscle activation values as colored leg muscles.

    Parameters
    ----------
    x : dict or Iterable
        Muscle activation values. Can be a dictionary mapping Segment enum
        values or strings to float values, or an iterable of float values
        in the order of the Segment enum.
    cmap : str or Colormap, optional
        Colormap for mapping values to colors. Default is "viridis".
    norm : Normalize, optional
        Normalization instance for mapping values to colormap range.
    vmin : float, optional
        Minimum value for color scaling. Auto-computed if None.
    vmax : float, optional
        Maximum value for color scaling. Auto-computed if None.
    **kwargs
        Additional keyword arguments passed to add_muscles.

    Returns
    -------
    tuple
        A tuple containing the muscle visualization object and ScalarMappable
        for creating colorbars.
    """
    from itertools import repeat
    from math import isfinite
    from matplotlib.cm import ScalarMappable

    d = dict(zip(Segment, repeat(float("nan"))))
    it = x.items() if isinstance(x, dict) else zip(Segment, x)
    for k, v in it:
        d[Segment(k)] = v

    if norm is None:
        from matplotlib.colors import Normalize

        if vmin is None:
            vmin = min(d.values())
        if vmax is None:
            vmax = max(d.values())
        norm = Normalize(vmin=vmin, vmax=vmax)

    mappable = ScalarMappable(norm=norm, cmap=cmap)
    facecolor = {
        svg_ids[k]: mappable.to_rgba(v) if isfinite(v) else "none"
        for k, v in d.items()
        if k in svg_ids
    }
    per_path_kwargs = {
        "facecolor": facecolor,
    }
    return add_muscles(
        per_path_kwargs=per_path_kwargs,
        **kwargs,
    ), mappable
