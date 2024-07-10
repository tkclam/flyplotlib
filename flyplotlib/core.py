from typing import Iterable
import re
import numpy as np
from matplotlib.collections import Collection, PathCollection
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


def get_collection_bbox(collection: Collection):
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
    extents = np.array([p.get_extents() for p in collection.get_paths()])
    return Bbox(np.array([extents[:, 0].min(0), extents[:, 1].max(0)]))


def get_affine_matrix(
    bbox: Bbox,
    source_xy,
    target_xy,
    rotation: float,
    width: float,
    height: float,
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

    affine2d.rotate_deg(rotation)
    affine2d.translate(*target_xy)

    return affine2d


class FlyPathCollection(PathCollection):
    def __init__(
        self,
        alpha=1,
        grayscale=False,
        svg_path=get_data_dir() / "fly.svg",
        exclude: Iterable[str] = (),
        **kwargs,
    ):
        """Create a fly from an SVG file.

        Parameters
        ----------
        alpha : float, optional
            Multiply the alpha channel of the colors by this value. The
            default is 1.
        grayscale : bool, optional
            Convert the colors to grayscale. The default is False.
        svg_path : Path, optional
            The path to the SVG file. The default is the fly.svg file in the
            data directory.
        exclude : Iterable[str], optional
            A list of regular expressions to exclude paths by their ID.
        **kwargs
            Additional keyword arguments passed to the PathCollection
            constructor.
        """

        root = ElementTree.parse(svg_path).getroot()
        elems = root.findall(".//{http://www.w3.org/2000/svg}path")
        ids = [e.attrib.get("id", "") for e in elems]

        if isinstance(exclude, str):
            exclude = (exclude,)

        is_included = [not any(re.match(r, i) for r in exclude) for i in ids]
        elems = [e for e, i in zip(elems, is_included) if i]
        paths = [parse_path(e.attrib["d"]) for e in elems]
        fcs = [e.attrib.get("fill", "none") for e in elems]
        ecs = [e.attrib.get("stroke", "none") for e in elems]
        lws = [float(e.attrib.get("stroke_width", 0)) for e in elems]
        alpha = [alpha * float(e.attrib.get("opacity", 1)) for e in elems]
        alpha = np.clip(alpha, 0, 1)

        if grayscale:
            from matplotlib.colors import to_rgb

            weights = np.array([0.2989, 0.5870, 0.1140])
            fcs = [(weights @ to_rgb(c),) * 3 for c in fcs]

        super().__init__(
            paths, edgecolors=ecs, facecolors=fcs, alpha=alpha, linewidths=lws, **kwargs
        )


def add_fly(
    xy=(0, 0),
    rotation=90,
    length=3,
    width=None,
    origin=(0.65, 0.5),
    transform=None,
    svg_path=get_data_dir() / "fly.svg",
    ax=None,
    **kwargs,
):
    """
    Add a fly to the current axes.

    Parameters
    ----------
    xy : tuple, optional
        The position of the fly in the data coordinates. The default is
        (0, 0).
    rotation : float, optional
        The rotation angle in degrees. The default is 90.
    length : float, optional
        The length of the fly. The default is 3.
    width : float, optional
        The width of the fly. The default is None (scaled according to the
        length).
    origin : tuple, optional
        The origin of the fly in the bounding box. (0, 0) is the
        posterior-right corner, (1, 0) is the anterior-right corner, and
        (0, 1) is the posterior-left corner. The default is (0.65, 0.5)
        (thorax).
    transform : Transform, optional
        The transform to apply to the fly. The default is None, which
        applies the ax.transData transform.
    svg_path : Path, optional
        The path to the SVG file. The default is the fly.svg file in the data
        directory.
    ax : Axes, optional
        The axes to add the fly to. The default is the current axes.
    **kwargs
        Additional keyword arguments passed to the PathCollection constructor.

    Returns
    -------
    FlyPathCollection
        The fly path collection.
    """
    fly = FlyPathCollection(svg_path=svg_path, **kwargs)

    bbox = get_collection_bbox(fly)
    affine2d = get_affine_matrix(
        bbox=bbox,
        source_xy=origin,
        target_xy=xy,
        rotation=rotation,
        width=length,
        height=width,
    )

    if ax is None:
        ax = plt.gca()

    if transform is None:
        transform = ax.transData

    fly.set_transform(affine2d + transform)
    ax.add_collection(fly)
    ax.autoscale()

    return fly
