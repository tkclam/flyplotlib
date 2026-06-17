"""flyplotlib: add vector graphics of fruit flies and leg muscles to matplotlib.

The public API is intentionally small:

- :func:`add_fly` -- draw a fly at a position/rotation/size.
- :func:`add_paths` -- draw the paths of any bundled or custom SVG.
- :func:`add_muscles` -- draw the leg muscles.
- :func:`mushow` -- color the leg muscles by a per-muscle scalar value.
- :class:`Segment` -- the leg muscle segments enumeration.
- :func:`add_neuromechfly` -- draw a posed NeuroMechFly cartoon from joint angles.
"""

from importlib.metadata import PackageNotFoundError, version

from flyplotlib.core import add_fly, add_paths
from flyplotlib.leg import Segment, add_muscles, mushow
from flyplotlib.neuromechfly import (
    NEUROMECHFLY_SEGMENTS,
    add_neuromechfly,
    neuromechfly_segments,
    neutral_pose,
)

try:
    __version__ = version("flyplotlib")
except PackageNotFoundError:  # not installed (e.g. running from a source tree)
    __version__ = "0.0.0+unknown"

__all__ = [
    "add_fly",
    "add_paths",
    "add_muscles",
    "mushow",
    "Segment",
    "add_neuromechfly",
    "neuromechfly_segments",
    "neutral_pose",
    "NEUROMECHFLY_SEGMENTS",
    "__version__",
]
