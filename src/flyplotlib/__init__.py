"""FlyPlotLib: A library for plotting fly anatomical structures.

This library provides tools for visualizing fruit fly anatomy including
whole fly bodies and detailed muscle structures from SVG files.
"""

from flyplotlib.core import add_fly, add_paths
from flyplotlib.leg import add_muscles, mushow, Segment

__all__ = ["add_fly", "add_paths", "add_muscles", "mushow", "Segment"]
