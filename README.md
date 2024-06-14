# flyplotlib

flyplotlib is a Python library for adding vector graphics of fruit flies to matplotlib plots.

## Installation
To install flyplotlib, simply run the following command:

```sh
pip install flyplotlib
```

## Usage
Here's a basic example of how to use flyplotlib to add flies to a matplotlib plot:

```python
import matplotlib.pyplot as plt

from flyplotlib import add_fly
fig, ax = plt.subplots()
add_fly(xy=(0, 0), ax=ax)
add_fly(xy=(2, -0.5), rotation=30, length=2, alpha=0.3, ax=ax)
add_fly(xy=(3.5, 0), rotation=-45, grayscale=True, ax=ax)
ax.set_aspect(1)
```
