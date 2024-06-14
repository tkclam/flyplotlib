from setuptools import setup, find_packages

setup(
    name="flyplotlib",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "svgpath2mpl",
    ],
    include_package_data=True,
)
