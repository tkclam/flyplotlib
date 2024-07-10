from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="flyplotlib",
    version="0.3.0",
    author="Thomas Ka Chung Lam",
    author_email="thomas.lam@epfl.ch",
    description="Add flies to matplotlib plots.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tkclam/flyplotlib",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "svgpath2mpl",
    ],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
