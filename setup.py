
from setuptools import setup, find_packages

__version__ = "0.1.0"

setup(
    name="DelayRepay",
    version=__version__,
    description="Delayed evaluation for GPUs in Python",
    author="Magnus Morton",
    author_email="magnus.morton@ed.ac.uk",
    packages=["delayrepay"],
    include_package_data=True,
    test_suite="tests",
    install_requires=[
        "numpy>=1.16",
# set this to your CUDA version "cupy-cuda92>=7.5.0"
    ]
)
