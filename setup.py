
from setuptools import setup, Extension, find_packages

__version__="0.1.0"

setup(
    name="DelayArray",
    version=__version__,
    description="Delayed evaluation for GPUs in Python",
    author="Magnus Morton",
    author_email="magnus.morton@ed.ac.uk",
    packages=find_packages(),
    include_package_data = True,
    test_suite="tests",
    install_requires=[
#        "transmuter>=0.5",
        "numpy>=1.15",
        "pyopencl"
    ]
)
