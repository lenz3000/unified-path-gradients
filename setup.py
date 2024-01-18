#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name="unified_path",
    use_scm_version=True,
    packages=find_packages(),
    install_requires=[
        "torch>=0.4.1",
        "numpy",
        "tqdm",
        "matplotlib",
        "tensorboard",
        "click",
        "jupyter",
        "einops",
    ],
    setup_requires=[
        "setuptools_scm",
    ],
)
