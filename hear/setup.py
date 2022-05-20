#!/usr/bin/env python3
from setuptools import find_packages, setup

long_description = open("README.md", "r").read()

setup(
    name="hear_msm",
    description="Masked Spectrogram Modeling using Masked Autoencoders (MSM-MAE)",
    author="Daisuke Niizumi",
    url="https://github.com/nttcslab/msm-mae",
    license="See LICENSE",
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls={
        "Bug Tracker": "https://github.com/nttcslab/msm-mae/issues",
        "Source Code": "https://github.com/nttcslab/msm-mae",
    },
    packages=find_packages(exclude=("tests",)),
    python_requires=">=3.7",
    install_requires=["timm", "nnAudio", "einops"]
)