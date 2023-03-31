#!/usr/bin/env python3
import os
from setuptools import find_packages
from setuptools import setup


requirements = {
    "install": [
        "numpy",
    ],
    "setup": ["pytest-runner"],
}

install_requires = requirements["install"]
setup_requires = requirements["setup"]
extras_require = {
    k: v for k, v in requirements.items() if k not in ["install", "setup"]
}

dirname = os.path.dirname(__file__)
setup(
    name="espnet2",
    version="0.1.7",
    description="ESPnet Model Zoo",
    long_description='a python command tool for camel case',
    license="Apache Software License",
    entry_points={
        "console_scripts": [
            "espnet2.bin.asr_inference = espnet2.bin.asr_inference:main",
        ],
    },
    install_requires=install_requires,
    setup_requires=setup_requires,
    classifiers=[
    ],
)
