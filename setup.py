#!/usr/bin/env python
from setuptools import find_packages, setup

with open('README.md') as f:
    readme = f.read()
with open('requirements/build.txt') as f:
    requirements = f.read()
setup(
    # Metadata
    name='easyrobust',
    version='0.2.3',
    python_requires='>=3.6',
    author='Alibaba Security',
    author_email='mxf164419@alibaba-inc.com',
    url='',
    description='Alibaba EasyRobust Toolkit',
    long_description=readme,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    include_package_data=True,
    #Package info
    install_requires=requirements)
