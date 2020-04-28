#!/usr/bin/env python
from setuptools import setup, find_packages
from pyhrp import __version__ as version

# read the contents of your README file
with open('README.md') as f:
    long_description = f.read()

setup(
    name='pyhrp',
    version=version,
    packages=find_packages(include=["pyhrp*"]),
    author='Thomas Schmelzer',
    author_email='thomas.schmelzer@gmail.com',
    description='Python for Hierarchical Risk Parity',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/tschm/hrp',
    project_urls={
        "Source Code": "https://github.com/tschm/hrp"
    },
    install_requires=['pandas>=0.25.3', 'scipy', 'numpy', 'matplotlib'],
)