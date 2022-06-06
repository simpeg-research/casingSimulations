#!/usr/bin/env python
from __future__ import print_function
"""CasingSimulations: Numerical simulations of electromagnetic surveys over
in settings where steel cased wells are present.
"""

from setuptools import setup, Extension
from setuptools import find_packages

CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Scientific/Engineering :: Physics',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: POSIX',
    'Operating System :: Unix',
    'Operating System :: MacOS',
    'Natural Language :: English',
]

with open("README.rst") as f:
    LONG_DESCRIPTION = ''.join(f.readlines())

setup(
    name="CasingSimulations",
    version="0.2.1",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.7',
        'scipy>=0.13',
        'cython',
        'pymatsolver>=0.1.1',
        'ipywidgets',
        'jupyter',
        'matplotlib',
        'properties',
        'vectormath',
        'SimPEG',
    ],

    author="Lindsey Heagy",
    author_email="lindseyheagy@gmail.com",
    description="Casing Simulations: Electromagnetics + Steel Cased Wells",
    long_description=LONG_DESCRIPTION,
    license="MIT",
    keywords="geophysics electromagnetics",
    url="http://github.com/simpeg-research/casingResearch",
    download_url="http://github.com/simpeg-research/casingResearch",
    classifiers=CLASSIFIERS,
    platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    use_2to3=False
)
