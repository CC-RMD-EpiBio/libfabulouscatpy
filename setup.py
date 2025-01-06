#!/usr/bin/env python3

import os

from setuptools import find_packages, setup

dirname = os.path.dirname(__file__)


def extract_version():
    with open(os.path.join(dirname, 'libfabulouscatpy/__init__.py')) as fd:
        ns = {}
        for line in fd:
            if line.startswith('__version__'):
                exec(line.strip(), ns)
                return ns['__version__']


def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths





setup(
    name="libfabulouscatpy",
    version=extract_version(),
    author="Josh Chang and Aaron Heuser",
    author_email="josh.chang@nih.gov",
    packages=find_packages(),
    package_dir={"libfabulouscatpy": "libfabulouscatpy"},
    package_data={
    },
    include_package_data=True,
    description="libfabulouscatpy",
    long_description=open(os.path.join(dirname, "README.md")).read(),
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.11'
    ],
    entry_points={
        'console_scripts': [],
    },

    zip_safe=False
)
