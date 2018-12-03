#!/usr/bin/env python

from setuptools import setup

setup(name='datasetutil',
    version='0.1.2',
    description='Simple image dataset format, with several useful download scripts',
    author='Larry Neal',
    author_email='nealla@lwneal.com',
    packages=[
        'datasetutil',
    ],
    install_requires=[
        'requests',
        'numpy',
        'Pillow',
    ],
)
