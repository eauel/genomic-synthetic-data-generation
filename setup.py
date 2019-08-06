#!/usr/bin/env python3

from setuptools import setup, find_packages

VERSION = '0.1.0'

DESCRIPTION = 'Genotype synthetic data generator toolkit'

CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Natural Language :: English',
    'Operating System :: OS Independent',
    'Environment :: Console',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
]

setup(
    name='synthdatagen',
    version=VERSION,
    description=DESCRIPTION,
    url='https://github.com/eauel/genomic-synthetic-data-generation',
    license='MIT',
    author='Eric Auel',
    packages=find_packages(),
    include_package_data=True,
    classifiers=CLASSIFIERS,
    install_requires=[
        'dask',
        'distributed',
        'numpy',
        'msprime',
        'etaprogress'
    ],
    entry_points={
        'console_scripts': [
            'synthdatagen = DataGenerator:main'
        ]
    }
)
