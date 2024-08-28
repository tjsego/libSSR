import os
from setuptools import setup

__version__ = open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                'VERSION.txt')).readline().strip()

setup(
    name='libssr',
    version=__version__,
    description='A library for stochastic simulation reproducibility',
    author="T.J. Sego",
    author_email="timothy.sego@medicine.ufl.edu",
    python_requires='>=3.8',
    install_requires=['numpy'],
    packages=['libssr'],
    package_dir={'libssr': 'libssr'},
    package_data={'libssr': ['../../../LICENSE', '../../../VERSION.txt']}
)
