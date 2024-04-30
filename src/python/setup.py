import os
from setuptools import setup

__version__ = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'VERSION.txt')).readline().strip()

setup(
    name='libssr',
    version=__version__,
    description='Library for Stochastic Simulation Reproducibility',
    author="T.J. Sego",
    author_email="timothy.sego@medicine.ufl.edu",
    install_requires=['numpy']
)
