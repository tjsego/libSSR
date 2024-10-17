from pathlib import Path
from setuptools import setup

__version__ = (Path(__file__).resolve().parents[2] / 'VERSION.txt').read_text()

setup(
    name='libssr',
    version=__version__,
    description='A library for stochastic simulation reproducibility',
    author="T.J. Sego",
    author_email="timothy.sego@medicine.ufl.edu",
    python_requires='>=3.8',
    install_requires=['numpy', 'mkstd >= 0.0.4'],
    packages=['libssr'],
    package_dir={'libssr': 'libssr'},
    package_data={'libssr': ['../../../LICENSE', '../../../VERSION.txt']}
)
