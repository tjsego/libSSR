import os
from pathlib import Path
from setuptools import setup
import shutil

project_dir = Path(__file__).resolve().parents[2]
package_dir = Path(__file__).resolve().parents[0].joinpath('libssr')
license_fp = package_dir.joinpath('LICENSE')
version_fp = package_dir.joinpath('VERSION.txt')

# move docs into package if necessary
_tmp_docs = not version_fp.exists()

if _tmp_docs:
    shutil.copy(Path(__file__).resolve().parents[2].joinpath('LICENSE'), license_fp)
    shutil.copy(Path(__file__).resolve().parents[2].joinpath('VERSION.txt'), version_fp)

__version__ = version_fp.read_text()

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
    package_data={'libssr': ['LICENSE', 'VERSION.txt']},
    extras_require={
        'mkstd': ['mkstd >= 0.0.5']
    }
)

if _tmp_docs:
    os.remove(license_fp)
    os.remove(version_fp)
