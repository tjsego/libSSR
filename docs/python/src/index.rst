===============
 LibSSR Python
===============

---------------------------------------------------------------
 A Library for Stochastic Simulation Reproducibility in Python
---------------------------------------------------------------

libSSR is an open-source software library designed to enhance the reproducibility of stochastic simulations.
libSSR provides tools that help researchers and developers ensure that their stochastic models yield
consistent and verifiable results across different runs and environments.
This capability delivers critical reproducibility for validation and peer review in computational domains
where randomness plays a central role. libSSR provides features to accomplish the following tasks:

* Quantifying how well others can reproduce a given sample of stochastic simulation results.
* Packaging encoded results and metadata to facilitate information exchange allowing reproducibility studies at scale.
* Statistical testing of whether a given sample of stochastic simulation results matches those encoded in packaged data.

Installation
=============

The libSSR Python package is available for installation via pip:

.. code-block:: bash

    pip install libssr

The libSSR Python package is also available for installation via conda:

.. code-block:: bash

    conda install -c conda-forge libssr-py

Citation
=========

libSSR is based on the Empirical Characteristic Function Equality Convergence Test (EFECT).
To use libSSR in research, please cite the publication that describes EFECT:

    Sego, T. J., et al. "EFECT--A Method and Metric to Assess the Reproducibility of Stochastic Simulation Studies." arXiv preprint arXiv:2406.16820 (2024).

.. toctree::
    :maxdepth: 1

    comparison_wfs
    packaging
    simulators
    notes
    api/index
    history
