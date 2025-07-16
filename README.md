# libSSR
## Library for Stochastic Simulation Reproducibility

libSSR is an open-source software library designed to enhance the reproducibility of stochastic simulations.
libSSR provides tools that help researchers and developers ensure that their stochastic models yield
consistent and verifiable results across different runs and environments.
This capability delivers critical reproducibility for validation and peer review in computational domains
where randomness plays a central role. libSSR provides features to accomplish the following tasks:

* Quantifying how well others can reproduce a given sample of stochastic simulation results.
* Packaging encoded results and metadata to facilitate information exchange allowing reproducibility studies at scale.
* Testing whether a given sample of stochastic simulation results matches those encoded in packaged data.

## Installation

The libSSR python module is available for installation via pip:

```
pip install libssr
```

The libSSR python module is also available for installation via conda:

```
conda install -c conda-forge libssr-py
```
