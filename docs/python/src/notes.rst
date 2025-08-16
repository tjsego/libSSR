.. _notes:

Notes
======

Code Acceleration with Numba JIT Compilation
---------------------------------------------

libSSR will accelerate computations with Just-in-time (JIT) compilation
if the `numba <https://numba.pydata.org/>`_ package is installed.
JIT compilation via Numba can be disabled by
setting the environment variable ``LIBSSR_NO_NUMBA`` before
first import of libSSR.

Differences in Data Precision
------------------------------

In most cases, differences in floating point precision do not affect
outcomes of comparing samples. However, in some scenarios such differences
can produce false negatives. An EFECT Report records the numerical precision
of the encoded data for this reason. While nothing prevents using a
different numerical precision in a reproducibility study,
practitioners are advised to consider differences in numerical precision
when evaluating causes of detected differences among samples.

Below is a running list of known scenarios where differences in
numerical precision are known to cause spurious results:

* When a distribution approaches a constant value
