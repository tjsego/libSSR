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
