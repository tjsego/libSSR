import sys

DEF_EVAL_NUM = 100
"""Default number of transform variable evaluations"""

DEF_NUM_VAR_PERS = 5
"""Default parameterization periods of empirical characteristic function transform variable evaluations"""

has_numba = False
"""Flag signifying whether numba is usable."""

# Numba seems to only behave well on Windows
if sys.platform.startswith('darwin') or sys.platform.startswith('linux'):
    pass
else:
    try:
        import numba
        has_numba = True
    except ImportError:
        pass
