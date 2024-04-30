import numpy as np

from . import consts

if consts.has_numba:
    import numba


def round_to_sigfigs(_val: float, _sigfigs: int) -> float:
    """
    Round a value to a number of significant figures

    :param _val: value to round
    :param _sigfigs: number of significant figures
    """
    if _val == 0:
        return _val
    else:
        return np.round(_val, -int(np.multiply(np.sign(_val), np.floor(np.log10(np.abs(_val))))) + _sigfigs - 1)


if consts.has_numba:
    round_to_sigfigs = numba.njit(round_to_sigfigs)


def round_arr_to_sigfigs(_vals: np.ndarray, _sigfigs: int) -> np.ndarray:
    """
    Round each element of an array to a number of significant figures

    :param _vals: array to round
    :param _sigfigs: number of significant figures
    """
    result = np.zeros_like(_vals)
    result_r = result.ravel()
    _vals_r = _vals.ravel()
    for i in range(_vals_r.shape[0]):
        result_r[i] = round_to_sigfigs(_vals_r[i], _sigfigs)
    return result


if consts.has_numba:
    round_arr_to_sigfigs = numba.njit(round_arr_to_sigfigs)
