from functools import partial
import numpy as np
from typing import Dict

from . import consts

if not consts.has_mlx:
    raise ImportError('MLX is not available in this installation.')

import mlx.core as mx


def get_eval_info_times(_eval_num, _eval_fin, **kwargs):
    return mx.linspace(0.0, _eval_fin, _eval_num, **kwargs)


@mx.compile
def ecfi(var_vals: mx.array, func_evals: mx.array) -> mx.array:
    """
    Evaluate the empirical characteristic function of a sample of a variable
    at a simulation time at given transform variable values.

    :param var_vals: trajectory values of a variable at a simulation time
    :param func_evals: transform variable values at which to compute the empirical characteristic function
    :return: empirical characteristic function evaluations; dim 0 is evaluations; dim 1 is real and imaginary components
    """
    return mx.mean(mx.exp(1j * var_vals[None, :] * func_evals[:, None]), axis=1)


def ecf(var_vals: mx.array, func_evals: mx.array) -> mx.array:
    """
    Evaluate the empirical characteristic function of a sample of a variable
    at a simulation time at given transform variable values.

    :param var_vals: trajectory values of a variable at a simulation time
    :param func_evals: transform variable values at which to compute the empirical characteristic function
    :return: empirical characteristic function evaluations; dim 0 is evaluations; dim 1 is real and imaginary components
    """
    x = ecfi(var_vals, func_evals)
    result = mx.zeros((func_evals.shape[0], 2))
    result[:, 0] = mx.real(x)
    result[:, 1] = mx.imag(x)
    return result


@mx.compile
def ecfi_compare(_ecf_1: mx.array, _ecf_2: mx.array):
    """
    Compare empirical characteristic function evaluations.

    Arrays are assumed to correspond to empirical characteristic functions that were
    evaluated at the same transform variable values.

    Array first dimension elements are per transform variable value, in increasing order.

    Array second dimension elements 0 and 1 are the real and imaginary components of the empirical characteristic
    function, respectively.

    :param _ecf_1: First empirical characteristic function
    :param _ecf_2: Second empirical characteristic function
    :return: error metric.
    """
    return mx.max(mx.abs(_ecf_1 - _ecf_2))


def ecf_compare(_ecf_1: mx.array, _ecf_2: mx.array):
    """
    Compare empirical characteristic function evaluations.

    Arrays are assumed to correspond to empirical characteristic functions that were
    evaluated at the same transform variable values.

    Array first dimension elements are per transform variable value, in increasing order.

    Array second dimension elements 0 and 1 are the real and imaginary components of the empirical characteristic
    function, respectively.

    :param _ecf_1: First empirical characteristic function
    :param _ecf_2: Second empirical characteristic function
    :return: error metric.
    """
    return mx.max(mx.sqrt(mx.square(_ecf_1[:, 0] - _ecf_2[:, 0]) + mx.square(_ecf_1[:, 1] - _ecf_2[:, 1])))


@mx.compile
def _TestSamplingProblem_f(_res1, _res2, _epts):
    ecf1 = ecfi(_res1, _epts)
    ecf2 = ecfi(_res2, _epts)
    return ecfi_compare(ecf1, ecf2)


_TestSamplingProblem_finner = mx.vmap(_TestSamplingProblem_f, in_axes=1, out_axes=0)
_TestSamplingProblem_fouter = mx.vmap(lambda r1, r2, p: _TestSamplingProblem_finner(r1, r2, p), in_axes=0, out_axes=0)


@mx.compile
def _TestSamplingProblem_fall(_r, _p):
    _r1, _r2 = mx.split(_r, 2, axis=1)
    return _TestSamplingProblem_fouter(_r1, _r2, _p)


@mx.compile
def _TestSamplingProblem_stdev_k(_results):
    res_std_m = mx.std(_results, axis=0)
    return mx.nan_to_num(mx.reciprocal(res_std_m), nan=1.0, posinf=1.0, neginf=1.0)


_TestSamplingProblem_stdev = mx.vmap(_TestSamplingProblem_stdev_k, in_axes=0)


class _TestSamplingProblem:

    def __init__(self, _results: mx.array, _num_steps: int, _num_var_pers: int, _num_iter: int):

        self.results = _results

        self.eval_pts = mx.zeros((_results.shape[0], _num_steps, _results.shape[2]), stream=mx.gpu)
        _eval_t_fin = _TestSamplingProblem_stdev(_results) * 2.0 * _num_var_pers * mx.pi
        for midx in range(_results.shape[0]):
            for idx in range(_results.shape[2]):
                self.eval_pts[midx, :, idx] = get_eval_info_times(_num_steps, _eval_t_fin[midx, idx], stream=mx.gpu)

        @partial(mx.compile, inputs=_num_iter)
        def _fiter(_r, _p):
            err = mx.zeros(_num_iter, stream=mx.gpu)
            for i in range(_num_iter):
                _r = mx.random.permutation(_r, axis=1)
                err[i] = mx.max(_TestSamplingProblem_fall(_r, _p))
            return err, _r

        self._fiter = _fiter

    def __call__(self):
        err, self.results = self._fiter(self.results, self.eval_pts)
        return err.tolist()


def test_sampling(_results: Dict[str, np.ndarray],
                  incr_sampling=100,
                  err_thresh=1E-4,
                  max_sampling: int = None,
                  num_steps: int = consts.DEF_EVAL_NUM,
                  num_var_pers: int = consts.DEF_NUM_VAR_PERS):
    var_names = list(_results.keys())

    num_vars = len(var_names)
    sample_size, num_times = _results[var_names[0]].shape

    data_mlx = mx.zeros((num_vars, sample_size, num_times), stream=mx.gpu)
    for i, n in enumerate(var_names):
        data_mlx[i, :, :] = _results[n]

    ecf_problem = _TestSamplingProblem(data_mlx, num_steps, num_var_pers, incr_sampling)
    ecf_errs = ecf_problem()

    # Do iterative work

    ecf_err_avg_curr = np.average(ecf_errs)
    iter_cur = 0
    err_curr = err_thresh + 1.0
    while err_curr >= err_thresh:
        ecf_errs.extend(ecf_problem())

        ecf_err_avg_next = np.average(ecf_errs)
        err_curr = abs(ecf_err_avg_next - ecf_err_avg_curr) / ecf_err_avg_curr

        ecf_err_avg_curr = ecf_err_avg_next
        if ecf_err_avg_curr == 0:
            break

        iter_cur += 1
        if max_sampling is not None and len(ecf_errs) >= max_sampling:
            break

    # return result
    return ecf_errs, iter_cur, err_curr
