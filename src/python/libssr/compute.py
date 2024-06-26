import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
from typing import Dict, List

from . import consts
from . import par

if consts.has_numba:
    import numba


def get_eval_info_times(_eval_num: int, _eval_fin: float):
    return np.linspace(0.0, _eval_fin, _eval_num)


if consts.has_numba:
    get_eval_info_times = numba.njit(get_eval_info_times)


def eval_final(sample: np.ndarray, num_var_pers=consts.DEF_NUM_VAR_PERS):
    """
    Get the parameterized final transform variable value for evaluating an empirical characteristic function

    :param sample: sample to which to parameterize
    :param num_var_pers: number of parameterization periods of the empirical characteristic function
    :return:
    """
    v_stdev = np.std(sample)
    return 1.0 if v_stdev == 0 else 2 * num_var_pers * np.pi / v_stdev


if consts.has_numba:
    eval_final = numba.njit(eval_final)


def ecf_compare(_ecf_1: np.ndarray, _ecf_2: np.ndarray) -> float:
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
    return np.max(np.sqrt(np.square(_ecf_1[:, 0] - _ecf_2[:, 0]) + np.square(_ecf_1[:, 1] - _ecf_2[:, 1])))


if consts.has_numba:
    ecf_compare = numba.njit(ecf_compare)


def ecf(var_vals: np.ndarray, func_evals: np.ndarray) -> np.ndarray:
    """
    Evaluate the empirical characteristic function of a sample of a variable
    at a simulation time at given transform variable values.

    :param var_vals: trajectory values of a variable at a simulation time
    :param func_evals: transform variable values at which to compute the empirical characteristic function
    :return: empirical characteristic function evaluations; dim 0 is evaluations; dim 1 is real and imaginary components
    """
    result = np.zeros((func_evals.shape[0], 2))

    func_evals_mat = np.repeat(func_evals[:, np.newaxis], var_vals.shape[0], 1)
    var_vals_mat = np.repeat(var_vals[:, np.newaxis], func_evals.shape[0], 1).T
    x = func_evals_mat * var_vals_mat
    result[:, 0] = np.average(np.cos(x), 1)
    result[:, 1] = np.average(np.sin(x), 1)

    return result


def _ecf_njit(var_vals: np.ndarray, func_evals: np.ndarray):
    """
    Evaluate the empirical characteristic function of a sample of a variable
    at a simulation time at given transform variable values.

    :param var_vals: trajectory values of a variable at a simulation time
    :param func_evals: transform variable values at which to compute the empirical characteristic function
    :return: empirical characteristic function evaluations; dim 0 is evaluations; dim 1 is real and imaginary components
    """
    result = np.zeros((func_evals.shape[0], 2))

    for i in range(func_evals.shape[0]):
        t = func_evals[i]
        result[i, 0] = np.average(np.cos(var_vals * t))
        result[i, 1] = np.average(np.sin(var_vals * t))

    return result


if consts.has_numba:
    ecf = numba.njit(_ecf_njit)


def _ecf_err(results: np.ndarray, num_steps: int, num_var_pers: int):
    res_std = np.std(results)
    if res_std == 0.0:
        incr_max = 1 / num_steps
        err = 0.0
    else:

        eval_t_fin = 2 * num_var_pers * np.pi / res_std
        incr_max = eval_t_fin / num_steps

        eval_pts = get_eval_info_times(num_steps, eval_t_fin)
        n = int(results.shape[0] / 2)
        ecf1 = ecf(results[:n], eval_pts)
        ecf2 = ecf(results[n:], eval_pts)
        err = ecf_compare(ecf1, ecf2)

    return incr_max, err


if consts.has_numba:
    _ecf_err = numba.njit(_ecf_err)


def _ecf_err_s(results: np.ndarray, num_steps: int, num_var_pers: int):
    res_std = np.std(results)
    if res_std == 0.0:
        err = 0.0
    else:

        eval_t_fin = 2 * num_var_pers * np.pi / res_std

        eval_pts = get_eval_info_times(num_steps, eval_t_fin)
        n = int(results.shape[0] / 2)
        ecf1 = ecf(results[:n], eval_pts)
        ecf2 = ecf(results[n:], eval_pts)
        err = ecf_compare(ecf1, ecf2)

    return err


if consts.has_numba:
    _ecf_err = numba.njit(_ecf_err_s)


def err_sample(_results: Dict[str, np.ndarray],
               num_steps: int = consts.DEF_EVAL_NUM,
               num_var_pers: int = consts.DEF_NUM_VAR_PERS):
    """
    Evaluate the error for all variables at a simulation time when dividing a sample into two evenly-sized subsamples.

    :param _results: trajectories by variable name
    :param num_steps: number of transform variable evaluations
    :param num_var_pers: number of parameterization periods of the empirical characteristic function
    :return: error metric by variable name, final transform variable value by name
    """
    res_err = {}
    res_eval_fin = {}
    for k, res in _results.items():
        incr_max, res_err[k] = _ecf_err(res, num_steps, num_var_pers)
        res_eval_fin[k] = incr_max * num_steps

    return res_err, res_eval_fin


def _find_ecfs(_results: Dict[str, np.ndarray],
               idx: int,
               num_steps: int,
               num_var_pers: int):
    eval_fin = {}
    for k, v in _results.items():
        eval_fin[k] = eval_final(v, num_var_pers)
    res_ecfs = {n: ecf(_results[n], get_eval_info_times(num_steps, eval_fin[n])) for n in _results.keys()}
    return idx, res_ecfs, eval_fin


def find_ecfs(_results: Dict[str, np.ndarray],
              num_steps: int = None,
              num_var_pers: int = None,
              num_workers: int = None):
    """
    Find the empirical characteristic functions of a sample.

    :param _results: trajectories by name; dim 0 is by realization; dim 1 is by time
    :param num_steps: number of transform variable evaluations
    :param num_var_pers: number of parameterization periods of the empirical characteristic function
    :param num_workers: number of CPUs
    :return: empirical characteristic functions and final transform variable value, by name and time
    """
    result_ecf = {name: [None for _ in range(_results[name].shape[1])] for name in _results.keys()}
    eval_info = {name: [None for _ in range(_results[name].shape[1])] for name in _results.keys()}

    if num_steps is None:
        num_steps = consts.DEF_EVAL_NUM
    if num_var_pers is None:
        num_var_pers = consts.DEF_NUM_VAR_PERS

    input_args = []
    for idx in range(_results[list(_results.keys())[0]].shape[1]):
        input_args.append((
            {name: _results[name][:, idx].T for name in _results.keys()},
            idx,
            num_steps,
            num_var_pers
        ))

    pool = par.get_pool()
    if pool is None:
        if num_workers is None:
            num_workers = mp.cpu_count()
        pool = mp.Pool(num_workers)

    for idx, res_ecf, eval_fin in pool.starmap(_find_ecfs, input_args):
        for name in res_ecf.keys():
            result_ecf[name][idx] = res_ecf[name]
            eval_info[name][idx] = eval_fin[name]

    return result_ecf, eval_info


def _test_sampling_impl_shared(_results: np.ndarray,
                               _indices: np.ndarray,
                               _hsize: int,
                               _num_times: int,
                               _num_steps: int,
                               _num_var_pers: int):
    err = np.zeros((_num_times,))
    for idx in range(_num_times):
        res_std = np.std(_results[:, idx])
        if res_std == 0.0:
            continue

        eval_t_fin = 2.0 * _num_var_pers * np.pi / res_std
        eval_pts = get_eval_info_times(_num_steps, eval_t_fin)
        ecf1 = ecf(_results[_indices[:_hsize], idx], eval_pts)
        ecf2 = ecf(_results[_indices[_hsize:], idx], eval_pts)
        err[idx] = ecf_compare(ecf1, ecf2)
    return np.max(err)


if consts.has_numba:
    _test_sampling_impl_shared = numba.njit(_test_sampling_impl_shared)


def _test_sampling_shared_(results: List[np.ndarray],
                           indices: np.ndarray,
                           num_results: int,
                           num_times: int,
                           num_steps: int,
                           num_var_pers: int):

    out_arr = np.zeros((num_results,), dtype=float)
    hsize = results[0].shape[0] // 2

    for i in range(num_results):
        np.random.shuffle(indices)
        err = 0.0
        for res in results:
            err = max(err, _test_sampling_impl_shared(res, indices, hsize, num_times, num_steps, num_var_pers))
        out_arr[i] = err

    return out_arr


if consts.has_numba:
    _test_sampling_shared_ = numba.njit(_test_sampling_shared_)


def _test_sampling_shared(_shm_in_info: Dict[str, str],
                          _shm_out_info: str,
                          _shm_out_idx: int,
                          _shm_out_len: int,
                          _arr_shape0: int,
                          _arr_shape1: int,
                          _num_results: int,
                          _num_steps: int,
                          _num_var_pers: int) -> bool:
    # Ensure worker has unique seed
    np.random.seed()
    # Get shared data
    shm_in = {k: shared_memory.SharedMemory(name=v) for k, v in _shm_in_info.items()}
    shm_out = shared_memory.SharedMemory(name=_shm_out_info)

    shm_out_arr = np.ndarray((_shm_out_len,), dtype=float, buffer=shm_out.buf)
    _results = [np.ndarray((_arr_shape0, _arr_shape1), dtype=float, buffer=v.buf) for v in shm_in.values()]
    indices = np.asarray(list(range(_arr_shape0)), dtype=int)

    out_arr = _test_sampling_shared_(_results, indices, _num_results, _arr_shape1, _num_steps, _num_var_pers)

    shm_out_arr[_shm_out_idx:_shm_out_idx + _num_results] = out_arr[:]
    return True


def test_sampling_shared(_results: Dict[str, np.ndarray],
                         incr_sampling=100,
                         err_thresh=1E-4,
                         max_sampling: int = None,
                         num_steps: int = consts.DEF_EVAL_NUM,
                         num_var_pers: int = consts.DEF_NUM_VAR_PERS,
                         num_workers: int = None):
    var_names = list(_results.keys())

    # Allocate shared memory
    shm_to = {k: shared_memory.SharedMemory(create=True, size=v.nbytes) for k, v in _results.items()}
    shm_to_arr = {k: np.ndarray(v.shape, dtype=v.dtype, buffer=shm_to[k].buf) for k, v in _results.items()}
    for k, v in _results.items():
        shm_to_arr[k][:] = v[:]
    shm_to_info = {k: v.name for k, v in shm_to.items()}

    from_arr = np.ndarray((incr_sampling,), dtype=float)
    shm_from = shared_memory.SharedMemory(create=True, size=from_arr.nbytes)
    shm_from_arr = np.ndarray(from_arr.shape, dtype=from_arr.dtype, buffer=shm_from.buf)

    # Do stuff
    sample_size, num_times = _results[var_names[0]].shape

    ecf_errs = []

    # Do initial work

    if num_workers is None:
        num_workers = mp.cpu_count()
    num_workers = min(incr_sampling, num_workers)

    num_jobs = [0 for _ in range(num_workers)]
    jobs_left = int(incr_sampling)
    while jobs_left > 0:
        for i in range(num_workers):
            if jobs_left > 0:
                num_jobs[i] += 1
                jobs_left -= 1
    num_jobs = [n for n in num_jobs if n > 0]
    num_workers = len(num_jobs)
    job_indices = [ji - num_jobs[i] for i, ji in enumerate(np.cumsum(num_jobs))]

    if sum(num_jobs) != incr_sampling:
        raise RuntimeError(f'Scheduled {sum(num_jobs)} jobs, though {incr_sampling} jobs were requested')

    pool = par.get_pool()
    if pool is None:
        pool = mp.Pool(num_workers)

    input_args = [(shm_to_info,
                   shm_from.name,
                   job_indices[i],
                   incr_sampling,
                   sample_size,
                   num_times,
                   num_jobs[i],
                   num_steps,
                   num_var_pers)
                  for i in range(num_workers)]

    pool.starmap(_test_sampling_shared, input_args)

    from_arr[:] = shm_from_arr[:]
    ecf_errs.extend(from_arr.tolist())

    # Do iterative work

    ecf_err_avg_curr = np.average(ecf_errs)
    iter_cur = 0
    err_curr = err_thresh + 1.0
    while err_curr >= err_thresh:
        pool.starmap(_test_sampling_shared, input_args)

        from_arr[:] = shm_from_arr[:]
        ecf_errs.extend(from_arr.tolist())

        ecf_err_avg_next = np.average(ecf_errs)
        err_curr = abs(ecf_err_avg_next - ecf_err_avg_curr) / ecf_err_avg_curr

        ecf_err_avg_curr = ecf_err_avg_next
        if ecf_err_avg_curr == 0:
            break

        iter_cur += 1
        if max_sampling is not None and len(ecf_errs) >= max_sampling:
            break

    # Free shared memory
    for m in shm_to.values():
        m.close()
        m.unlink()
    shm_from.close()
    shm_from.unlink()

    # return result
    return ecf_errs, iter_cur, err_curr


def test_reproducibility(_results: Dict[str, np.ndarray],
                         incr_sampling=100,
                         err_thresh=1E-4,
                         max_sampling: int = None,
                         num_steps: int = consts.DEF_EVAL_NUM,
                         num_var_pers: int = consts.DEF_NUM_VAR_PERS,
                         num_workers: int = None):
    """
    Perform the test for reproducibility on a sample.

    :param _results: trajectories by name; dim 0 is by realization; dim 1 is by time
    :param incr_sampling: number of additional trajectories when increasing sample size
    :param err_thresh: convergence criterion
    :param max_sampling: maximum error metric sample size
    :param num_steps: number of transform variable evaluations
    :param num_var_pers: number of parameterization periods of the empirical characteristic function
    :param num_workers: number of CPUs
    :return: error metric sample, number of iterations, final convergence value
    """
    return test_sampling_shared(
        _results, incr_sampling, err_thresh, max_sampling, num_steps, num_var_pers, num_workers
    )


def pvals(err_dist_mean: float, err_dist_stdev: float, err_compare: float, sample_size: int) -> float:
    """
    Calculate the p-value of an error metric from comparison to another sample for an error metric distribution
    when testing a sample for reproducibility.

    :param err_dist_mean: distribution of the error metric mean when testing for reproducibility
    :param err_dist_stdev: distribution of the error metric standard deviation when testing for reproducibility
    :param err_compare: error metric when comparing to another sample
    :param sample_size: size of the sample
    :return: p-value
    """

    if err_compare < err_dist_mean:
        return 1.0
    q2 = (sample_size + 1) / sample_size * err_dist_stdev * err_dist_stdev
    lam2 = (err_compare - err_dist_mean) * (err_compare - err_dist_mean) / q2
    pr = np.floor((sample_size + 1) / sample_size * ((sample_size - 1) / lam2 + 1)) / (sample_size + 1)
    return min(1.0, pr)


def pval(err_dist: np.ndarray, err_compare: float, sample_size: int) -> float:
    """
    Calculate the p-value of an error metric from comparison to another sample for an error metric distribution
    when testing a sample for reproducibility.

    :param err_dist: distribution of the error metric when testing for reproducibility
    :param err_compare: error metric when comparing to another sample
    :param sample_size: size of the sample
    :return: p-value
    """

    return pvals(np.average(err_dist), np.std(err_dist, ddof=1), err_compare, sample_size)
