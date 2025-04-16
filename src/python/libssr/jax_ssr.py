import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict

from . import consts


def get_eval_info_times(_eval_num: int, _eval_fin: float):
    return jnp.linspace(0.0, _eval_fin, _eval_num)


@jax.jit
def ecf_compare(_ecf_1: jnp.ndarray, _ecf_2: jnp.ndarray):
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
    return jnp.sqrt(jnp.max(jnp.sum(jnp.square(_ecf_1 - _ecf_2), axis=1)))


@jax.jit
def _ecf(var_vals: jnp.ndarray, func_eval):
    x = var_vals * func_eval
    return jnp.stack([jnp.average(jnp.cos(x)), jnp.average(jnp.sin(x))])


_ecf_vmap = jax.vmap(_ecf, in_axes=(None, 0), out_axes=0)


@jax.jit
def ecf(var_vals: jnp.ndarray, func_evals: jnp.ndarray):
    """
    Evaluate the empirical characteristic function of a sample of a variable
    at a simulation time at given transform variable values.

    :param var_vals: trajectory values of a variable at a simulation time
    :param func_evals: transform variable values at which to compute the empirical characteristic function
    :return: empirical characteristic function evaluations; dim 0 is evaluations; dim 1 is real and imaginary components
    """
    return _ecf_vmap(var_vals, func_evals)


@jax.jit
def _test_name(key, result_name_time, eval_pts, res_std):
    res_shuffle = jax.random.permutation(key, result_name_time)
    res1, res2 = jnp.split(res_shuffle, 2)
    ecf1 = ecf(res1, eval_pts)
    ecf2 = ecf(res2, eval_pts)
    return jax.lax.select(
        jax.lax.gt(res_std, 0.0),
        ecf_compare(ecf1, ecf2),
        jnp.zeros_like(res_std)
    )


test_name_vmap = jax.vmap(_test_name, in_axes=(None, 1, 1, 0))
test_name_rep_vmap = jax.vmap(lambda k, r, p, s: jnp.max(test_name_vmap(k, r, p, s)),
                              in_axes=(0, None, None, None))


@jax.jit
def res_std_time(results):
    return jnp.std(results)


res_std_time_vmap = jax.vmap(res_std_time, in_axes=1)


def _make_eval_info_t(res_std, num_steps, num_var_pers):
    eval_t_fin = jnp.true_divide(1., res_std) * 2.0 * num_var_pers * np.pi
    return get_eval_info_times(num_steps, eval_t_fin)


make_eval_info_t = jax.jit(_make_eval_info_t, static_argnums=(1, 2))


def _gen_(v, num_steps, num_var_pers):
    res_std_v = res_std_time_vmap(v)
    eval_info_times_v = make_eval_info_t(res_std_v, num_steps, num_var_pers)
    return v, eval_info_times_v, res_std_v


_gen = jax.jit(_gen_, static_argnums=(1, 2))


def test_reproducibility(_results: Dict[str, np.ndarray],
                         incr_sampling=100,
                         err_thresh=1E-4,
                         max_sampling: int = None,
                         num_steps: int = consts.DEF_EVAL_NUM,
                         num_var_pers: int = consts.DEF_NUM_VAR_PERS):
    """
    Perform the test for reproducibility on a sample.

    :param _results: trajectories by name; dim 0 is by realization; dim 1 is by time
    :param incr_sampling: number of additional trajectories when increasing sample size
    :param err_thresh: convergence criterion
    :param max_sampling: maximum error metric sample size
    :param num_steps: number of transform variable evaluations
    :param num_var_pers: number of parameterization periods of the empirical characteristic function
    :return: error metric sample, number of iterations, final convergence value
    """
    var_names = list(_results.keys())

    # Do stuff
    # sample_size, num_times = _results[var_names[0]].shape

    # Do initial work
    key = jax.random.PRNGKey(np.random.randint(int(1E12)))
    # results_jax = [jnp.array(v) for v in _results.values()]
    # results_jax = [v for v in _results.values()]

    mesh = jax.make_mesh((jax.device_count(),), ('a',))
    part = jax.sharding.PartitionSpec(('a',))
    named_sharding = jax.sharding.NamedSharding(mesh, part)

    def _do_sample():
        nonlocal key
        key, subkey = jax.random.split(key)
        subkeys = jax.random.split(subkey, incr_sampling)
        subkeys = jax.device_put(subkeys, named_sharding)
        err_i = []
        for v in _results.values():
            err_i.append(test_name_rep_vmap(subkeys, *_gen(jnp.array(v), num_steps, num_var_pers)))
        return jnp.max(jnp.column_stack(err_i), axis=1).tolist()

    ecf_errs = _do_sample()

    # Do iterative work

    ecf_err_avg_curr = np.average(ecf_errs)
    iter_cur = 0
    err_curr = err_thresh + 1.0

    while err_curr >= err_thresh:
        ecf_errs.extend(_do_sample())

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
