.. _packaging:

Packaging
==========

libSSR strives to enable sharing data for reproducible stochastic simulation results at scale.
libSSR enables scalable reproducibility by providing infrastructure and tools that allow results to be
packaged, shared, and independently tested without requiring any interaction between the original authors
and those conducting the reproducibility studies.

Central to data sharing for reproducibility is the :class:`EFECT Report <EFECTReport>`,
a dataset that encodes stochastic simulation results with the necessary information to
perform a reproducibility study.
libSSR provides an implementation of the EFECT Report with support for multiple data formats.

For example, suppose someone produces stochastic simulation results for two variables ``"x"`` and ``"y"``:

.. code-block:: python

    import libssr
    import numpy as np

    def generate_sample(_times: np.ndarray, _mean: float, _stdev: float, _size: int):
        result = np.ndarray((_size, _times.shape[0]), dtype=float)
        offset = np.random.normal(_mean, _stdev, _size)
        for i in range(_size):
            result[i, :] = np.sin(_times + offset[i])
        return result

    num_steps = 21
    var_names = ['x', 'y']
    sample_size = 1000
    sig_figs = 16

    times = np.asarray([2 * np.pi * x / (num_steps - 1) for x in range(num_steps)], dtype=float)
    sample = {
        n: libssr.round_arr_to_sigfigs(generate_sample(times, 0.0, 0.25, sample_size), sig_figs)
        for n in var_names
    }

After performing the :ref:`Test for Reproducibility <comparison_wfs>`, an EFECT Report
can be generated using libSSR functions:

.. code-block:: python

    # Perform the Test for Reproducibility
    eval_num = 101
    err_sample = libssr.sample_efect_error(sample, num_steps=eval_num)[0]

    # Insert some evaluation of the EFECT Error distribution to determine
    # whether the sample is sufficiently reproducible

    # Build EFECT Report data for a half sample...
    size_exported = sample_size // 2
    # ... empirical characteristic functions
    ecf_evals = np.ndarray((num_steps, len(var_names), eval_num, 2), dtype=float)
    # ... minimum transform variable information
    ecf_tvals = np.ndarray((num_steps, len(var_names)), dtype=float)
    for i in range(num_steps):
        for j, n in enumerate(var_names):
            sample_ij = sample[n][:size_exported, i]
            ecf_tval[i, j] = libssr.eval_final(sample_ij)
            ecf_evals[i, j, :, :] = libssr.ecf(sample_ij,
                                               libssr.get_eval_info_times(eval_num, ecf_tval[i, j]))
    # Generate the EFECT Report
    sdata = libssr.EFECTReport.create(
        variable_names=var_names,
        simulation_times=times,
        sample_size=size_export,
        ecf_evals=ecf_evals,
        ecf_tval=ecf_tval,
        ecf_nval=eval_num,
        error_metric_mean=np.mean(err_sample),
        error_metric_stdev=np.std(err_sample),
        sig_figs=sig_figs
    )

An EFECT Report can be exported to a variety of formats, such as JSON:

.. code-block:: python

    import json
    # Export the EFECT Report to share
    with open('efect_report.json', 'w') as f:
        json.dump(sdata.to_json(), f, indent=4)

Exported data can be reloaded for testing against other stochastic simulation results:

.. code-block:: python

    # Load the shared EFECT Report
    with open('efect_report.json', 'r') as f:
        sdata = libssr.EFECTReport.from_json(json.load(f))
    # Generate a sample and perform the Test for Reproducibility
    sample2 = {
        n: libssr.round_arr_to_sigfigs(generate_sample(sdata.simulation_times, 0.0, 0.25, sdata.sample_size),
                                       sdata.sig_figs)
        for n in sdata.variable_names
    }
    err_sample2 = libssr.sample_efect_error(sample2, num_steps=sdata.ecf_nval)[0]
    # Compute the EFECT Error comparing this sample to the sample encoded in the EFECT Report
    err_max = 0.0
    for i in range(sdata.simulation_times.shape[0]):
        for j, n in enumerate(sdata.variable_names):
            sample2_ij = sample2[n][:, i]
            ecf1 = sdata.ecf_evals[i, j, :, :]
            ecf2 = libssr.ecf(sample2_ij, libssr.get_eval_info_times(sdata.ecf_nval, sdata.ecf_tval[i, j]))
            err_max = max(err_max, libssr.ecf_compare(ecf1, ecf2))
    # Compute the rejection p-value that results are not equal in distribution
    rejection_pval = libssr.pval(err_sample2, err_max)
