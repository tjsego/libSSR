.. _comparison_wfs:

Comparing Results
==================

Every reproducibility problem with libSSR consists of one or more named real-valued variables
that are recorded at one or more simulation times during repeated simulation execution.
libSSR can quantify the reproducibility of such results when formatted as a dictionary of
variable names as key and formatted NumPy arrays as values, where the first index is
by simulation replicate, and the second index is by simulation time.
As a simple example, the following function generates a sine wave with
offset sampled from a normal distribution in correct format:

.. code-block:: python

    import numpy as np

    def generate_sample(_times: np.ndarray, _mean: float, _stdev: float, _size: int):
        result = np.ndarray((_size, _times.shape[0]), dtype=float)
        offset = np.random.normal(_mean, _stdev, _size)
        for i in range(_size):
            result[i, :] = np.sin(_times + offset[i])
        return result

Data for a sample of size 1,000 with one variable and 21 uniformly distributed simulation steps
can be generated as follows:

.. code-block:: python

    num_steps = 21      # Number of simulation steps
    sample_size = 1000  # Size of the sample

    # Generate simulation times
    times = np.asarray([2 * np.pi * x / (num_steps - 1) for x in range(num_steps)], dtype=float)
    # Generate sample data
    sample = generate_sample(times, 0.0, 0.25, sample_size)

To mitigate false negatives due to rounding errors in certain edge cases,
data can be rounded to a fixed numerical precision by number of significant figures:

.. code-block:: python

    import libssr

    # Number of significant figures of the data
    sig_figs = 16
    # Round data
    sample = libssr.round_arr_to_sigfigs(sample, sig_figs)

The EFECT Test for Reproducibility produces the data for quantitatively testing reproducibility.
The test requires a name for each evaluated variable and, among other optional details,
uses a given number of values at which to evaluate empirical characteristic functions that form the
basis of EFECT:

.. code-block:: python

    var_name = 'Y'  # Variable name
    eval_num = 101  # Number of transform variable evaluations
    # Only capture EFECT Error sampling
    err_sample = libssr.test_reproducibility({var_name: sample}, num_steps=eval_num)[0]

The returned EFECT Error sample from the Test for Reproducibility can be used to test the
hypothesis that the tested sample and another sample are realizations of the
same distribution (*i.e.*, whether they are equal in distribution).
In general, EFECT Error distributions closer to zero are more reproducible.
Importantly, the test is valid for samples with size equal to half of the sample used in the Test for Reproducibility:

.. code-block:: python

    # Extract half of the original sample for comparison
    sample1 = sample[:sample_size // 2, :]
    # Generate another sample of the same (half) size
    sample2 = libssr.round_arr_to_sigfigs(generate_sample(times, 0.0, 0.25, sample_size // 2), sig_figs)
    # Compute the EFECT Error comparing the two samples
    err_max = 0.0
    for i in range(num_steps):
        eval_t = libssr.get_eval_info_times(eval_num, sample1[:, i])
        ecf1 = libssr.ecf(sample1[:, i], eval_t)
        ecf2 = libssr.ecf(sample2[:, i], eval_t)
        err_max = max(err_max, libssr.ecf_compare(ecf1, ecf2))
    # Compute the hypothesis rejection p-value
    rejection_pval = libssr.pval(err_sample, err_max)
