.. _simulators:

Simulator API
==============

A simulator can provide general support for libSSR reproducibility workflows by providing an implementation of
:class:`SSRSimAPI <SSRSimAPI.SSRSimAPI>`.

An implementation defines two methods:

* :py:meth:`load_model <SSRSimAPI.SSRSimAPI.load_model>`: performs initializations for executing simulations on demand.

  * Inputs: variable arguments and keyword arguments

* :py:meth:`produce_results <SSRSimAPI.SSRSimAPI.produce_results>`: produces a sample of simulation results

  * Inputs:

    * ``names: List[str]``: Names of model variables to record in the sample.
    * ``times: List[float]``: Simulation times at which to record in the sample.
    * ``sample_info: Union[int, List[Dict[str, float]]]``: Either a sample size (as ``int``) or list of modifications per sample (*e.g.*, ``[{'a': 1.0}]``)

  * Outputs: Results by variable name as Numpy arrays (index 0: replicate; index 1: simulation time)

For example, `BasiCO <https://github.com/copasi/basico>`_ provides an implementation
that can be used with other implementations to automate reproducibility tests like the following:

.. code-block:: python

    import libssr
    import numpy as np
    from basico.ssr.CopasiSSR import CopasiSSR

    # Storage for all available implementations
    sims = []
    # Instantiate the BasiCO implementation
    with open('model.xml', 'r') as f:
        model_spec = f.read()
    sim_cp = CopasiSSR()
    sim_cp.load_model(model_spec)
    sims.append(sim_cp)

    # ... Instantiate other implementations here ...

    # Generate samples
    model_names = ['x', 'y']
    results_times = np.linspace(0.0, 10.0, 101)
    sample_size = 1000
    results = [sim.produce_results(model_names, results_times, sample_size) for sim in sims]
    # Test reproducibility
    err_samples = [libssr.sample_efect_error(res)[0] for res in results]
    # Continue to EFECT Report generation and simulator comparison...
