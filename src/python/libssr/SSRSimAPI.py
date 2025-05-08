"""
Simulator API
"""

import numpy as np
from typing import Dict, List, Union


class SSRSimAPI:

    def load_model(self, *args, **kwargs):
        """
        Perform initializations for executing simulations on demand.

        The signature of this method is implementation-specific.

        :param args: positional arguments
        :param kwargs: keyword arguments
        :return: None
        """
        raise NotImplementedError

    def produce_results(self,
                        names: List[str],
                        times: List[float],
                        sample_info: Union[int, List[Dict[str, float]]]) -> Dict[str, np.ndarray]:
        """
        Produce a sample of simulation results.

        The sample is formatted by output name and data, where data is formatted as replicate (dim 0) by time (dim 1).

        Sample specification (via `sample_info`) can define a sample size,
        or a list of modifications to impose per replicate.
        Modifications per replicate are specified by model name and value.
        For example, `[{'a': 1.0}]` specifies a sample size of 1,
        where the model parameter 'a' of the replicate is set to 1.0 before execution.

        :param names: names of model variables to record in the sample.
        :param times: simulation times at which to record in the sample.
        :param sample_info: sample specification.
        :return: a sample
        """
        raise NotImplementedError
