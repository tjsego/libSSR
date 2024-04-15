import numpy as np
from typing import List


class SupportingData:
    """
    SBSR supporting data
    """

    def __init__(self):

        self.level: int = 0
        """SBSR level"""

        self.version: int = 0
        """SBSR version"""

        self.variable_names: List[str] = []
        """Variable names"""

        self.simulation_times: np.ndarray = np.ndarray(())
        """Simulation times (1D float array)"""

        self.sample_size: int = 0
        """Size of sample"""

        self.ecf_evals: np.ndarray = np.ndarray(())
        """
        ECF evaluations
        
        - dim 0 is by simulation time
        - dim 1 is by variable name
        - dim 2 is by transform variable evaluation
        - dim 3 is real/imaginary
        """

        self.ecf_tval: np.ndarray = np.ndarray(())
        """
        ECF transform variable final value
        
        - dim 0 is by simulation time
        - dim 1 is by variable name
        """

        self.ecf_nval: int = 0
        """Number of ECF evaluations per time and name"""

        self.error_metric_mean: float = 0.0
        """Error metric mean, from test for reproducibility"""

        self.error_metric_stdev: float = 0.0
        """Error metric standard deviation, from test for reproducibility"""

        self.sig_figs: int = 0
        """Significant figures of sample data"""


def verify_data(inst: SupportingData):
    if not inst.variable_names:
        raise ValueError('No variable names')
    if inst.sample_size <= 0:
        raise ValueError(f'Incorrect sample size ({inst.sample_size})')
    if inst.ecf_nval <= 0:
        raise ValueError(f'Incorrect number of ECF evaluations ({inst.ecf_nval})')
    if inst.sig_figs <= 0:
        raise ValueError(f'Incorrect number of significant figures ({inst.sig_figs})')
    if inst.ecf_evals.shape[0] != len(inst.simulation_times):
        raise ValueError(f'Incorrect ECF shape: times ({inst.ecf_evals.shape[0], len(inst.simulation_times)})')
    if inst.ecf_tval.shape[0] != len(inst.simulation_times):
        raise ValueError(
            f'Incorrect ECF transform variable shape: times ({inst.ecf_tval.shape[0], len(inst.simulation_times)})'
        )
    if inst.ecf_evals.shape[1] != len(inst.variable_names):
        raise ValueError(f'Incorrect ECF shape: names ({inst.ecf_evals.shape[1], len(inst.variable_names)})')
    if inst.ecf_tval.shape[1] != len(inst.variable_names):
        raise ValueError(
            f'Incorrect ECF transform variable shape: names ({inst.ecf_tval.shape[1], len(inst.variable_names)})'
        )
    if inst.ecf_evals.shape[2] != inst.ecf_nval:
        raise ValueError(f'Incorrect ECF shape: evaluations ({inst.ecf_evals.shape[2], inst.ecf_nval})')

    return inst
