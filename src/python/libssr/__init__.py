from .compute import (get_eval_info_times,
                      ecf_compare,
                      ecf,
                      err_sample,
                      eval_final,
                      find_ecfs,
                      test_reproducibility,
                      pval,
                      pvals)
from .data import EFECTReport
from .par import get_pool, start_pool, close_pool
from .utils import round_to_sigfigs, round_arr_to_sigfigs

__ssr_level__ = 0
__ssr_version__ = 0
