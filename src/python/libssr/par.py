import multiprocessing as mp
import numpy as np
import os
from typing import Optional

lib_pool: Optional[mp.Pool] = None
"""Module pool"""


def _seed_pool(seed):
    if seed is None:
        seed = os.getpid()
    np.random.seed(seed)
    return seed


def start_pool(num_workers: int = None):
    """
    Start the module pool and initialize each worker with a random seed

    :param num_workers: number of CPUs
    """
    if num_workers is None:
        num_workers = mp.cpu_count()

    global lib_pool
    lib_pool = mp.Pool(num_workers, initializer=_seed_pool, initargs=[None])


def get_pool():
    """Get the module pool"""
    return lib_pool


def close_pool():
    """Close the module pool (if any)"""
    global lib_pool
    lib_pool = None
