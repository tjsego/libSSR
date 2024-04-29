"""
Demonstrates testing equality of samples generated from different distributions with the
same summary statistics.

Requires matplotlib, scipy
"""
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats.sampling import NumericalInversePolynomial

import sbsr


def as_si(x, ndp):
    s = '{x:0.{ndp:d}e}'.format(x=x, ndp=ndp)
    m, e = s.split('e')
    st = r'{m:s}\times 10^{{{e:d}}}'.format(m=m, e=int(e))
    return r"${0:s}$".format(st)


class NormalDistribution:

    def __init__(self, _mean, _variance):
        self.a = _mean
        self.b = np.sqrt(_variance)

    def pdf(self, x):
        dx = (x - self.a) / self.b
        dx2 = np.multiply(dx, dx) * -0.5
        return np.exp(dx2) / self.b / np.sqrt(np.pi * 2)


class LaplaceDistribution:

    def __init__(self, _mean, _variance):
        self.a = _mean
        self.b = np.sqrt(_variance / 2)

    def pdf(self, x):
        return np.exp(np.abs(x - self.a) / -self.b) / self.b / 2


class UniformDistribution:

    def __init__(self, _mean, _variance):
        self.w = np.sqrt(12 * _variance)
        self.a = _mean - self.w / 2

    def pdf(self, x):
        if isinstance(x, float):
            return 1 / self.w if self.a <= x <= self.a + self.w else 0.0
        result = np.zeros_like(x)
        result[x >= self.a] = 1 / self.w
        result[x > self.a + self.w] = 0.0
        return result


class LogisticDistribution:

    def __init__(self, _mean, _variance):
        self.a = _mean
        self.b = np.sqrt(3 * _variance) / np.pi

    def pdf(self, x):
        f = np.exp((x - self.a) / -self.b)
        g = f + 1
        return np.divide(f, np.multiply(g, g)) / self.b


if __name__ == '__main__':
    mean = 0.0
    variance = 4.0

    x_test = np.arange(start=-10.0, stop=10.0, step=0.01)

    fig_dist, ax_dist = plt.subplots(1, 2, figsize=(8, 3))
    ax_dist[0].plot(x_test, NormalDistribution(mean, variance).pdf(x_test), label='Normal')
    ax_dist[0].plot(x_test, LaplaceDistribution(mean, variance).pdf(x_test), label='Laplace')
    ax_dist[0].plot(x_test, UniformDistribution(mean, variance).pdf(x_test), label='Uniform')
    ax_dist[0].plot(x_test, LogisticDistribution(mean, variance).pdf(x_test), label='Logistic')
    ax_dist[0].legend()

    urng = np.random.default_rng()
    dist_norm = NumericalInversePolynomial(NormalDistribution(mean, variance), random_state=urng)
    dist_lplc = NumericalInversePolynomial(LaplaceDistribution(mean, variance), random_state=urng)
    dist_ufrm = NumericalInversePolynomial(UniformDistribution(mean, variance), random_state=urng)
    dist_lgsc = NumericalInversePolynomial(LogisticDistribution(mean, variance), random_state=urng)

    ax_dist[1].plot(x_test, dist_norm.cdf(x_test), label='Normal')
    ax_dist[1].plot(x_test, dist_lplc.cdf(x_test), label='Laplace')
    ax_dist[1].plot(x_test, dist_ufrm.cdf(x_test), label='Uniform')
    ax_dist[1].plot(x_test, dist_lgsc.cdf(x_test), label='Logistic')

    sample_size = 10000

    sample_norm = dist_norm.rvs(sample_size)
    sample_lplc = dist_lplc.rvs(sample_size)
    sample_ufrm = dist_ufrm.rvs(sample_size)
    sample_lgsc = dist_lgsc.rvs(sample_size)

    sample_norm_2 = dist_norm.rvs(sample_size)
    sample_lplc_2 = dist_lplc.rvs(sample_size)
    sample_ufrm_2 = dist_ufrm.rvs(sample_size)
    sample_lgsc_2 = dist_lgsc.rvs(sample_size)

    print('Sample 1 mean:', np.mean(sample_norm), np.mean(sample_lplc), np.mean(sample_ufrm), np.mean(sample_lgsc))
    print('Sample 2 mean:', np.mean(sample_norm_2), np.mean(sample_lplc_2), np.mean(sample_ufrm_2), np.mean(sample_lgsc_2))
    print('Sample 1 variance:', np.var(sample_norm), np.var(sample_lplc), np.var(sample_ufrm), np.var(sample_lgsc))
    print('Sample 2 variance:', np.var(sample_norm_2), np.var(sample_lplc_2), np.var(sample_ufrm_2), np.var(sample_lgsc_2))

    fig_hist, ax_hist = plt.subplots(1, 1)
    alpha = 0.1
    bins = 30
    ax_hist.hist(sample_norm, alpha=alpha, bins=bins, label='Normal')
    ax_hist.hist(sample_lplc, alpha=alpha, bins=bins, label='Laplace')
    ax_hist.hist(sample_ufrm, alpha=alpha, bins=bins, label='Uniform')
    ax_hist.hist(sample_lgsc, alpha=alpha, bins=bins, label='Logistic')
    ax_hist.set_xlim(-10, 10)
    ax_hist.legend()

    eval_num = 100
    eval_per = 3
    eval_fin = 2 * eval_per * np.pi / np.sqrt(variance)
    eval_t = sbsr.get_eval_info_times(eval_num, eval_fin)

    ecf_norm = sbsr.ecf(sample_norm, eval_t)
    ecf_lplc = sbsr.ecf(sample_lplc, eval_t)
    ecf_ufrm = sbsr.ecf(sample_ufrm, eval_t)
    ecf_lgsc = sbsr.ecf(sample_lgsc, eval_t)

    ecf_norm_2 = sbsr.ecf(sample_norm_2, eval_t)
    ecf_lplc_2 = sbsr.ecf(sample_lplc_2, eval_t)
    ecf_ufrm_2 = sbsr.ecf(sample_ufrm_2, eval_t)
    ecf_lgsc_2 = sbsr.ecf(sample_lgsc_2, eval_t)

    ecf_all = [ecf_norm, ecf_lplc, ecf_ufrm, ecf_lgsc]
    ecf_all_2 = [ecf_norm_2, ecf_lplc_2, ecf_ufrm_2, ecf_lgsc_2]
    labels_all = ['Normal', 'Laplace', 'Uniform', 'Logistic']
    err_mat = np.ndarray((len(ecf_all), len(ecf_all)), dtype=float)

    for i, ecf_i in enumerate(ecf_all):
        for j, ecf_j in enumerate(ecf_all_2):
            err_mat[i, j] = sbsr.ecf_compare(ecf_i, ecf_j)

    fig_hm, ax_hm = plt.subplots()
    im = ax_hm.imshow(err_mat, cmap='rainbow')

    ax_hm.set_xticks(np.arange(len(labels_all)), labels=labels_all)
    ax_hm.set_yticks(np.arange(len(labels_all)), labels=labels_all)
    plt.setp(ax_hm.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    for i in range(len(labels_all)):
        for j in range(len(labels_all)):
            text = ax_hm.text(j, i, as_si(err_mat[i, j], 2), ha='center', va='center', color='w')
    ax_hm.set_title('Error metric')
    fig_hm.colorbar(im)

    fig_ecf, ax_ecf = plt.subplots(1, 2, figsize=(8, 3))
    for i in range(2):
        ax_ecf[i].plot(eval_t, ecf_norm[:, i], label='Normal')
        ax_ecf[i].plot(eval_t, ecf_lplc[:, i], label='Laplace')
        ax_ecf[i].plot(eval_t, ecf_ufrm[:, i], label='Uniform')
        ax_ecf[i].plot(eval_t, ecf_lgsc[:, i], label='Logistic')
    ax_ecf[0].set_title('Real')
    ax_ecf[1].set_title('Imaginary')
    ax_ecf[0].legend()

    plt.show()
