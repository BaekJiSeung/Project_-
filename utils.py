
import numpy as np

def HPD(mcmc_samples, probability):
    sorted_samples = np.sort(mcmc_samples)
    n_samples = len(sorted_samples)
    n_interval_samples = int(np.floor(n_samples * probability))

    min_range = np.inf
    hpd_indices = [0, n_interval_samples - 1]

    for i in range(n_samples - n_interval_samples + 1):
        current_range = sorted_samples[i + n_interval_samples - 1] - sorted_samples[i]
        if current_range < min_range:
            min_range = current_range
            hpd_indices = [i, i + n_interval_samples - 1]

    hpd_lower = sorted_samples[hpd_indices[0]]
    hpd_upper = sorted_samples[hpd_indices[1]]
    return hpd_lower, hpd_upper


def ESS(mcmc_samples):
    n = len(mcmc_samples)
    max_lags = min(int(np.ceil(n / 4)), 1000)
    acf = np.correlate(mcmc_samples - np.mean(mcmc_samples), 
                       mcmc_samples - np.mean(mcmc_samples), 
                       mode='full')[n-1:]
    acf /= acf[0]  # Normalize

    sum_acf = 2 * np.sum(acf[1:max_lags + 1])
    ess = n / (1 + sum_acf)
    return ess
