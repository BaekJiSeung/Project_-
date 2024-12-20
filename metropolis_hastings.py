import numpy as np  # numpy import
from scipy.stats import binom
class MetropolisHastings:
    def __init__(self, age, I, T, n_samples=11000, sigma=0.01):
        import numpy as np
        self.age = age
        self.I = I
        self.T = T
        self.n_samples = n_samples
        self.sigma = sigma

    def catalytic_likelihood(self, lambda_):
        import numpy as np
        from scipy.stats import binom  # scipy.stats.binom은 메서드 내부에서만 사용
        likelihood = sum(binom.logpmf(self.I, self.T, 1 - np.exp(-lambda_ * self.age)))
        return likelihood

    def run(self):
        lambda_samples = np.zeros(self.n_samples)  # np 사용
        accepted = 0
        current_lambda = 0.15
        current_log_likelihood = self.catalytic_likelihood(current_lambda)

        for i in range(self.n_samples):
            proposed_lambda = current_lambda + np.random.randn() * self.sigma  # np 사용
            if proposed_lambda < 0 or proposed_lambda > 1:
                lambda_samples[i] = current_lambda
                continue

            proposed_log_likelihood = self.catalytic_likelihood(proposed_lambda)
            log_acceptance_ratio = proposed_log_likelihood - current_log_likelihood

            if np.log(np.random.rand()) < log_acceptance_ratio:  # np 사용
                current_lambda = proposed_lambda
                current_log_likelihood = proposed_log_likelihood
                accepted += 1

            lambda_samples[i] = current_lambda

        acceptance_prob = accepted / self.n_samples
        return lambda_samples, acceptance_prob
