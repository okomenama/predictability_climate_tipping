##Test codes for Monte-carlo
import numpy as np
import monte_calro

class DonutPDF:
    def __init__(self, radius=3, sigma2=0.05):
        self.radius = radius
        self.sigma2 = sigma2

    def log_density(self, x):
        r = np.linalg.norm(x)
        return -((r - self.radius) ** 2) / self.sigma2

    def grad_log_density(self, x):
        r = np.linalg.norm(x)
        if r == 0:
            return np.zeros_like(x)
        return 2 * x * (self.radius / r - 1) / self.sigma2
    
def gaussian():
    samples = np.random.normal()
    return samples