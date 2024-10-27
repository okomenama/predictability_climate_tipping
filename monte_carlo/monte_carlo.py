###Hamiltonian Monte-Calro 
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Video
from matplotlib.animation import FFMpegWriter

class Monte_carlo(object):
    ##Object variables defined 
    ##initial, target : function class
    #    class DonutPDF:
    #        def __init__(self, radius=3, sigma2=0.05):
    #            self.radius = radius
    #            self.sigma2 = sigma2
    #
    #        def log_density(self, x):
    #            r = np.linalg.norm(x)
    #            return -((r - self.radius) ** 2) / self.sigma2
    #
    #       def grad_log_density(self, x):
    #           r = np.linalg.norm(x)
    #           if r == 0:
    #               return np.zeros_like(x)
    #           return 2 * x * (self.radius / r - 1) / self.sigma2

    def __init__(self,  target, initial,  iterations=100000):
        self.target = target
        self.initial = initial
        self.iterations = iterations

    def leap_frog(self, p0, q0, step_size):
        p = p0.copy()
        q = q0.copy()

        for i in range(self.L):
            p += self.target.log_density(q)*step_size/2
            q += p*step_size
            p += self.target.log_density(q)*step_size/2

        return p, q
    
    def metropolis_hastings(self, proposal):
        samples = []

        for _ in range(self.iterations):
            current = samples[-1]
            proposed = proposal(current)
            if np.random.random() < self.target(proposed) / self.target(current):
                samples.append(proposed)
            else:
                samples.append(current)

        return samples
    
    def hmc(self, L, step_size):
        samples = [self.initial()]

        for i in range(self.iterations):
            q0 = samples[-1]
            p0 = np.random.standard_normal(size=q0.size)

            q_star, p_star = self.leap_frog(q0, p0, self.target, L, step_size)
            #Proceed hamilton dynamics by using leapfrog
            #which accurately preserves the volume of the states in phase space
            h0 = -self.target.log_density(q0) + (p0 * p0).sum() / 2
            h = -self.target.log_density(q_star) + (p_star * p_star).sum() / 2
            log_accept_ratio = h0 - h

            if np.random.random() < np.exp(log_accept_ratio):
                samples.append(q_star)
            else:
                samples.append(q0)
        
        return samples