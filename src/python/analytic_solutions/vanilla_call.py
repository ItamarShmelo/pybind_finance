import numpy as np
from scipy.stats import norm
from scipy.optimize import root_scalar
import sys
sys.path.append('../../..')
from src.python.interest_rate import InterestRate

class BlackScholesVanillaCall:
    def __init__(self, K, r, sigma, q=InterestRate(0.0, 1)):
        assert isinstance(r, InterestRate)
        assert isinstance(q, InterestRate)

        self.K = K
        self.r = r
        self.sigma = sigma
        self.q = q

    def d_plus(self, S, T, K):
        return (np.log(S/K) + (self.r.rate - self.q.rate + 0.5*self.sigma**2)*T) / (self.sigma*np.sqrt(T))

    def d_minus(self, S, T, K):
        return self.d_plus(S, T, K) - self.sigma*np.sqrt(T)

    def unpack(self):
        return self.K, self.r, self.sigma, self.q
    
    def option_price(self, S, T, K=None):
        K_, r, _, q = self.unpack()
        
        if K is None: K = K_
        
        d_plus = self.d_plus(S, T, K)
        d_minus = self.d_minus(S, T, K)

        return norm.cdf(d_plus) * S*np.exp(-q.rate*T)  - norm.cdf(d_minus) * K * np.exp(-r.rate*T)
    
    def delta(self, S, T, K=None):
        if K is None: K = self.K

        d_plus = self.d_plus(S, T, K=K)
        return norm.cdf(d_plus) * np.exp(-self.q.rate*T)
    
    def strike_given_delta(self, S, T, delta, tol=1e-3):
        f = lambda K: self.delta(S, T, K) - delta
        
        return root_scalar(f, x0=S, xtol=tol).root

