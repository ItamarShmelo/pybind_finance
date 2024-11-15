import numpy as np
from scipy.stats import norm
import sys
sys.path.append('../..')
from src.python.interest_rate import InterestRate

class BlackScholesVanillaCall:
    def __init__(self, K, r, sigma, q=InterestRate(0.0, 1)):
        assert isinstance(r, InterestRate)
        assert isinstance(q, InterestRate)

        self.K = K
        self.r = r
        self.sigma = sigma
        self.q = q

    def d_plus(self, S, T):
        return (np.log(S/self.K) + (self.r.rate - self.q.rate + 0.5*self.sigma**2)*T) / (self.sigma*np.sqrt(T))

    def d_minus(self, S, T):
        return self.d_plus(S, T) - self.sigma*np.sqrt(T)

    def unpack(self):
        return self.K, self.r, self.sigma, self.q
    
    def option_price(self, S, T):
        d_plus = self.d_plus(S, T)
        d_minus = self.d_minus(S, T)
        K, r, _, q = self.unpack()

        return norm.cdf(d_plus) * S*np.exp(-q.rate*T)  - norm.cdf(d_minus) * K * np.exp(-r.rate*T)
    
    def delta(self, S, T):
        d_plus = self.d_plus(S, T)
        return norm.cdf(d_plus) * np.exp(-self.q.rate*T)
    
