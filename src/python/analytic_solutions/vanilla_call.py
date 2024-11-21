import numpy as np
from scipy.stats import norm
from scipy.optimize import root_scalar
import sys
sys.path.append('../../..')
from src.python.interest_rate import InterestRate

def assert_values_for_black_scholes(*, S, K, T, sigma, r, q):
    assert isinstance(r, InterestRate), 'r must be an instance of InterestRate'
    assert isinstance(q, InterestRate), 'q must be an instance of InterestRate'
    assert isinstance(S, float) and S > 0.0, 'S must be greater than zero'
    assert isinstance(T, float) and T > 0.0, 'T must be greater than zero'
    assert isinstance(sigma, float) and sigma > 0.0, f'sigma must be a float greater than 0 but got {sigma:g}"'
    assert K > 0.0, f"K must be a float greater than zero but got {K}"

class BlackScholesVanillaCall:
    @staticmethod
    def d_plus(*, S, K, T, sigma, r, q):
        assert_values_for_black_scholes(S=S, K=K, T=T, sigma=sigma, r=r, q=q)
        return (np.log(S/K) + (r.rate - q.rate + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))

    @staticmethod
    def d_minus(*, S, K, T, sigma, r, q):
        return BlackScholesVanillaCall.d_plus(S=S, K=K, T=T, sigma=sigma, r=r, q=q) - sigma*np.sqrt(T)

    @staticmethod
    def option_price(*, S, K, T, sigma, r, q):
        d_plus = BlackScholesVanillaCall.d_plus(S=S, K=K, T=T, sigma=sigma, r=r, q=q)
        d_minus = BlackScholesVanillaCall.d_minus(S=S, K=K, T=T, sigma=sigma, r=r, q=q)

        return norm.cdf(d_plus) * S*np.exp(-q.rate*T)  - norm.cdf(d_minus) * K * np.exp(-r.rate*T)
    
    @staticmethod
    def delta(*, S, K, T, sigma, r, q):
        d_plus = BlackScholesVanillaCall.d_plus(S=S, K=K, T=T, sigma=sigma, r=r, q=q)
        return norm.cdf(d_plus) * np.exp(-q.rate*T)
    
    @staticmethod
    def strike_given_delta(*, delta, S, T, sigma, r, q, tol=1e-3, **kwargs):
        f = lambda K: BlackScholesVanillaCall.delta(S=S, K=K, T=T, sigma=sigma, r=r, q=q) - delta
        
        return root_scalar(f, x0=S, xtol=tol).root
    
if __name__ == "__main__":
    pass

        
