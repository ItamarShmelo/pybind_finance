import sys
sys.path.append('..')
import numpy as np
from src.python.analytic_solutions.vanilla_call import BlackScholesVanillaCall
from src.python.interest_rate import InterestRate

def test_vanilla_call_option_price():
    vanilla_call_page_392 = BlackScholesVanillaCall(K=900, r=InterestRate(0.08, "continuous"), sigma=0.2, q=InterestRate(0.03, 'continuous'))

    S=930.
    T=2./12.
    assert np.isclose(vanilla_call_page_392.d_plus(S, T), 0.5444, atol=1e-4)
    assert np.isclose(vanilla_call_page_392.d_minus(S, T), 0.4628, atol=1e-4)
    assert np.isclose(vanilla_call_page_392.option_price(S, T), 51.83, atol=1e-2)

    vanilla_call_page_395 = BlackScholesVanillaCall(K=1.6, r=InterestRate(0.08, "continuous"), sigma=0.2, q=InterestRate(0.11, 'continuous'))

    S=1.6
    T=0.3333
    assert np.isclose(vanilla_call_page_395.option_price(S, T), 0.0639, atol=1e-4)

    vanilla_call_page_395.sigma = 0.1
    assert np.isclose(vanilla_call_page_395.option_price(S, T), 0.0285, atol=1e-4)

    vanilla_call_page_395.sigma = 0.141
    assert np.isclose(vanilla_call_page_395.option_price(S, T), 0.043, atol=1e-3)
