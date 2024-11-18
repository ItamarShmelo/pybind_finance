import sys
sys.path.append('..')
import numpy as np
from src.python.analytic_solutions.vanilla_call import BlackScholesVanillaCall
from src.python.interest_rate import InterestRate, ForwardRate, ZeroRate
from src.python.analytic_solutions.simple_formulas import ForwardRateAgreement

def test_vanilla_call_option_price():
    vanilla_call_page_392 = BlackScholesVanillaCall(K=900, r=InterestRate(0.08, "continuous"), sigma=0.2, q=InterestRate(0.03, 'continuous'))

    S=930.
    T=2./12.
    assert np.isclose(vanilla_call_page_392.d_plus(S, T, vanilla_call_page_392.K), 0.5444, atol=1e-4)
    assert np.isclose(vanilla_call_page_392.d_minus(S, T, vanilla_call_page_392.K), 0.4628, atol=1e-4)
    assert np.isclose(vanilla_call_page_392.option_price(S, T, vanilla_call_page_392.K), 51.83, atol=1e-2)

    vanilla_call_page_395 = BlackScholesVanillaCall(K=1.6, r=InterestRate(0.08, "continuous"), sigma=0.2, q=InterestRate(0.11, 'continuous'))

    S=1.6
    T=0.3333
    assert np.isclose(vanilla_call_page_395.option_price(S, T), 0.0639, atol=1e-4)

    vanilla_call_page_395.sigma = 0.1
    assert np.isclose(vanilla_call_page_395.option_price(S, T), 0.0285, atol=1e-4)

    vanilla_call_page_395.sigma = 0.141
    assert np.isclose(vanilla_call_page_395.option_price(S, T), 0.043, atol=1e-3)


    # comparison with https://www.math.drexel.edu/~pg/fin/VanillaCalculator.html
    vanilla_call_test_1 = BlackScholesVanillaCall(K=100., r=InterestRate(0.05, "continuous"), sigma=0.2, q=InterestRate(0.0, 'continuous'))
    
    S=100.
    T=1.
    assert np.isclose(vanilla_call_test_1.option_price(S, T), 10.45058, atol=1e-5)
    assert np.isclose(vanilla_call_test_1.delta(S, T, vanilla_call_test_1.K), 0.63683, atol=1e-5)
    assert np.isclose(vanilla_call_test_1.strike_given_delta(S, T, 0.63683, tol=1e-5), 100., atol=1e-5)

    vanilla_call_test_1.q = InterestRate(0.2, 'continuous')
    assert np.isclose(vanilla_call_test_1.option_price(S, T), 2.30841, atol=1e-5)
    assert np.isclose(vanilla_call_test_1.delta(S, T, vanilla_call_test_1.K), 0.21111, atol=1e-5)
    assert np.isclose(vanilla_call_test_1.strike_given_delta(S, T, 0.21111, tol=1e-5), 100., atol=1e-5)

    S=120.
    T=0.5
    assert np.isclose(vanilla_call_test_1.strike_given_delta(S, T, 0.7208065, tol=1e-5), 100., atol=1e-5)

def test_forward_rate_agreement():
    fra_p_112 = ForwardRateAgreement(principal=100e6, tstart=1.5, tend=2.0, risk_free_rate=InterestRate(0.04, 'continuous'), fixed_rate=InterestRate(0.058, 2))

    fra_price = fra_p_112.price(forward_rate=ForwardRate(0.05, compounding_frequency=2, t1=1.5, t2=2.0), compounding_frequency=2)

    assert np.isclose(fra_price, 369200, atol=1000) 
    

    fra_pq_4_5 = ForwardRateAgreement(principal=1e6, fixed_rate=InterestRate(0.045, 4), tstart=1.0, tend=1.25, risk_free_rate=InterestRate(0.036, 'continuous')) 

    zero_rates_pq_4_4 = np.array([
        [0.25, ZeroRate(0.03, 'continuous')],
        [0.50, ZeroRate(0.032, 'continuous')],
        [0.75, ZeroRate(0.034, 'continuous')],
        [1.00, ZeroRate(0.035, 'continuous')],
        [1.25, ZeroRate(0.036, 'continuous')],
        [1.50, ZeroRate(0.037, 'continuous')],
    ])

    solution_pq_4 = [0.034, 0.038, 0.038, 0.04, 0.042]

    t1 = zero_rates_pq_4_4[3][0]
    t2 = zero_rates_pq_4_4[4][0]
    forward_rate = ForwardRate.calculate_forward_rate_from_zero_rates(zero_rates_pq_4_4, t1, t2, compounding_frequency=4)

    assert np.allclose(fra_pq_4_5.price(forward_rate, compounding_frequency=4), 1195.0, atol=1)