import sys
sys.path.append('..')
import numpy as np
from src.python.analytic_solutions.vanilla_call import BlackScholesVanillaCall
from src.python.interest_rate import InterestRate, ForwardRate, ZeroRate
from src.python.analytic_solutions.simple_formulas import ForwardRateAgreement

from tests.get_test_logger import get_test_logger

logger = get_test_logger('analytic solutions')

def test_vanilla_call_option_price():
    # page 392
    option_data_p392 = dict(S=930., K=900., T=2./12., sigma=0.2, r=InterestRate(0.08, "continuous"), q=InterestRate(0.03, 'continuous'))

    d_plus = BlackScholesVanillaCall.d_plus(**option_data_p392)
    d_minus = BlackScholesVanillaCall.d_minus(**option_data_p392)
    option_price = BlackScholesVanillaCall.option_price(**option_data_p392)

    assert np.isclose(d_plus, 0.5444, atol=1e-4)
    assert np.isclose(d_minus, 0.4628, atol=1e-4)
    assert np.isclose(option_price, 51.83, atol=1e-2)


    option_data_p395 = dict(S=1.6, K=1.6, T=0.3333, sigma=0.2, r=InterestRate(0.08, "continuous"), q=InterestRate(0.11, 'continuous'))
    option_price = BlackScholesVanillaCall.option_price(**option_data_p395)
    assert np.isclose(option_price, 0.0639, atol=1e-4)

    option_data_p395['sigma'] = 0.1
    option_price = BlackScholesVanillaCall.option_price(**option_data_p395)
    assert np.isclose(option_price, 0.0285, atol=1e-4)

    option_data_p395['sigma'] = 0.141
    option_price = BlackScholesVanillaCall.option_price(**option_data_p395)
    assert np.isclose(option_price, 0.043, atol=1e-3)


    # # comparison with https://www.math.drexel.edu/~pg/fin/VanillaCalculator.html
    option_data_web = dict(S=100., K=100., T=1., sigma=0.2, r=InterestRate(0.05, "continuous"), q=InterestRate(0.0, 'continuous'))
    option_price = BlackScholesVanillaCall.option_price(**option_data_web)
    delta = BlackScholesVanillaCall.delta(**option_data_web)
    strike_given_delta = BlackScholesVanillaCall.strike_given_delta(delta=0.63683, tol=1e-5, **option_data_web)
    assert np.isclose(option_price, 10.45058, atol=1e-5)
    assert np.isclose(delta, 0.63683, atol=1e-5)
    assert np.isclose(strike_given_delta, 100., atol=1e-5)

    option_data_web['q'] = InterestRate(0.2, 'continuous')
    option_price = BlackScholesVanillaCall.option_price(**option_data_web)
    delta = BlackScholesVanillaCall.delta(**option_data_web)
    strike_given_delta = BlackScholesVanillaCall.strike_given_delta(delta=0.21111, tol=1e-5, **option_data_web)
    assert np.isclose(option_price, 2.30841, atol=1e-5)
    assert np.isclose(delta, 0.21111, atol=1e-5)
    assert np.isclose(strike_given_delta, 100., atol=1e-5)

    option_data_web['S'] = 120.
    option_data_web['T'] = 0.5
    strike_given_delta = BlackScholesVanillaCall.strike_given_delta(delta=0.7208065, tol=1e-5, **option_data_web)

    assert np.isclose(strike_given_delta, 100., atol=1e-5)

def test_forward_rate_agreement():
    fra_p_112 = ForwardRateAgreement(principal=100e6, tstart=1.5, tend=2.0, risk_free_rate=InterestRate(0.04, 'continuous'), fixed_rate=InterestRate(0.058, 2))

    fra_price = fra_p_112.price(forward_rate=ForwardRate(0.05, compounding_frequency=2, t1=1.5, t2=2.0), compounding_frequency=2)

    assert np.isclose(fra_price, 369200, atol=1000) 
    

    fra_pq_4_5 = ForwardRateAgreement(principal=1e6, fixed_rate=InterestRate(0.045, 4), tstart=1.0, tend=1.25, risk_free_rate=InterestRate(0.036, 'continuous')) 

    zero_rates_pq_4_4 = np.array([
        ZeroRate(0.25, 0.03, 'continuous') ,
        ZeroRate(0.50, 0.032, 'continuous'),
        ZeroRate(0.75, 0.034, 'continuous'),
        ZeroRate(1.00, 0.035, 'continuous'),
        ZeroRate(1.25, 0.036, 'continuous'),
        ZeroRate(1.50, 0.037, 'continuous'),
    ])

    solution_pq_4 = [0.034, 0.038, 0.038, 0.04, 0.042]

    t1 = zero_rates_pq_4_4[3].time
    t2 = zero_rates_pq_4_4[4].time
    forward_rate = ForwardRate.calculate_forward_rate_from_zero_rates(zero_rates_pq_4_4, t1, t2, compounding_frequency=4)

    assert np.allclose(fra_pq_4_5.price(forward_rate, compounding_frequency=4), 1195.0, atol=1)