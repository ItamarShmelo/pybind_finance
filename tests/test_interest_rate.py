import sys
sys.path.append('..')
import numpy as np
from src.python.interest_rate import InterestRate, ZeroRate, ForwardRate

def test_interest_rates_change_interest_frequency():
    rate_page_103 = InterestRate(0.06, 2)
    assert np.isclose(rate_page_103(4), 0.0596, atol=1e-4)

    # page 104
    rate_page_104 = InterestRate(0.1, 2)
    assert np.isclose(rate_page_104('continuous'), 0.09758, atol=1e-5)

    rate_page_104_two = InterestRate(0.08, "continuous")
    assert np.isclose(rate_page_104_two(4), 0.0808, atol=1e-4)

    # pq stands for practice question
    rate_pq_1 = InterestRate(0.07, 4)
    assert np.isclose(rate_pq_1('continuous'), 0.0694, atol=1e-4)
    assert np.isclose(rate_pq_1(1), 0.0719, atol=1e-4)

    #
    interest_rate_pq_3 = InterestRate((1100.0-1000.0)/1000.0, 1)

    assert np.isclose(interest_rate_pq_3(1), 0.1, atol=1e-4) # a
    assert np.isclose(interest_rate_pq_3(2), 0.0976, atol=1e-4) # b
    assert np.isclose(interest_rate_pq_3(12), 0.0957, atol=1e-4) # c
    assert np.isclose(interest_rate_pq_3(), 0.0953, atol=1e-4)  # d

    interest_rate_pq_4_8 = InterestRate(0.08, 12)
    assert np.isclose(interest_rate_pq_4_8(), 0.0797, atol=1e-4)  

    interest_rate_pq_4_9 = InterestRate(0.04, 'continuous')
    assert np.isclose(interest_rate_pq_4_9(4), 0.0402, atol=1e-4)
    assert np.isclose(1e4*interest_rate_pq_4_9(4)/4.0, 100.50, atol=1e-2)



def test_calculate_forward_rate_from_zero_rates():
    # pq 4
    zero_rates_pq_4_4 = np.array([
        ZeroRate(0.25, 0.030, 'continuous'),
        ZeroRate(0.50, 0.032, 'continuous'),
        ZeroRate(0.75, 0.034, 'continuous'),
        ZeroRate(1.00, 0.035, 'continuous'),
        ZeroRate(1.25, 0.036, 'continuous'),
        ZeroRate(1.50, 0.037, 'continuous')
    ])

    solution_pq_4 = [0.034, 0.038, 0.038, 0.04, 0.042]

    for i in range(1, len(zero_rates_pq_4_4)):
        t1 = zero_rates_pq_4_4[i-1].time
        t2 = zero_rates_pq_4_4[i].time
        forward_rate = ForwardRate.calculate_forward_rate_from_zero_rates(zero_rates_pq_4_4, t1, t2)
        assert np.isclose(forward_rate(), solution_pq_4[i-1], atol=1e-4)
    
    # pq 4.13
    zero_rates_pq_4_13 = np.array([
        ZeroRate(1.0, 0.020, 'continuous'),
        ZeroRate(2.0, 0.030, 'continuous'),
        ZeroRate(3.0, 0.037, 'continuous'),
        ZeroRate(4.0, 0.042, 'continuous'),
        ZeroRate(5.0, 0.045, 'continuous')
    ])

    results_pq_4_13 = [0.04, 0.051, 0.057, 0.057]

    for i in range(1, len(zero_rates_pq_4_13)):
        t1 = zero_rates_pq_4_13[i-1].time
        t2 = zero_rates_pq_4_13[i].time
        forward_rate = ForwardRate.calculate_forward_rate_from_zero_rates(zero_rates_pq_4_13, t1, t2)
        logger.info(f'Forward rate from t={t1} to t={t2}: {forward_rate():g}')
        np.isclose(forward_rate(), results_pq_4_13[i-1], atol=1e-4)