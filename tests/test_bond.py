import sys
sys.path.append('..')
import numpy as np
from src.python.interest_rate import InterestRate, ZeroRate, ZeroRateCurve
from src.python.bond import Bond
from tests.get_test_logger import get_test_logger

logger = get_test_logger(name='test_bond')

def test_bond_price():
    zero_rates_table_42 = np.array([ZeroRate(0.5, 0.050, 'continuous'), 
                                    ZeroRate(1.0, 0.058, 'continuous'), 
                                    ZeroRate(1.5, 0.064, 'continuous'), 
                                    ZeroRate(2.0, 0.068, 'continuous')])
    
    bond = Bond(principal=100.0, interest_rate=InterestRate(0.06, 1), coupon_frequency=2, time_to_maturity=2.0)
    assert np.isclose(bond.get_bond_price_from_zero_rates(zero_rates=zero_rates_table_42), 98.39, atol=1e-2)

    zero_rates_pq_4_10 = np.array([
        ZeroRate(0.5, 0.040, 'continuous'), 
        ZeroRate(1.0, 0.042, 'continuous'), 
        ZeroRate(1.5, 0.044, 'continuous'), 
        ZeroRate(2.0, 0.046, 'continuous'),
        ZeroRate(2.5, 0.048, 'continuous')
    ])

    bond_pq_4_10 = Bond(principal=100.0, interest_rate=InterestRate(0.04, 1), coupon_frequency=2, time_to_maturity=2.5)
    assert np.isclose(bond_pq_4_10.get_bond_price_from_zero_rates(zero_rates=zero_rates_pq_4_10), 98.04, atol=1e-2)


def test_bond_get_bond_price_from_yield():
    zero_rates_pq_42 = np.array([ZeroRate(0.5, 0.05, 1), ZeroRate(1.0, 0.05, 1)])
    bond_pq_42 = Bond(principal=100.0, interest_rate=InterestRate(0.04, 1), coupon_frequency=2, time_to_maturity=1.5)

    bond_pq_42_price = bond_pq_42.get_bond_price_from_yield(bond_yield=InterestRate(0.052, 2))
    assert np.isclose(bond_pq_42_price, 98.29, atol=1e-3)


def test_calculate_zero_rate_at_time_of_maturity_from_bond_price():
    zero_rates_pq_42 = np.array([ZeroRate(0.5, 0.05, 1), ZeroRate(1.0, 0.05, 1)])
    bond_pq_42 = Bond(principal=100.0, interest_rate=InterestRate(0.04, 1), coupon_frequency=2, time_to_maturity=1.5)

    bond_pq_42_price = bond_pq_42.get_bond_price_from_yield(bond_yield=InterestRate(0.052, 2))
    zero_rate_18month = bond_pq_42.calculate_zero_rate_at_time_of_maturity_from_bond_price(bond_price=bond_pq_42_price, zero_rates=zero_rates_pq_42)

    assert np.isclose(zero_rate_18month(2), 0.052, atol=1e-4)

def test_bond_calculate_yield():
    # pq 4.11
    bond = Bond(principal=100.0, interest_rate=InterestRate(0.08, 1), coupon_frequency=2, time_to_maturity=3.0)
    
    np.isclose(bond.calculate_bond_yield(bond_price=104.0).rate, 0.06407, atol=1e-5)

def test_bond_calculate_par_yield():
    zero_rates_table_42 = np.array([ZeroRate(0.5, 0.050, 'continuous'), 
                                    ZeroRate(1.0, 0.058, 'continuous'), 
                                    ZeroRate(1.5, 0.064, 'continuous'), 
                                    ZeroRate(2.0, 0.068, 'continuous')])
    
    bond = Bond(principal=100.0, interest_rate=InterestRate(0.03, 1), coupon_frequency=2, time_to_maturity=2.0)

    assert np.isclose(bond.calculate_bond_par_yield(zero_rates=zero_rates_table_42), 6.87, atol=1e-2)

    zero_rates_pq_4_12 = np.array([
        ZeroRate(0.5, 0.050, 'continuous'), 
        ZeroRate(1.0, 0.060, 'continuous'), 
        ZeroRate(1.5, 0.065, 'continuous'), 
        ZeroRate(2.0, 0.070, 'continuous') 
    ])
    
    assert np.isclose(bond.calculate_bond_par_yield(zero_rates=zero_rates_pq_4_12), 7.0741, atol=1e-4)
    