# import sys
# sys.path.append('../../..')
import numpy as np
from typing import Callable
from scipy.optimize import root_scalar
from scipy.interpolate import interp1d

from src.python.interest_rate import InterestRate, ZeroRate, ZeroRateCurve
# from interest_rate import InterestRate, ZeroRate, ZeroRateCurve

class Bond:
    def __init__(self,*, principal, interest_rate, coupon_frequency, time_to_maturity):
        self.principal = principal
        self.interest_rate = interest_rate
        self.coupon_frequency = coupon_frequency
        self.time_to_maturity = time_to_maturity
        
        if self.coupon_frequency==0: 
            self.coupon = 0.0
        else:
            self.coupon = self.interest_rate(1)*self.principal/self.coupon_frequency

    def get_bond_price_from_zero_rates(self, *, zero_rates)->float:
        if not isinstance(zero_rates, ZeroRateCurve):
            zero_rates = ZeroRateCurve(zero_rates[:, 0], zero_rates[:, 1])

        T = self.time_to_maturity
        m = self.coupon_frequency
        dt = 1.0/m if m != 0 else 0.0
        c = self.coupon
        
        price = zero_rates.discount(values=[c]*int(T//dt) + [self.principal], times=[n*dt for n in range(1, int(m*T)+1)] + [T])

        return price

    def get_bond_price_from_yield(self, *, bond_yield)->float:
        T = self.time_to_maturity
        m = self.coupon_frequency
        dt = 1.0/m if m != 0 else 0.0
        print(int(m*T)+1)
        
        return bond_yield.discount(values=[self.coupon]*int(T//dt) + [self.principal], times=[n*dt for n in range(1, int(m*T)+1)] + [T])    
        
    def calculate_zero_rate_at_time_of_maturity_from_bond_price(self, *, bond_price, zero_rates=None)->float:
        if zero_rates is None:
            # return yield
            # bond_yield = self.
            # return 
            pass

        if not isinstance(zero_rates, ZeroRateCurve):
            zero_rates = ZeroRateCurve(zero_rates[:, 0], zero_rates[:, 1])
        
        T = self.time_to_maturity
        m = self.coupon_frequency
        dt = 1.0/m if m != 0 else 0.0
        c = self.coupon
        last_time_zero_rate_curve = zero_rates.times[-1]
        assert last_time_zero_rate_curve < self.time_to_maturity

        zero_rates_discounted_coupons = zero_rates.discount(values=[c]*(int(last_time_zero_rate_curve//dt)), times=[n*dt for n in range(1, int(m*last_time_zero_rate_curve)+1)])
        f = lambda R: zero_rates_discounted_coupons + np.sum([c*np.exp(-R*i*dt) for i in range(int(m*last_time_zero_rate_curve)+1, int(m*T)+1)]) + self.principal*np.exp(-R*T) - bond_price

        sol = root_scalar(f=f, x0=self.interest_rate.rate, xtol=1e-6).root

        return ZeroRate(sol, 'continuous')

    def calculate_bond_yield(self, *, bond_price):
        T = self.time_to_maturity
        m = self.coupon_frequency
        dt = 1.0/m if m != 0 else 0.0
        c = self.coupon
        
        f = lambda y: np.sum([c*np.exp(-y*i*dt) for i in range(1, int(m*T)+1)]) + self.principal*np.exp(-y*T) - bond_price

        sol = root_scalar(f=f, x0=self.interest_rate.rate)
        return InterestRate(sol.root, 'continuous')

    def calculate_bond_par_yield(self, *, zero_rates=None):
        if not isinstance(zero_rates, ZeroRateCurve):
            zero_rates = ZeroRateCurve(zero_rates[:, 0], zero_rates[:, 1])
            
        m = self.coupon_frequency
        T = self.time_to_maturity
        dt = 1.0/m if m!= 0 else 0.0

        A = zero_rates.discount(values=[1.0]*int(self.coupon_frequency*self.time_to_maturity), 
        times=[n*dt for n in range(1, int(m*T)+1)])
        
        return m*(self.principal - zero_rates.discount(values=[self.principal], times=[T]))/A
