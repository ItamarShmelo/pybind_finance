# import sys
# sys.path.append('../../..')
import numpy as np
from typing import Callable
from scipy.optimize import root_scalar
from scipy.interpolate import interp1d

from src.python.interest_rate import InterestRate, ZeroRate, ZeroRateCurve
# from interest_rate import InterestRate, ZeroRate, ZeroRateCurve

from src.python.cashflow import CashFlow

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

        dt = 1.0/self.coupon_frequency if self.coupon_frequency!=0 else 0.0
        # bond cashflow
        self.cashflow = CashFlow(
                                times = [n*dt for n in range(1, int(self.time_to_maturity*self.coupon_frequency)+1)] + [self.time_to_maturity],
                                amounts = [self.coupon]*int(self.time_to_maturity*self.coupon_frequency) + [self.principal]
                                )

    def get_bond_price_from_zero_rates(self, *, zero_rates)->float:
        if not isinstance(zero_rates, ZeroRateCurve):
            zero_rates = ZeroRateCurve(zero_rates)

        return zero_rates.discount_cashflow(self.cashflow)

    def get_bond_price_from_yield(self, *, bond_yield)->float:
        return bond_yield.discount_cashflow(self.cashflow)    
        
    def calculate_zero_rate_at_time_of_maturity_from_bond_price(self, *, bond_price, zero_rates=None)->float:
        if zero_rates is None:
            bond_yield = self.calculate_bond_yield(bond_price=bond_price)
            return ZeroRate(self.time_to_maturity, bond_yield.rate, 'continuous')

        if not isinstance(zero_rates, ZeroRateCurve):
            zero_rates = ZeroRateCurve(zero_rates)
        
        T = self.time_to_maturity
        m = self.coupon_frequency
        dt = 1.0/m if m != 0 else 0.0
        c = self.coupon
        
        last_time_zero_rate_curve = zero_rates.times[-1]
        assert last_time_zero_rate_curve < self.time_to_maturity

        cashflow_covered_by_zero_curve = CashFlow(
            times=[n*dt for n in range(1, int(m*last_time_zero_rate_curve)+1)],
            amounts=[c]*(int(m*last_time_zero_rate_curve))
        )

        rest_of_cashflow = CashFlow(
            times=[n*dt for n in range(int(m*last_time_zero_rate_curve)+1, int(m*T)+1)]+[T],
            amounts = [c for n in range(int(m*last_time_zero_rate_curve)+1, int(m*T)+1)]+[self.principal]
        )

        zero_rates_discounted_coupons = zero_rates.discount_cashflow(cashflow_covered_by_zero_curve)
        f = lambda R: zero_rates_discounted_coupons + InterestRate(R, 'continuous').discount_cashflow(rest_of_cashflow) - bond_price

        sol = root_scalar(f=f, x0=self.interest_rate.rate, xtol=1e-6).root

        return ZeroRate(self.time_to_maturity, sol, 'continuous')

    def calculate_bond_yield(self, *, bond_price):
        f = lambda y: InterestRate(y, 'continuous').discount_cashflow(self.cashflow) - bond_price

        sol = root_scalar(f=f, x0=self.interest_rate.rate)
        return InterestRate(sol.root, 'continuous')

    def calculate_bond_par_yield(self, *, zero_rates=None):
        if not isinstance(zero_rates, ZeroRateCurve):
            zero_rates = ZeroRateCurve(zero_rates)
            
        annuity_cashflow = CashFlow(times=self.cashflow.times[:-1], amounts=[1.0]*(len(self.cashflow.times)-1))
        A = zero_rates.discount_cashflow(annuity_cashflow)
        
        principal_cashflow = CashFlow(times=[self.time_to_maturity], amounts=[self.principal])
        
        return self.coupon_frequency*(self.principal - zero_rates.discount_cashflow(principal_cashflow))/A
