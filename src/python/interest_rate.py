import numpy as np
from scipy.interpolate import interp1d
from src.python.cashflow import CashFlow

class InterestRate:
    def __init__(self, rate, compounding_frequency):
        # save interest rate as continuous
        self.rate = InterestRate.change_interest_frequency(r1=rate, m1=compounding_frequency, m2="continuous")

    def __call__(self, compounding_frequency='continuous'):
        return InterestRate.change_interest_frequency(r1=self.rate, m1="continuous", m2=compounding_frequency)
    
    def discount(self, time, value):
        return value*np.exp(-self.rate*time)
    
    def discount_cashflow(self, cashflow: CashFlow):
        return np.sum([amount*np.exp(-self.rate*time) for (time, amount) in cashflow])

    @staticmethod
    def change_interest_frequency(*, r1:float, m1:int|str, m2:int|str)->float:
        """
        Changes the interest rate from a period of m1 to a period of m2.

        Parameters
        ----------
        r1 : float
            Interest rate for period m1.
        m1 : int or str 
            Period of the interest rate r1 [1/(times per year)].
        m2 : int or str
            Period of the interest rate to be calculated [1/(times per year)].

        Returns
        -------
        float
            The equiavalent interest rate for period m2.
        """
        assert isinstance(m1, int) or m1 == "continuous", "m2 must be a int or 'continuous'"
        assert isinstance(m2, int) or m2 == "continuous", "m2 must be a int or 'continuous'"

        if m1 == m2:
            return r1

        if m1 == "continuous":
            return m2*(np.exp(r1/m2) - 1.0)

        if m2 == "continuous":
            return m1*np.log(1.0 + r1/m1)

        return m2*((1.0 + r1/m1)**(m1/m2) - 1.0)
    
class InterestRateCurve:
    def __init__(self, times, interest_rates):
        self.times = times
        self.interest_rates = interest_rates
        self.continuous_rates = [r.rate for r in self.interest_rates]

        self.curve = interp1d(x=self.times, y=self.continuous_rates, kind="linear", bounds_error=False, fill_value=(self.continuous_rates[0], self.continuous_rates[-1]))
    
    def discount_cashflow(self, cashflow: CashFlow):
        return np.sum([amount*np.exp(-self.curve(time)*time) for (time, amount) in cashflow])
    
class ZeroRate(InterestRate): 
    def __init__(self, time, rate, compounding_frequency):
        super().__init__(rate, compounding_frequency)
        self.time = time

class ZeroRateCurve(InterestRateCurve):
    def __init__(self, zero_rates):
        for rate in zero_rates: assert isinstance(rate, ZeroRate), f"All zero rates must be ZeroRate instances and got {type(rate)} val={rate}"
        times = [rate.time for rate in zero_rates]
        super().__init__(times, zero_rates)

class ForwardRate(InterestRate):
    def __init__(self, rate, compounding_frequency, t1, t2):
        super().__init__(rate, compounding_frequency)

        self.t1 = t1
        self.t2 = t2

    @staticmethod
    def calculate_forward_rate_from_zero_rates(zero_rates, t1, t2, compounding_frequency='continuous'):
        if not isinstance(zero_rates, ZeroRateCurve):
            zero_rates = ZeroRateCurve(zero_rates)

        forward_rate = (zero_rates.curve(t2)*t2 - zero_rates.curve(t1)*t1) / (t2- t1)
        return ForwardRate(rate=forward_rate, compounding_frequency=compounding_frequency, t1=t1, t2=t2)
