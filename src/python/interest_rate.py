import numpy as np
from scipy.interpolate import interp1d

class InterestRate:
    def __init__(self, rate, compounding_frequency):
        # save interest rate as continuous
        self.rate = InterestRate.change_interest_frequency(r1=rate, m1=compounding_frequency, m2="continuous")

    def __call__(self, compounding_frequency='continuous'):
        return InterestRate.change_interest_frequency(r1=self.rate, m1="continuous", m2=compounding_frequency)
    
    def discount(self, values, times):
        return np.sum([value*np.exp(-self.rate*time) for (value, time) in zip(values, times)])

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
        self.continuous_rates = [rate for rate in self.interest_rates]

        self.curve = interp1d(x=self.times, y=self.continuous_rates, kind="linear", bounds_error=False, fill_value=(self.continuous_rates[0], self.continuous_rates[-1]))
    
    def discount(self, values, times):
        return np.sum(values*np.exp(-self.curve(times)))
    
class ZeroRate(InterestRate):
    pass

class ZeroRateCurve(InterestRateCurve):
    pass