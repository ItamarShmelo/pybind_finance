from src.python.interest_rate import ForwardRate

class ForwardRateAgreement:
    def __init__(self, fixed_rate, principal, risk_free_rate, tstart, tend):
        self.fixed_rate = fixed_rate
        self.principal = principal
        self.risk_free_rate = risk_free_rate
        self.tstart = tstart
        self.tend = tend
    
    def price(self, forward_rate, compounding_frequency='continuous'):
        dtime = self.tend - self.tstart

        return self.risk_free_rate.discount(values=[dtime*self.principal*(self.fixed_rate(compounding_frequency)-forward_rate(compounding_frequency))], times=[self.tend])