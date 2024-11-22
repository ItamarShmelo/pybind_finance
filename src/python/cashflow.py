import numpy as np

class CashFlow:
    def __init__(self, times, amounts):
        self.times = times
        self.amounts = amounts

    def __iter__(self):
        for time, amount in zip(self.times, self.amounts):
            yield time, amount