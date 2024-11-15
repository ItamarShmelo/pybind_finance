import sys
sys.path.append('..')
import numpy as np
from src.python.interest_rate import InterestRate

def test_interest_rates():
    rate_page_103 = InterestRate(0.06, 2)
    assert np.isclose(rate_page_103(4), 0.0596, atol=1e-4)

    # page 104
    rate_page_104 = InterestRate(0.1, 2)
    assert np.isclose(rate_page_104('continuous'), 0.09758, atol=1e-5)

    rate_page_104_two = InterestRate(0.08, "continuous")
    assert np.isclose(rate_page_104_two(4), 0.0808, atol=1e-4)