from my_package_project import my_package_project
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pybacktestchain.data_module import UNIVERSE_SEC, FirstTwoMoments, get_stocks_data, DataModule, Information
from my_package_project.data_treatment import compute_risk_contributions, portfolio_volatility, RiskParity


# Mock data for testing
MOCK_INFORMATION_SET = {
    "covariance_matrix": np.array([
        [0.04, 0.02, 0.01],
        [0.02, 0.03, 0.015],
        [0.01, 0.015, 0.05]
    ]),
    "companies": ["SPY", "TLT", "GLD"]
}

MOCK_PORTFOLIO = {
    "SPY": 0.4,
    "TLT": 0.3,
    "GLD": 0.3
}


def test_portfolio_volatility():
    """Test the portfolio volatility computation."""
    vol = portfolio_volatility(MOCK_PORTFOLIO, MOCK_INFORMATION_SET)
    assert isinstance(vol, float), "Volatility should be a float."
    assert vol > 0, "Volatility should be positive."

def test_compute_risk_contributions():
    """Test the computation of risk contributions."""
    risk_contrib = compute_risk_contributions(MOCK_PORTFOLIO, MOCK_INFORMATION_SET)
    assert isinstance(risk_contrib, dict), "Risk contributions should be a dictionary."
    assert set(risk_contrib.keys()) == set(MOCK_PORTFOLIO.keys()), "Risk contributions keys should match portfolio keys."
    assert pytest.approx(sum(risk_contrib.values()), 0.01) == 1  # Risk contributions should sum to 1


def test_compute_portfolio_riskparity():
    """Test the basic risk parity portfolio computation."""
    risk_parity = RiskParity()
    portfolio = risk_parity.compute_portfolio_riskparity(
        t=datetime.now(),
        information_set=MOCK_INFORMATION_SET
    )
    assert abs(sum(portfolio.values()) - 1.0) < 1e-6, "Portfolio weights should sum to 1."
    assert all(w >= 0 for w in portfolio.values()), "All portfolio weights should be non-negative."

