from my_package_project import my_package_project
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from my_package_project.graphs import PortfolioVisualizer, PortfolioVisualizer_over_time

from pybacktestchain.broker import Broker


# Mock data for testing
MOCK_PORTFOLIO = {
    "SPY": 0.4,
    "TLT": 0.3,
    "GLD": 0.3
}

MOCK_INFORMATION_SET = {
    "covariance_matrix": np.array([
        [0.04, 0.02, 0.01],
        [0.02, 0.03, 0.015],
        [0.01, 0.015, 0.05]
    ]),
    "companies": ["SPY", "TLT", "GLD"]
}

MOCK_PORTFOLIO_HISTORY = [
    {"SPY": 0.4, "TLT": 0.3, "GLD": 0.3},
    {"SPY": 0.35, "TLT": 0.4, "GLD": 0.25},
    {"SPY": 0.3, "TLT": 0.5, "GLD": 0.2}
]

MOCK_PRICES_HISTORY = pd.DataFrame({
    "SPY": [400, 405, 410],
    "TLT": [110, 112, 115],
    "GLD": [180, 182, 185]
}, index=pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]))

MOCK_TIMESTAMPS = ["2024-01-01", "2024-02-01", "2024-03-01"]
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from my_package_project.graphs import PortfolioVisualizer, PortfolioVisualizer_over_time
from my_package_project.data_treatment import compute_risk_contributions

# Mock data for testing
MOCK_PORTFOLIO = {
    "SPY": 0.4,
    "TLT": 0.3,
    "GLD": 0.3
}

MOCK_INFORMATION_SET = {
    "covariance_matrix": np.array([
        [0.04, 0.02, 0.01],
        [0.02, 0.03, 0.015],
        [0.01, 0.015, 0.05]
    ]),
    "companies": ["SPY", "TLT", "GLD"]
}

MOCK_PORTFOLIO_HISTORY = [
    {"SPY": 0.4, "TLT": 0.3, "GLD": 0.3},
    {"SPY": 0.35, "TLT": 0.4, "GLD": 0.25},
    {"SPY": 0.33, "TLT": 0.33, "GLD": 0.34},
]
MOCK_TIMESTAMPS = [
    datetime(2024, 1, 1),
    datetime(2024, 2, 1),
    datetime(2024, 3, 1),
]

MOCK_PRICES_HISTORY = pd.DataFrame({
    "SPY": [100, 105, 110],
    "TLT": [50, 52, 54],
    "GLD": [200, 198, 202],
}, index=pd.to_datetime(MOCK_TIMESTAMPS))



def test_plot_portfolio_weights():
    """Test the PortfolioVisualizer's plot_portfolio_weights method."""
    visualizer = PortfolioVisualizer(portfolio=MOCK_PORTFOLIO, information_set=MOCK_INFORMATION_SET)
    fig = visualizer.plot_portfolio_weights()
    assert fig is not None, "The plot should not be None."


def test_plot_risk_allocation_pie():
    """Test the PortfolioVisualizer's plot_risk_allocation_pie method."""
    visualizer = PortfolioVisualizer(portfolio=MOCK_PORTFOLIO, information_set=MOCK_INFORMATION_SET)
    fig = visualizer.plot_risk_allocation_pie()
    assert fig is not None, "The risk allocation pie chart should not be None."


def test_plot_portfolio_weights_over_time():
    """Test the PortfolioVisualizer_over_time's plot_portfolio_weights_over_time method."""
    visualizer = PortfolioVisualizer_over_time(
        portfolio_history=MOCK_PORTFOLIO_HISTORY, 
        timestamps=MOCK_TIMESTAMPS
    )
    fig = visualizer.plot_portfolio_weights_over_time()
    assert fig is not None, "The portfolio weights over time plot should not be None."


def test_compute_annualized_returns():
    """Test the compute_annualized_returns method."""
    visualizer = PortfolioVisualizer_over_time(
        portfolio_history=MOCK_PORTFOLIO_HISTORY, 
        timestamps=MOCK_TIMESTAMPS
    )
    annualized_returns = visualizer.compute_annualized_returns(prices_history=MOCK_PRICES_HISTORY)
    assert isinstance(annualized_returns, float), "Annualized returns should be a float."


def test_compute_annualized_volatility():
    """Test the compute_annualized_volatility method."""
    visualizer = PortfolioVisualizer_over_time(
        portfolio_history=MOCK_PORTFOLIO_HISTORY, 
        timestamps=MOCK_TIMESTAMPS
    )
    annualized_volatility = visualizer.compute_annualized_volatility(prices_history=MOCK_PRICES_HISTORY)
    assert isinstance(annualized_volatility, float), "Annualized volatility should be a float."
    assert annualized_volatility > 0, "Annualized volatility should be positive."


def test_plot_portfolio_value_over_time():
    """Test the plot_portfolio_value_over_time method."""
    from pybacktestchain.broker import Broker

    broker = Broker(cash=1000000)
    visualizer = PortfolioVisualizer_over_time(
        portfolio_history=MOCK_PORTFOLIO_HISTORY, 
        timestamps=MOCK_TIMESTAMPS
    )
    fig = visualizer.plot_portfolio_value_over_time(broker=broker, prices_history=MOCK_PRICES_HISTORY)
    assert fig is not None, "The portfolio value over time plot should not be None."


def test_compute_sharpe_ratio():
    """Test the computation of the Sharpe Ratio."""
    visualizer = PortfolioVisualizer_over_time(
        portfolio_history=MOCK_PORTFOLIO_HISTORY,
        timestamps=MOCK_TIMESTAMPS
    )
    risk_free_rate = 0.02  # 3% annualized risk-free rate
    sharpe_ratio = visualizer.compute_sharpe_ratio(prices_history=MOCK_PRICES_HISTORY, risk_free_rate=risk_free_rate)
    
    assert isinstance(sharpe_ratio, float), "Sharpe Ratio should be a float."
    