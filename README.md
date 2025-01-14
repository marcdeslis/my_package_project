# my_package_project

A package that I created for my Python project :)

## Installation

```bash
$ pip install my_package_project
```

## Usage

- This package is an extenstion of the `pybacktestchain` package. 
- This package is designed to provide tools for backtesting financial strategies, with a focus on risk parity portfolios.
- It enables to evaluate portfolio performance through various metrics, including portfolio volatility, annaulized return, risk contributions or Sharpe ratio.
- It also includes functionality to visualize portfolio characteristics and performance over time.

- Output and Reporting:
  - Generates interactive visualizations.
  - Exports results as a Jupyter Notebook for detailed review.
  - Saves plots and transaction logs for further analysis.

## Example of usage 

```python
from datetime import date, timedelta, datetime
from my_package_project.data_treatment import *
from my_package_project.graphs  import *
from my_package_project.operations import *

test = Backtest_up(
    initial_date = datetime(2017, 1, 1),
    final_date = datetime(2020, 12, 31),
    information_class = FirstTwoMoments,
    risk_model=StopLoss,
)
test.run_backtest()

```

## Key classes and functions 

Backtest Framework: (operations)
- `Backtest_up`:
  - Runs the backtest, calculates metrics, and generates reports.

Portfolio Visualizations: (graphs)
- `PortfolioVisualizer`:
  - Visualizes initial portfolio weights and risk contributions.
  - Methods:
    - `plot_portfolio_weights`: Bar chart of portfolio weights.
    - `plot_risk_allocation_pie`: Pie chart of risk contributions.
- `PortfolioVisualizer_over_time`:
  - Tracks portfolio performance and weights over time.
  - Methods:
    - `plot_portfolio_weights_over_time`: Stacked area chart of weights.
    - `plot_portfolio_value_over_time`: Line chart of portfolio value.
    - `compute_annualized_returns`: Calculates portfolio annualized returns.
    - `compute_annualized_volatility`: Calculates portfolio annualized volatility.
    - `compute_sharpe_ratio`: Calculates the Sharpe ratio.

Risk Parity Framework: (data_treatment)

- `compute_risk_contributions`: Computes the contribution of each asset to portfolio risk.
- `portfolio_volatility`: Calculates portfolio volatility using covariance matrix.
- `RiskParity`:
  - `compute_portfolio_riskparity`: Constructs risk parity portfolio.
  - `compute_portfolio_riskparity_voltarget_leverage` : Constructs risk parity portfolios with optional leverage and target volatility.


## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`my_package_project` was created by marcdeslis. It is licensed under the terms of the MIT license.

## Credits

`my_package_project` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
