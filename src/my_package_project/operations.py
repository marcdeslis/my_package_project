import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from datetime import datetime
import plotly.express as px
import os 
import pickle
from pybacktestchain.data_module import UNIVERSE_SEC, FirstTwoMoments, get_stocks_data, DataModule, Information
from pybacktestchain.utils import generate_random_name
from pybacktestchain.blockchain import Block, Blockchain
from pybacktestchain.broker import Position, Broker, RebalanceFlag, EndOfMonth, RiskModel, StopLoss

from my_package_project.data_treatment import portfolio_volatility, compute_risk_contributions, RiskParity
from my_package_project.graphs import PortfolioVisualizer, PortfolioVisualizer_over_time

from numba import jit 
import nbformat


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from datetime import timedelta, datetime



@dataclass
class Backtest_up:
    initial_date: datetime
    final_date: datetime
    universe = ['SPY', 'TLT', 'GLD']
    information_class : type  = Information
    s: timedelta = timedelta(days=360)
    time_column: str = 'Date'
    company_column: str = 'ticker'
    adj_close_column : str ='Adj Close'
    rebalance_flag : type = EndOfMonth
    risk_model : type = None # or StopLoss
    initial_cash: int = 1000000  # Initial cash in the portfolio
    name_blockchain: str = 'backtest'
    verbose: bool = True
    broker = Broker(cash=initial_cash, verbose=verbose)

    def __post_init__(self):
        self.backtest_name = generate_random_name()

    def run_backtest(self):
        logging.info(f"Running backtest from {self.initial_date} to {self.final_date}.")
        logging.info(f"Retrieving price data for universe")
        self.risk_model = self.risk_model(threshold=0.1)
        
        # Format initial and final dates
        init_ = self.initial_date.strftime('%Y-%m-%d')
        final_ = self.final_date.strftime('%Y-%m-%d')
        df = get_stocks_data(self.universe, init_, final_)

        # Initialize the DataModule
        data_module = DataModule(df)

        # Create the Information object
        info = self.information_class(s=self.s, 
                                    data_module=data_module,
                                    time_column=self.time_column,
                                    company_column=self.company_column,
                                    adj_close_column=self.adj_close_column)
        
        # Prepare backtest folder
        backtest_folder = os.path.join('backtests', self.backtest_name)
        if not os.path.exists(backtest_folder):
            os.makedirs(backtest_folder)

        # Run the backtest
        portfolio_history = []
        timestamps = []
        initial_portfolio = None

        for t in pd.date_range(start=self.initial_date, end=self.final_date, freq='D'):
            if self.risk_model is not None:
                portfolio = info.compute_portfolio_riskparity(t, info.compute_information(t))
                prices = info.get_prices(t)
                self.risk_model.trigger_stop_loss(t, portfolio, prices, self.broker)
            
            if self.rebalance_flag().time_to_rebalance(t):
                logging.info("-----------------------------------")
                logging.info(f"Rebalancing portfolio at {t}")
                information_set = info.compute_information(t)
                portfolio = info.compute_portfolio_riskparity(t, information_set)
                if initial_portfolio is None:  # Capture initial weights
                    initial_portfolio = portfolio
                    information_0= information_set
                timestamps.append(t)
                portfolio_history.append(portfolio)
                prices = info.get_prices(t)
                self.broker.execute_portfolio(portfolio, prices, t)
        
        # Plots of initial weights & save the graph
        initial_visualization = PortfolioVisualizer(portfolio=initial_portfolio, information_set=information_0)
        fig_weights = initial_visualization.plot_portfolio_weights()
        weights_path = os.path.join(backtest_folder, 'initial_weights.png')
        fig_weights.write_image(weights_path)

         # Compute risk contributions
        risk_contributions = compute_risk_contributions(portfolio=initial_portfolio,information_set= information_0)
        print(f"Risk Contributions: {risk_contributions}")
        # Plot risk allocation pie chart and save the graph
        fig_risk=initial_visualization.plot_risk_allocation_pie()
        risk_path = os.path.join(backtest_folder, 'risk_allocation_pie.png')
        fig_risk.write_image(risk_path)

        #Plot portfolio graphs over time and save it
        visualizer = PortfolioVisualizer_over_time(portfolio_history=portfolio_history, timestamps=timestamps)
        fig_weights_time = visualizer.plot_portfolio_weights_over_time()
        weights_time_path = os.path.join(backtest_folder, 'weights_over_time.png')
        fig_weights_time.write_image(weights_time_path)

        prices_history = df.pivot(index=self.time_column,columns=self.company_column,values=self.adj_close_column)

        # Calculate portfolio volatility
        ptf_vol = visualizer.compute_annualized_volatility(prices_history=prices_history)
        print(f"Portfolio Volatility: {ptf_vol:.2%}")

        # Calculate annualized return
        annualized_return = visualizer.compute_annualized_returns(prices_history=prices_history)
        print(f"Annualized Return: {annualized_return:.2%}")    

        # Calculate Sharpe Ratio
        risk_free_rate = 0.02  # Example: 2% annual risk-free rate
        sharpe_ratio = visualizer.compute_sharpe_ratio(prices_history=prices_history, risk_free_rate=risk_free_rate)
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

        #Plot the portfolio value over time
        fig_portfolio_value = visualizer.plot_portfolio_value_over_time(broker=self.broker, prices_history=prices_history)
        portfolio_value_path = os.path.join(backtest_folder, 'portfolio_value_over_time.png')
        fig_portfolio_value.write_image(portfolio_value_path)

        # save to csv, use the backtest name 
        df = self.broker.get_transaction_log()
        df.to_csv(os.path.join(backtest_folder, 'transaction_log.csv'))

        logging.info(f" Backtest completed. Final portfolio value: {self.broker.get_portfolio_value(info.get_prices(self.final_date))}")


        #We create a jupyter notebook file in which we insert all the informations 

        # Create Jupyter Notebook
        notebook_path = os.path.join(backtest_folder, f"{self.backtest_name}.ipynb")
        notebook = nbformat.v4.new_notebook()
        
        # Add cells to the notebook
        notebook['cells'] = [
            nbformat.v4.new_markdown_cell("## <center> Backtest Results Notebook </center>"),
            nbformat.v4.new_markdown_cell(f"Backtest Name: {self.backtest_name}"),
            nbformat.v4.new_markdown_cell(f"Backtest Period: {self.initial_date.strftime('%Y-%m-%d')} to {self.final_date.strftime('%Y-%m-%d')}"),
            nbformat.v4.new_markdown_cell(f"### Stocks Used in Backtest:\n- " + "\n- ".join(self.universe)),
            nbformat.v4.new_markdown_cell("Initial Portfolio Weights"),
            nbformat.v4.new_code_cell(f"from IPython.display import Image\nImage(filename='initial_weights.png')"),
            nbformat.v4.new_markdown_cell("Risk Contributions Pie Chart"),
            nbformat.v4.new_code_cell(f"Image(filename='risk_allocation_pie.png')"),
            nbformat.v4.new_markdown_cell("Portfolio Weights Over Time"),
            nbformat.v4.new_code_cell(f"Image(filename='weights_over_time.png')"),
            nbformat.v4.new_markdown_cell(f"- **Annualized Volatility**: {ptf_vol:.2%}\n" 
                                      f"- **Annualized Return**: {annualized_return:.2%}\n"
                                      f"- **Sharpe Ratio**: {sharpe_ratio:.2f}"),
            nbformat.v4.new_markdown_cell("Portfolio Value Over Time"),
            nbformat.v4.new_code_cell(f"Image(filename='portfolio_value_over_time.png')"),
            nbformat.v4.new_markdown_cell("Transaction Log"),
            nbformat.v4.new_code_cell(f"import pandas as pd\ndf = pd.read_csv('transaction_log.csv').drop(columns=['Unnamed: 0'])\ndf.head()")
        ]

        # Save the notebook
        with open(notebook_path, 'w') as f:
            nbformat.write(notebook, f)

        return backtest_folder
    
    #push sur pypi








