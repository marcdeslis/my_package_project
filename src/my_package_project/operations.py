import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from datetime import datetime

import os 
import pickle
from pybacktestchain.data_module import UNIVERSE_SEC, FirstTwoMoments, get_stocks_data, DataModule, Information
from pybacktestchain.utils import generate_random_name
from pybacktestchain.blockchain import Block, Blockchain
from pybacktestchain.broker import Position, Broker, RebalanceFlag, EndOfMonth, RiskModel, StopLoss

from my_package_project.data_treatment import portfolio_volatility, compute_risk_contributions, RiskParity
from my_package_project.graphs import PortfolioVisualizer, PortfolioVisualizer_over_time

from numba import jit 


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
    risk_model : type = None #StopLoss
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
        
        # Run the backtest
        portfolio_history = []
        timestamps = []

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
                timestamps.append(t)
                portfolio_history.append(portfolio)
                prices = info.get_prices(t)
                self.broker.execute_portfolio(portfolio, prices, t)

        # Create the visualizer with portfolio history and timestamps
        visualizer = PortfolioVisualizer_over_time(portfolio_history=portfolio_history, timestamps=timestamps)
        visualizer.plot_portfolio_weights_over_time()  # Call the plotting method on the instance

        prices_history = df.pivot(
            index=self.time_column,
            columns=self.company_column,
            values=self.adj_close_column
        )
        
        # Calculate annualized return
        annualized_return = visualizer.compute_annualized_returns(prices_history=prices_history)
        visualizer.plot_portfolio_value_over_time(broker=self.broker, prices_history=prices_history)
        

        logging.info(f"Backtest completed. Final portfolio value: {self.broker.get_portfolio_value(info.get_prices(self.final_date))}")
        logging.info(f"Annualized Return: {annualized_return:.2%}")

        df = self.broker.get_transaction_log()

        # Create backtests folder if it does not exist
        if not os.path.exists('backtests'):
            os.makedirs('backtests')
        



            

    ### Prochaine étape : faire une fonction qui compute la performance du portfolio et la comparer avec les performance d'un 40/60
    # fonction qui compute annualized return, volatility, sharpe ratio, 
    ### output : créer un notebook et y mettre tout
    ##Sharpe ratio pour comparer
        






