import pandas as pd
import logging
from dataclasses import dataclass
from datetime import datetime

import os 
import pickle
from pybacktestchain.data_module import *
from pybacktestchain.utils import generate_random_name
from pybacktestchain.blockchain import Block, Blockchain
from pybacktestchain.broker import Position, Broker, RebalanceFlag, EndOfMonth, RiskModel, StopLoss

from my_package_project.data_treatment import *

from numba import jit 


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from datetime import timedelta, datetime

# #We decided to create a rebalancing flag class that would enable us to rebalance the portfolio if the portfolio volatility exceeds a certain threshold
# @dataclass
# class RebalanceFlag:
#     def time_to_rebalance(self, t: datetime):
#         pass 



@dataclass
class Backtest:
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
    






