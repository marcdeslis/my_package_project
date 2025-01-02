import yfinance as yf
import pandas as pd 
from sec_cik_mapper import StockMapper
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging 
from scipy.optimize import minimize
import numpy as np

import pybacktestchain
from pybacktestchain import data_module
from pybacktestchain.data_module import *

# Setup logging
logging.basicConfig(level=logging.INFO)

UNIVERSE_SEC = list(StockMapper().ticker_to_cik.keys())

def portfolio_volatility(portfolio: dict, information_set):
        """
        This function computes the volatility of a given portfolio.
        """
        Sigma = information_set.get('covariance_matrix')
        weights = np.array(list(portfolio.values()))
        
        # Compute portfolio variance and volatility
        portfolio_variance = np.dot(weights.T, np.dot(Sigma, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        return portfolio_volatility

def compute_risk_contributions(portfolio: dict, information_set):
        """
        Computes the risk contributions for a given portfolio.
        Returns:
            dict: A dictionary where keys are asset names and values are risk contributions.
        """
        Sigma = information_set['covariance_matrix']
        weights = np.array(list(portfolio.values()))
        portfolio_var = portfolio_volatility(portfolio, information_set)**2 # Portfolio variance
        marginal_contrib = np.dot(Sigma, weights)  # Marginal contributions
        risk_contrib = (weights * marginal_contrib) / portfolio_var   # Risk contributions

        # Convert risk contributions into a dictionary
        return {asset: round(risk_contrib[i], 3) for i, asset in enumerate(portfolio.keys())}

@dataclass 
class RiskParity(Information):
    def compute_portfolio_riskparity(self, t: datetime, information_set):
        """
        This function computes the risk parity portfolio
        """
        try:
            Sigma = information_set['covariance_matrix']
            n = len(Sigma)

            # Objective function to minimize the difference in risk contributions
            def risk_parity_obj(weights):
                portfolio_var =np.dot(weights.T, np.dot(Sigma, weights))
                marginal_contrib = np.dot(Sigma, weights)
                risk_contrib = (weights * marginal_contrib)/portfolio_var
                target_risk = np.mean(risk_contrib)
                return np.sum((risk_contrib - target_risk) ** 2) # we want to minimize the difference between the sum of the risk contributions of all assets and the target risk

            # Constraints: weights sum to 1
            cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            # Bounds: weights between 0 and 1
            bounds = [(0.0, 1.0)] * n
            # Initial guess: equal weights
            x0 = np.ones(n) / n
            # Minimize
            res = minimize(risk_parity_obj, x0, constraints=cons, bounds=bounds)

            # Prepare dictionary
            portfolio = {k: None for k in information_set['companies']}

            # If converged, update
            if res.success:
                for i, company in enumerate(information_set['companies']):
                    portfolio[company] = res.x[i]
            else:
                raise Exception("Optimization did not converge")

            return portfolio
        except Exception as e:
             # if something goes wrong return an equal weight portfolio but let the user know 
            logging.warning("Error computing portfolio, returning equal weight portfolio")
            logging.warning(e)
            return {k: 1/len(information_set['companies']) for k in information_set['companies']}
            

    def compute_portfolio_riskparity_voltarget_leverage(self, t: datetime, information_set, leverage_factor=1.0, target_volatility=0.1):
        """
        This function computes a risk parity portfolio with a target volatility and leverage factor.
        """
        try:
            Sigma = information_set['covariance_matrix']
            n = len(Sigma)

            # Objective function to minimize the difference in risk contributions
            def risk_parity_obj(weights):
                portfolio_var = np.dot(weights.T, np.dot(Sigma, weights))
                marginal_contrib = np.dot(Sigma, weights)
                risk_contrib = (weights * marginal_contrib) / portfolio_var
                target_risk = np.mean(risk_contrib)
                return np.sum((risk_contrib - target_risk) ** 2)

            # Constraints: weights sum to 1
            cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

            # Bounds: weights between 0 and 1
            bounds = [(0.0, 1.0)] * n

            # Initial guess: equal weights
            x0 = np.ones(n) / n

            # Minimize the risk parity objective
            res = minimize(risk_parity_obj, x0, constraints=cons, bounds=bounds)

            # Prepare the portfolio weights dictionary
            portfolio = {k: None for k in information_set['companies']}

            if res.success:
                # Retrieve the optimized weights
                risk_parity_weights = res.x

                # Compute the portfolio's current volatility
                portfolio_volatility = np.sqrt(np.dot(risk_parity_weights.T, np.dot(Sigma, risk_parity_weights)))
                logging.info(f"Original Volatility: {portfolio_volatility}")

                # We adjust the weights to match the desired target_volatility
                scaling_factor = target_volatility / portfolio_volatility
                scaled_weights = risk_parity_weights * scaling_factor

                # we multiply by the leverage_factor to achieve the desired total portfolio exposure:
                leveraged_weights = scaled_weights * leverage_factor

                # Update the portfolio dictionary
                for i, company in enumerate(information_set['companies']):
                    portfolio[company] = leveraged_weights[i]
            else:
                raise Exception("Optimization did not converge")

            # Return the final portfolio
            return portfolio

        except Exception as e:
            # If something goes wrong, return an equal weight portfolio scaled by leverage
            logging.warning("Error computing portfolio, returning equal weight portfolio")
            logging.warning(e)
            n_companies = len(information_set['companies'])
            return {k: leverage_factor / n_companies for k in information_set['companies']}
    
        
    

            


