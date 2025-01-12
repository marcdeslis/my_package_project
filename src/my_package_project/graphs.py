import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import os 
import pickle
from pybacktestchain.data_module import *
from pybacktestchain.utils import generate_random_name
from pybacktestchain.blockchain import Block, Blockchain
from pybacktestchain.broker import Position, Broker, RebalanceFlag, EndOfMonth, RiskModel, StopLoss

import my_package_project.data_treatment as data_treatment

import plotly.graph_objects as go
import matplotlib.cm as cm
import matplotlib.colors as mcolors

@dataclass 
class PortfolioVisualizer:
    portfolio : dict
    information_set: dict
    #transaction_log: pd.DataFrame 

    def plot_portfolio_weights(self):
        """
        This function plots the portfolio weights
        """
        if not self.portfolio:
            raise ValueError("Portfolio is empty or invalid.")
       
        #Extract asset names and weights
        assets = list(self.portfolio.keys())
        weights = list(self.portfolio.values())

        colors = ['#1F77B4', '#b41f49', '#f7cf07', '#FFA500', '#32CD32']  
        extended_colors = colors * (len(assets) // len(colors)) + colors[:len(assets) % len(colors)]

        # Create an interactive bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=assets,
                y=weights,
                text=[f"{w*100:.2f}%" for w in weights],
                textposition='auto',
                marker_color=extended_colors 
            )
        ])

        # Add layout details
        fig.update_layout(
            title="Initial Risk Parity Portfolio Weights",
            xaxis_title="Assets",
            yaxis_title="Weights",
            xaxis=dict(tickmode='linear'),
            width=700,  # Adjust width
            height=400,  # Adjust height
        )
        fig.show()
        return fig

    def plot_risk_allocation_pie(self):
        """
        Plots the risk allocation of a portfolio.
        """
        if not self.portfolio:
            raise ValueError("Portfolio is empty or invalid.")

        # Compute risk contributions
        risk_contributions = data_treatment.compute_risk_contributions(self.portfolio, self.information_set)

        # Extract assets and their risk contributions
        assets = list(risk_contributions.keys())
        contributions = list(risk_contributions.values())

        colors = ['#1F77B4', '#b41f49', '#f7cf07', '#FFA500', '#32CD32'] 
        extended_colors = colors * (len(assets) // len(colors)) + colors[:len(assets) % len(colors)]

        # Create a pie chart
        fig = go.Figure(data=[
            go.Pie(
                labels=assets,
                values=contributions,
                textinfo='label+percent',
                hoverinfo='label+value',
                marker=dict(colors=extended_colors)
            )
        ])

        # Add layout details
        fig.update_layout(
            title="Risk Parity Allocation Pie Chart",
            template='plotly_white'
        )
        fig.show()
        return fig

@dataclass
class PortfolioVisualizer_over_time:
    portfolio_history: list  # List of dictionaries with portfolio weights over time
    timestamps: list  # List of timestamps corresponding to portfolio weights

    def plot_portfolio_weights_over_time(self):
        """
        This function plots the portfolio weights over time.
        """
        if not self.portfolio_history or not self.timestamps:
            raise ValueError("Portfolio history or timestamps are empty or invalid.")

        # Ensure portfolio history matches timestamps
        if len(self.portfolio_history) != len(self.timestamps):
            raise ValueError("Portfolio history and timestamps length mismatch.")
        
        # Create a DataFrame to store weights over time
        weights_df = pd.DataFrame(self.portfolio_history, index=pd.to_datetime(self.timestamps))

        # Ensure column names are assets
        weights_df.columns.name = "Assets"

        # Define colors dynamically to match the number of assets
        colors = ['#1F77B4', '#b41f49', '#f7cf07', '#FFA500', '#32CD32', '#9467BD', '#E377C2', '#8C564B', '#7F7F7F', '#BCBD22']
        extended_colors = colors * (len(weights_df.columns) // len(colors)) + colors[:len(weights_df.columns) % len(colors)]

        # Create an interactive stacked area chart
        fig = go.Figure()

        # Add each asset as a trace with its corresponding color
        for asset, color in zip(weights_df.columns, extended_colors):
            fig.add_trace(go.Scatter(
                x=weights_df.index,
                y=weights_df[asset],
                mode='lines',
                stackgroup='one',  # Enable stacking
                name=asset,
                line=dict(color=color)  # Apply color to each trace
            ))

        # Add layout details
        fig.update_layout(
            title="Risk Parity Portfolio Weights Over Time",
            xaxis_title="Date",
            yaxis_title="Weights",
            yaxis=dict(tickformat=".0%", range=[0, 1]),  # Percentage format and limit to [0, 1]
            xaxis=dict(tickformat="%Y-%m-%d"),
            legend_title="Assets",
            width=900,  # Adjust width
            height=500,  # Adjust height
        )

        fig.show()
        return fig


    
    def compute_annualized_returns(self, prices_history: pd.DataFrame):
        """
        This method computes the annualized returns of the portfolio over time.
        """
        if not self.portfolio_history or not self.timestamps:
            raise ValueError("Portfolio history or timestamps are empty or invalid.")
        if prices_history is None or prices_history.empty:
            raise ValueError("Prices history is empty or invalid.")

        # Align weights and prices by ensuring timestamps match
        weights_df = pd.DataFrame(self.portfolio_history, index=pd.to_datetime(self.timestamps))
        common_dates = weights_df.index.intersection(prices_history.index)  # Find common dates
        weights_df = weights_df.loc[common_dates]
        prices_history = prices_history.loc[common_dates]

        # Normalize prices to start at 1 
        normalized_prices = prices_history / prices_history.iloc[0]

        # Calculate daily portfolio returns
        portfolio_returns = (weights_df * normalized_prices).sum(axis=1).pct_change()
        portfolio_returns = portfolio_returns.dropna()

        # Compute cumulative returns
        cumulative_returns = (1 + portfolio_returns).cumprod()

         # Annualized return = (Ending Value / Beginning Value)^(12/Months) - 1
        total_months = len(cumulative_returns) - 1  # Number of monthly periods
        annualized_returns = (cumulative_returns.iloc[-1] / cumulative_returns.iloc[0]) ** (12 / total_months) - 1

        return annualized_returns
    
    def plot_portfolio_value_over_time(self, broker: Broker, prices_history: pd.DataFrame):
        """
        This method computes and plots the portfolio value over time using the Broker class.

        Args:
            broker (Broker): An instance of the Broker class.
            prices_history (pd.DataFrame): Historical prices of assets in the portfolio.
        """
        if not self.portfolio_history or not self.timestamps:
            raise ValueError("Portfolio history or timestamps are empty or invalid.")
        if prices_history is None or prices_history.empty:
            raise ValueError("Prices history is empty or invalid.")
        if broker is None:
            raise ValueError("Broker instance is required.")

        # Align weights and prices by ensuring timestamps match
        weights_df = pd.DataFrame(self.portfolio_history, index=pd.to_datetime(self.timestamps))
        common_dates = weights_df.index.intersection(prices_history.index)  # Find common dates
        weights_df = weights_df.loc[common_dates]
        prices_history = prices_history.loc[common_dates]

        # Initialize portfolio values
        portfolio_values = []

        for date in common_dates:
            # Get market prices for the date
            market_prices = prices_history.loc[date].to_dict()

            # Calculate portfolio value using the Broker class
            portfolio_value = broker.get_portfolio_value(market_prices)
            portfolio_values.append(portfolio_value)

        # Create a DataFrame for visualization
        portfolio_values_df = pd.DataFrame({
            'Date': common_dates,
            'Portfolio Value': portfolio_values
        }).set_index('Date')

        # Plot portfolio value over time
        fig = go.Figure(data=go.Scatter(
            x=portfolio_values_df.index,
            y=portfolio_values_df['Portfolio Value'],
            mode='lines',
            name='Portfolio Value'
        ))

        fig.update_layout(
            title="Portfolio Value Over Time",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            xaxis=dict(tickformat="%Y-%m-%d"),
            yaxis=dict(tickprefix="$"),
            width=900,
            height=500,
        )

        fig.show()
        return fig

    def compute_annualized_volatility(self, prices_history: pd.DataFrame):
        """
        Computes the annualized volatility of the portfolio over time.
        """
        if not self.portfolio_history or prices_history.empty:
            raise ValueError("Portfolio history or prices history is empty or invalid.")
        
        # Convert portfolio history to DataFrame
        weights_df = pd.DataFrame(self.portfolio_history, index=pd.to_datetime(self.timestamps))
        common_dates = weights_df.index.intersection(prices_history.index)  # Find common dates
        weights_df = weights_df.loc[common_dates]
        prices_history = prices_history.loc[common_dates]
        
        # Normalize prices to start at 1 
        normalized_prices = prices_history / prices_history.iloc[0]

        # Calculate daily portfolio returns
        portfolio_returns = (weights_df * normalized_prices).sum(axis=1).pct_change()
        portfolio_returns = portfolio_returns.dropna()

        # Calculate daily volatility (standard deviation)
        monthly_volatility = portfolio_returns.std() 

        # Annualize volatility
        annualized_volatility = monthly_volatility * np.sqrt(12)

        return annualized_volatility
    


    def compute_sharpe_ratio(self, prices_history: pd.DataFrame, risk_free_rate: float = 0.02):
        """
        Computes the Sharpe Ratio of the portfolio over time using annualized return and volatility.

        Args:
            prices_history (pd.DataFrame): Historical prices of assets in the portfolio.
            risk_free_rate (float): The annualized risk-free rate, default is 2% (0.02).

        Returns:
            float: The Sharpe Ratio of the portfolio.
        """
        if not self.portfolio_history or not self.timestamps:
            raise ValueError("Portfolio history or timestamps are empty or invalid.")
        if prices_history is None or prices_history.empty:
            raise ValueError("Prices history is empty or invalid.")

        # Use existing methods to compute annualized return and volatility
        annualized_return = self.compute_annualized_returns(prices_history)
        annualized_volatility = self.compute_annualized_volatility(prices_history)

        # Compute excess return
        excess_return = annualized_return - risk_free_rate

        # Compute Sharpe Ratio
        if annualized_volatility == 0:
            raise ValueError("Annualized volatility is zero; Sharpe Ratio cannot be computed.")
        sharpe_ratio = excess_return / annualized_volatility

        return sharpe_ratio

