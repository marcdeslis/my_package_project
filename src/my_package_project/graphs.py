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

        colors = ['#FFA500', '#32CD32', '#1F77B4', '#b41f49', '#f7cf07']  # Orange, Green, Blue, Yeallow, Red
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
            title="Portfolio Weights",
            xaxis_title="Assets",
            yaxis_title="Weights",
            xaxis=dict(tickmode='linear'),
            width=700,  # Adjust width
            height=400,  # Adjust height
        )
        fig.show()

    def plot_risk_allocation_pie(self):
        """
        Plots therisk allocation of a portfolio.

        Args:
            portfolio (dict): A dictionary where keys are asset names and values are weights.
            information_set (dict): Contains the covariance matrix under 'covariance_matrix'.
        """
        if not self.portfolio:
            raise ValueError("Portfolio is empty or invalid.")

        # Compute risk contributions
        risk_contributions = data_treatment.compute_risk_contributions(self.portfolio, self.information_set)

        # Extract assets and their risk contributions
        assets = list(risk_contributions.keys())
        contributions = list(risk_contributions.values())

        colors = ['#FFA500', '#32CD32', '#1F77B4', '#b41f49', '#f7cf07']  # Orange, Green, Blue, Yeallow, Red
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
            title="Risk Allocation Pie Chart",
            template='plotly_white'
        )
        fig.show()


       






    