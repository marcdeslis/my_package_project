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
