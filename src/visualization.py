"""Interactive Plotly Visualizations"""
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


class Visualizer:
    def __init__(self, output_dir: str = "outputs/plots"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _save(self, fig, name: str) -> str:
        path = os.path.join(self.output_dir, name)
        fig.write_html(path)
        return path

    def price_chart(self, df: pd.DataFrame, ticker: str) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"], name="OHLC",
        ))
        fig.update_layout(
            title=f"{ticker} — Price History",
            xaxis_title="Date", yaxis_title="Price",
            template="plotly_white", height=500,
        )
        return fig

    def returns_chart(self, returns: pd.Series, ticker: str) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=returns.index, y=returns, mode="lines",
            line=dict(color="orange", width=1), name="Log Returns (%)",
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        fig.update_layout(
            title=f"{ticker} — Log Returns (Volatility Clustering)",
            xaxis_title="Date", yaxis_title="Return (%)",
            template="plotly_white", height=400,
        )
        return fig

    def volatility_chart(self, returns: pd.Series, cond_vol: pd.Series,
                         ticker: str) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=returns.index, y=returns.abs(),
            mode="lines", line=dict(color="lightgray", width=0.8),
            name="|Returns|",
        ))
        fig.add_trace(go.Scatter(
            x=cond_vol.index, y=cond_vol,
            mode="lines", line=dict(color="crimson", width=2),
            name="Conditional Volatility σ_t",
        ))
        fig.update_layout(
            title=f"{ticker} — GARCH Conditional Volatility",
            xaxis_title="Date", yaxis_title="Volatility (%)",
            template="plotly_white", height=450,
        )
        return fig

    def forecast_chart(self, cond_vol: pd.Series, forecast_df: pd.DataFrame,
                       ticker: str) -> go.Figure:
        tail = cond_vol.tail(120)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=tail.index, y=tail, mode="lines",
            line=dict(color="crimson", width=2), name="Historical σ_t",
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df.index, y=forecast_df["Forecast_Volatility"],
            mode="lines+markers", line=dict(color="blue", width=2, dash="dash"),
            name="Forecast σ_t",
        ))
        fig.update_layout(
            title=f"{ticker} — Volatility Forecast",
            xaxis_title="Date", yaxis_title="Volatility (%)",
            template="plotly_white", height=450,
        )
        return fig

    def returns_distribution(self, returns: pd.Series, ticker: str) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=returns, nbinsx=60, histnorm="probability density",
            name="Empirical", marker_color="steelblue", opacity=0.7,
        ))
        mu, sigma = returns.mean(), returns.std()
        x = np.linspace(returns.min(), returns.max(), 200)
        y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines", name="Normal Fit",
            line=dict(color="red", width=2),
        ))
        fig.update_layout(
            title=f"{ticker} — Return Distribution (Fat Tails Check)",
            xaxis_title="Return (%)", yaxis_title="Density",
            template="plotly_white", height=400,
        )
        return fig

    def var_backtest_chart(self, backtest_df: pd.DataFrame, ticker: str,
                           alpha: float) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=backtest_df.index, y=backtest_df["Actual_Return"],
            mode="lines", name="Actual Returns",
            line=dict(color="gray", width=1),
        ))
        fig.add_trace(go.Scatter(
            x=backtest_df.index, y=backtest_df["VaR"],
            mode="lines", name=f"VaR ({int(alpha*100)}%)",
            line=dict(color="red", width=2),
        ))
        breaches = backtest_df[backtest_df["Breach"]]
        fig.add_trace(go.Scatter(
            x=breaches.index, y=breaches["Actual_Return"],
            mode="markers", name="Breaches",
            marker=dict(color="red", size=8, symbol="x"),
        ))
        fig.update_layout(
            title=f"{ticker} — Rolling VaR Backtest",
            xaxis_title="Date", yaxis_title="Return (%)",
            template="plotly_white", height=450,
        )
        return fig

    def model_comparison_chart(self, comparison_df: pd.DataFrame) -> go.Figure:
        fig = make_subplots(rows=1, cols=2, subplot_titles=("AIC", "BIC"))
        df = comparison_df.dropna(subset=["AIC"])
        fig.add_trace(go.Bar(x=df["Model"], y=df["AIC"], marker_color="steelblue"),
                      row=1, col=1)
        fig.add_trace(go.Bar(x=df["Model"], y=df["BIC"], marker_color="coral"),
                      row=1, col=2)
        fig.update_layout(
            title="Model Comparison (Lower is Better)",
            template="plotly_white", height=400, showlegend=False,
        )
        return fig
