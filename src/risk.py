"""Risk Metrics: VaR, CVaR, Risk-Adjusted Metrics"""
import numpy as np
import pandas as pd
from scipy import stats


class RiskAnalyzer:
    def __init__(self, returns: pd.Series, conditional_volatility: pd.Series,
                 forecast_volatility: pd.Series = None):
        self.returns = returns.dropna()
        self.cond_vol = conditional_volatility
        self.forecast_vol = forecast_volatility

    def historical_var(self, alpha: float = 0.95) -> float:
        """Historical VaR at confidence level alpha."""
        return float(np.percentile(self.returns, (1 - alpha) * 100))

    def historical_cvar(self, alpha: float = 0.95) -> float:
        """Expected shortfall (CVaR)."""
        var = self.historical_var(alpha)
        return float(self.returns[self.returns <= var].mean())

    def parametric_var(self, alpha: float = 0.95, sigma: float = None) -> float:
        """Parametric VaR assuming normality, using GARCH σ."""
        if sigma is None:
            sigma = float(self.cond_vol.iloc[-1])
        z = stats.norm.ppf(1 - alpha)
        mu = float(self.returns.mean())
        return mu + z * sigma

    def parametric_cvar(self, alpha: float = 0.95, sigma: float = None) -> float:
        if sigma is None:
            sigma = float(self.cond_vol.iloc[-1])
        z = stats.norm.ppf(1 - alpha)
        mu = float(self.returns.mean())
        return mu - sigma * stats.norm.pdf(z) / (1 - alpha)

    def forecast_var(self, alpha: float = 0.95) -> pd.Series:
        """VaR forecast based on GARCH forecast σ."""
        if self.forecast_vol is None:
            raise ValueError("No forecast volatility provided")
        z = stats.norm.ppf(1 - alpha)
        mu = float(self.returns.mean())
        return mu + z * self.forecast_vol

    def risk_report(self) -> dict:
        report = {}
        for alpha in [0.95, 0.99]:
            report[f"Historical VaR ({int(alpha*100)}%)"] = self.historical_var(alpha)
            report[f"Historical CVaR ({int(alpha*100)}%)"] = self.historical_cvar(alpha)
            report[f"Parametric VaR ({int(alpha*100)}%)"] = self.parametric_var(alpha)
            report[f"Parametric CVaR ({int(alpha*100)}%)"] = self.parametric_cvar(alpha)

        report["Current Volatility (%)"] = float(self.cond_vol.iloc[-1])
        report["Annualized Volatility (%)"] = float(self.cond_vol.iloc[-1] * np.sqrt(252))
        report["Sharpe Ratio (daily, rf=0)"] = (
            float(self.returns.mean() / self.returns.std())
            if self.returns.std() > 0 else 0.0
        )
        return report
