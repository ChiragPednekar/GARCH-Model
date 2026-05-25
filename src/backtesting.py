"""Rolling Window Backtesting for VaR/Volatility"""
import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats


class Backtester:
    def __init__(self, returns: pd.Series, window: int = 500):
        self.returns = returns.dropna()
        self.window = window

    def rolling_var_backtest(self, alpha: float = 0.95, refit_every: int = 20,
                              vol_model: str = "GARCH", p: int = 1, q: int = 1,
                              dist: str = "normal") -> pd.DataFrame:
        """Rolling-window VaR backtest; refits model every N days."""
        n = len(self.returns)
        results = []
        last_fit = None

        for i in range(self.window, n):
            train = self.returns.iloc[i - self.window:i]
            if last_fit is None or (i - self.window) % refit_every == 0:
                try:
                    model = arch_model(train, vol=vol_model, p=p, q=q,
                                       mean="Constant", dist=dist)
                    last_fit = model.fit(disp="off", show_warning=False)
                except Exception:
                    continue

            forecast = last_fit.forecast(horizon=1, reindex=False)
            sigma = float(np.sqrt(forecast.variance.iloc[-1, 0]))
            mu = float(train.mean())
            z = stats.norm.ppf(1 - alpha)
            var_estimate = mu + z * sigma

            actual = float(self.returns.iloc[i])
            results.append({
                "Date": self.returns.index[i],
                "Actual_Return": actual,
                "VaR": var_estimate,
                "Forecast_Vol": sigma,
                "Breach": actual < var_estimate,
            })

        return pd.DataFrame(results).set_index("Date")

    def kupiec_test(self, backtest_df: pd.DataFrame, alpha: float = 0.95) -> dict:
        """Kupiec POF (proportion of failures) test."""
        N = len(backtest_df)
        x = int(backtest_df["Breach"].sum())
        p = 1 - alpha
        if x == 0 or x == N:
            return {
                "LR_stat": None, "p_value": None, "N": N,
                "Breaches": x, "Breach Rate": x / N if N > 0 else 0,
                "Expected Rate": p,
            }
        pi_hat = x / N
        lr = -2 * (
            (N - x) * np.log(1 - p) + x * np.log(p)
            - (N - x) * np.log(1 - pi_hat) - x * np.log(pi_hat)
        )
        p_val = 1 - stats.chi2.cdf(lr, df=1)
        return {
            "LR_stat": float(lr),
            "p_value": float(p_val),
            "N": N,
            "Breaches": x,
            "Breach Rate": x / N,
            "Expected Rate": p,
            "Model Adequate": bool(p_val > 0.05),
        }
