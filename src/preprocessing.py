"""Preprocessing: Returns calculation and statistical tests"""
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_arch
from scipy import stats


class Preprocessor:
    def __init__(self, df: pd.DataFrame, price_col: str = "Close"):
        self.df = df.copy()
        self.price_col = price_col

    def compute_log_returns(self) -> pd.DataFrame:
        self.df["Log_Return"] = np.log(self.df[self.price_col] / self.df[self.price_col].shift(1))
        self.df["Return_Pct"] = self.df["Log_Return"] * 100
        self.df.dropna(inplace=True)
        return self.df

    def adf_test(self, series: pd.Series) -> dict:
        result = adfuller(series.dropna(), autolag="AIC")
        return {
            "ADF Statistic": float(result[0]),
            "p-value": float(result[1]),
            "Lags Used": int(result[2]),
            "Observations": int(result[3]),
            "Critical Values": {k: float(v) for k, v in result[4].items()},
            "Stationary": bool(result[1] < 0.05),
        }

    def arch_effect_test(self, series: pd.Series, lags: int = 10) -> dict:
        """Engle's ARCH-LM test for heteroskedasticity."""
        lm_stat, lm_pval, f_stat, f_pval = het_arch(series.dropna(), nlags=lags)
        return {
            "LM Statistic": float(lm_stat),
            "LM p-value": float(lm_pval),
            "F Statistic": float(f_stat),
            "F p-value": float(f_pval),
            "ARCH Effects Present": bool(lm_pval < 0.05),
        }

    def summary_stats(self, series: pd.Series) -> dict:
        s = series.dropna()
        return {
            "Mean": float(s.mean()),
            "Std Dev": float(s.std()),
            "Skewness": float(stats.skew(s)),
            "Kurtosis": float(stats.kurtosis(s)),
            "Min": float(s.min()),
            "Max": float(s.max()),
            "Observations": int(len(s)),
        }
