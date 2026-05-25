"""Evaluation Metrics"""
import numpy as np
import pandas as pd


class ModelEvaluator:
    def __init__(self, fitted_result):
        self.result = fitted_result

    def metrics(self) -> dict:
        return {
            "Log-Likelihood": float(self.result.loglikelihood),
            "AIC": float(self.result.aic),
            "BIC": float(self.result.bic),
            "Observations": int(self.result.nobs),
            "Parameters": {k: float(v) for k, v in self.result.params.items()},
            "P-values": {k: float(v) for k, v in self.result.pvalues.items()},
        }

    def in_sample_mse(self, returns: pd.Series) -> float:
        realized = returns.dropna() ** 2
        cond_var = self.result.conditional_volatility ** 2
        aligned = pd.concat([realized, cond_var], axis=1, join="inner").dropna()
        return float(np.mean((aligned.iloc[:, 0] - aligned.iloc[:, 1]) ** 2))

    def qlike_loss(self, returns: pd.Series) -> float:
        """QLIKE loss for volatility models (robust to noisy proxies)."""
        realized = returns.dropna() ** 2
        cond_var = self.result.conditional_volatility ** 2
        aligned = pd.concat([realized, cond_var], axis=1, join="inner").dropna()
        r = aligned.iloc[:, 0].values
        p = aligned.iloc[:, 1].values
        p = np.where(p <= 0, 1e-8, p)
        return float(np.mean(np.log(p) + r / p))
