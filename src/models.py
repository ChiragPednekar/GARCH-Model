"""GARCH Family Models"""
from arch import arch_model
import pandas as pd


class VolatilityModel:
    def __init__(self, returns: pd.Series):
        self.returns = returns.dropna()
        self.model = None
        self.result = None
        self.model_name = None

    def fit(self, vol: str = "GARCH", p: int = 1, o: int = 0, q: int = 1,
            dist: str = "normal", mean: str = "Constant"):
        """Unified fit method for ARCH, GARCH, EGARCH, GJR-GARCH."""
        kwargs = {
            "vol": vol, "p": p, "q": q,
            "mean": mean, "dist": dist,
        }
        if o > 0:
            kwargs["o"] = o

        self.model = arch_model(self.returns, **kwargs)
        self.result = self.model.fit(disp="off", show_warning=False)
        self.model_name = (
            f"{vol}({p},{o},{q})-{dist}" if o > 0 else f"{vol}({p},{q})-{dist}"
        )
        return self.result

    def fit_garch(self, p=1, q=1, dist="normal"):
        return self.fit(vol="GARCH", p=p, q=q, dist=dist)

    def fit_egarch(self, p=1, q=1, dist="normal"):
        return self.fit(vol="EGARCH", p=p, q=q, dist=dist)

    def fit_gjr(self, p=1, o=1, q=1, dist="normal"):
        return self.fit(vol="GARCH", p=p, o=o, q=q, dist=dist)

    def fit_arch(self, p=1, dist="normal"):
        return self.fit(vol="ARCH", p=p, q=0, dist=dist)

    def get_conditional_volatility(self) -> pd.Series:
        if self.result is None:
            raise RuntimeError("Model not fitted")
        return self.result.conditional_volatility

    def summary(self):
        return self.result.summary()
