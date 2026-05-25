"""Basic smoke tests"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from src import Preprocessor, VolatilityModel, VolatilityForecaster, RiskAnalyzer


def _synthetic_returns(n: int = 1000, seed: int = 42) -> pd.Series:
    np.random.seed(seed)
    returns = np.random.normal(0, 1, n) * (1 + 0.3 * np.sin(np.arange(n) / 50))
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(returns, index=idx, name="Return_Pct")


def test_preprocessor():
    prices = pd.DataFrame(
        {"Close": np.cumprod(1 + np.random.normal(0, 0.01, 200)) * 100}
    )
    prices.index = pd.bdate_range("2022-01-01", periods=200)
    pre = Preprocessor(prices)
    df  = pre.compute_log_returns()
    assert "Return_Pct" in df.columns
    assert len(df) == 199


def test_garch_fit_and_forecast():
    r = _synthetic_returns()
    m = VolatilityModel(r)
    m.fit_garch(1, 1)
    assert m.result is not None
    fc = VolatilityForecaster(m.result).forecast(horizon=5)
    assert len(fc) == 5
    assert (fc["Forecast_Volatility"] > 0).all()


def test_risk():
    r = _synthetic_returns()
    m = VolatilityModel(r)
    m.fit_garch(1, 1)
    risk   = RiskAnalyzer(r, m.result.conditional_volatility)
    report = risk.risk_report()
    assert "Current Volatility (%)" in report


if __name__ == "__main__":
    test_preprocessor()
    print("✅ preprocessor OK")
    test_garch_fit_and_forecast()
    print("✅ GARCH fit OK")
    test_risk()
    print("✅ Risk OK")
    print("\n🎉 All tests passed")
