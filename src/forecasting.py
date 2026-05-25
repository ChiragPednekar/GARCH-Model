"""Volatility Forecasting"""
import numpy as np
import pandas as pd


class VolatilityForecaster:
    def __init__(self, fitted_result):
        self.result = fitted_result

    def forecast(self, horizon: int = 10) -> pd.DataFrame:
        forecasts = self.result.forecast(horizon=horizon, reindex=False)
        variance_forecast = forecasts.variance.iloc[-1].values
        volatility_forecast = np.sqrt(variance_forecast)

        last_date = self.result.conditional_volatility.index[-1]
        future_dates = pd.bdate_range(
            start=last_date + pd.Timedelta(days=1),
            periods=horizon,
        )

        df = pd.DataFrame({
            "Forecast_Variance": variance_forecast,
            "Forecast_Volatility": volatility_forecast,
            "Annualized_Volatility": volatility_forecast * np.sqrt(252),
        }, index=future_dates)
        df.index.name = "Date"
        return df

    def forecast_simulation(self, horizon: int = 10, simulations: int = 1000):
        """Monte Carlo simulation-based forecast with uncertainty bands."""
        forecasts = self.result.forecast(
            horizon=horizon, method="simulation",
            simulations=simulations, reindex=False,
        )
        sim_variance = forecasts.variance.iloc[-1].values
        return np.sqrt(sim_variance)
