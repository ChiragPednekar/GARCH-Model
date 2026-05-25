"""FastAPI REST API for GARCH Platform"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import warnings
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import (
    DataLoader, Preprocessor, ModelSelector,
    VolatilityForecaster, RiskAnalyzer,
)

warnings.filterwarnings("ignore")

app = FastAPI(
    title="GARCH Volatility API",
    description="REST API for volatility modeling, forecasting, and risk metrics",
    version="1.0.0",
)


# ── Request schema ───────────────────────────────────────────────────────────
class AnalysisRequest(BaseModel):
    ticker:         str   = Field(..., example="RELIANCE.NS")
    start_date:     str   = Field(..., example="2019-01-01")
    end_date:       str   = Field(..., example="2024-12-31")
    horizon:        int   = Field(10,   ge=1,  le=60)
    var_confidence: float = Field(0.95, ge=0.8, le=0.99)


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "GARCH Volatility API",
        "version": "1.0.0",
        "endpoints": ["/health", "/analyze", "/forecast", "/risk", "/docs"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(req: AnalysisRequest):
    try:
        df = DataLoader(req.ticker, req.start_date, req.end_date).fetch_data()
        pre = Preprocessor(df)
        df  = pre.compute_log_returns()
        returns = df["Return_Pct"]

        selector   = ModelSelector(returns)
        comparison = selector.compare()
        result, best_name, _ = selector.best_model("AIC")
        cond_vol = result.conditional_volatility

        forecast_df = VolatilityForecaster(result).forecast(horizon=req.horizon)
        risk        = RiskAnalyzer(returns, cond_vol, forecast_df["Forecast_Volatility"])

        return {
            "ticker":           req.ticker,
            "observations":     len(returns),
            "best_model":       best_name,
            "statistics":       pre.summary_stats(returns),
            "adf_test":         pre.adf_test(returns),
            "arch_test":        pre.arch_effect_test(returns),
            "model_comparison": comparison.to_dict(orient="records"),
            "current_volatility": float(cond_vol.iloc[-1]),
            "forecast":         forecast_df.reset_index().to_dict(orient="records"),
            "risk_metrics":     risk.risk_report(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/forecast")
def forecast_only(req: AnalysisRequest):
    try:
        df = DataLoader(req.ticker, req.start_date, req.end_date).fetch_data()
        df = Preprocessor(df).compute_log_returns()
        selector = ModelSelector(df["Return_Pct"])
        selector.compare()
        result, best_name, _ = selector.best_model("AIC")
        forecast_df = VolatilityForecaster(result).forecast(horizon=req.horizon)
        return {
            "ticker":     req.ticker,
            "best_model": best_name,
            "forecast":   forecast_df.reset_index().to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/risk")
def risk_only(req: AnalysisRequest):
    try:
        df = DataLoader(req.ticker, req.start_date, req.end_date).fetch_data()
        df = Preprocessor(df).compute_log_returns()
        returns = df["Return_Pct"]
        selector = ModelSelector(returns)
        selector.compare()
        result, _, _ = selector.best_model("AIC")
        risk = RiskAnalyzer(returns, result.conditional_volatility)
        return {
            "ticker":      req.ticker,
            "var_95":      risk.historical_var(0.95),
            "cvar_95":     risk.historical_cvar(0.95),
            "var_99":      risk.historical_var(0.99),
            "cvar_99":     risk.historical_cvar(0.99),
            "full_report": risk.risk_report(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
