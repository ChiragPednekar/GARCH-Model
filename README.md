# 📉 GARCH Volatility Platform

> A production-ready, modular Python platform for **GARCH-family volatility modelling**, multi-step forecasting, financial risk analytics, and interactive reporting — with a Streamlit dashboard and a FastAPI REST layer.

---

## ✨ Features

| Category | Details |
|---|---|
| **Models** | GARCH, EGARCH, GJR-GARCH, TGARCH |
| **Auto-selection** | AIC/BIC grid search across model types and (p, q) orders |
| **Forecasting** | Multi-step daily & annualised volatility with confidence bands |
| **Risk** | VaR (historical / parametric / t), CVaR, Sharpe, Sortino, Calmar, Max Drawdown |
| **Backtesting** | Rolling 1-step-ahead with Kupiec POF test |
| **Evaluation** | RMSE, MAE, MAPE, QLIKE, R², Mincer-Zarnowitz, Diebold-Mariano |
| **UI** | Streamlit dashboard (5 tabs) with dark-mode Plotly charts |
| **API** | FastAPI REST endpoints with Pydantic validation + auto-docs |
| **Reports** | Self-contained HTML report (optional PDF via weasyprint) |
| **Caching** | Parquet-backed local data cache (avoids re-downloads) |
| **Docker** | Multi-stage Dockerfile for dashboard and API |

---

## 🗂️ Project Structure

```
garch-volatility-platform/
│
├── app/
│   ├── __init__.py
│   ├── dashboard.py          # Streamlit UI (5 tabs)
│   └── api.py               # FastAPI REST endpoints
│
├── src/
│   ├── __init__.py
│   ├── config.py            # Global constants & paths
│   ├── data_loader.py       # yfinance + Parquet cache
│   ├── preprocessing.py     # Returns, ADF, outlier removal
│   ├── models.py            # GARCH family fitters (arch lib)
│   ├── model_selector.py    # Auto grid-search & selection
│   ├── forecasting.py       # Multi-step vol forecasting
│   ├── risk.py              # VaR, CVaR, Sharpe, drawdown …
│   ├── backtesting.py       # Rolling backtest + Kupiec POF
│   ├── visualization.py     # Plotly interactive charts
│   ├── evaluation.py        # RMSE, MAE, QLIKE, MZ, DM
│   └── report.py            # HTML / PDF report builder
│
├── tests/
│   └── test_pipeline.py     # pytest test suite
│
├── data/                    # Parquet cache (git-ignored)
├── outputs/
│   ├── plots/               # Saved Plotly figures
│   └── reports/             # Generated reports
│
├── main.py                  # CLI entry point
├── run_dashboard.sh         # Launch Streamlit
├── run_api.sh               # Launch FastAPI
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone <repo-url>
cd garch-volatility-platform
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. CLI — full pipeline

```bash
# S&P 500, defaults
python main.py

# Apple, EGARCH, auto-select, generate HTML report
python main.py --ticker AAPL --model EGARCH --auto-select --html

# Custom date range, Student-t distribution, skip backtest
python main.py --ticker TSLA --start 2020-01-01 --end 2024-01-01 --dist t --no-backtest
```

### 3. Streamlit dashboard

```bash
bash run_dashboard.sh
# → http://localhost:8501
```

### 4. FastAPI REST server

```bash
bash run_api.sh
# → http://localhost:8000/docs   (Swagger UI)
# → http://localhost:8000/redoc  (ReDoc)
```

### 5. Docker

```bash
# Build
docker build -t garch-platform .

# Dashboard
docker run -p 8501:8501 garch-platform

# API
docker run -p 8000:8000 garch-platform \
  "uvicorn app.api:app --host 0.0.0.0 --port 8000"
```

---

## 🧪 Tests

```bash
pytest tests/ -v --cov=src
```

---

## 📡 API Reference (key endpoints)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/tickers/cached` | List locally cached tickers |
| `POST` | `/analyse` | Full pipeline (data → model → risk → forecast) |
| `POST` | `/forecast` | Volatility forecast only |
| `POST` | `/risk` | Risk metrics only |
| `POST` | `/backtest` | Rolling backtest + Kupiec POF |
| `GET` | `/report/{ticker}` | Download last HTML report |

Full schema available at `/docs` after starting the API.

---

## ⚙️ Configuration

All defaults live in `src/config.py`:

| Variable | Default | Description |
|---|---|---|
| `DEFAULT_TICKER` | `^GSPC` | S&P 500 |
| `DEFAULT_START` | `2015-01-01` | History start date |
| `SUPPORTED_MODELS` | GARCH, EGARCH, GJR-GARCH, TGARCH | Searchable models |
| `DEFAULT_CONFIDENCE_LEVEL` | `0.95` | VaR / CVaR confidence |
| `DEFAULT_FORECAST_HORIZON` | `10` | Trading days ahead |
| `BACKTEST_TRAIN_SIZE` | `504` | ~2 years daily |

---

## 📦 Key Dependencies

| Package | Purpose |
|---|---|
| `arch` | GARCH-family model fitting |
| `yfinance` | Market data download |
| `statsmodels` | ADF test |
| `scipy` | Distributions & statistical tests |
| `plotly` | Interactive charts |
| `streamlit` | Dashboard UI |
| `fastapi` + `uvicorn` | REST API |
| `pyarrow` | Parquet cache |

---

## 📄 License

MIT © GARCH Volatility Platform
