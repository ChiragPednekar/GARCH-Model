"""Streamlit Interactive Dashboard"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import warnings

from src import (
    DataLoader, Preprocessor, ModelSelector,
    VolatilityForecaster, RiskAnalyzer, Backtester,
    Visualizer, ModelEvaluator,
)

warnings.filterwarnings("ignore")
st.set_page_config(
    page_title="GARCH Volatility Platform",
    layout="wide",
    page_icon="📈",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Configuration")
ticker     = st.sidebar.text_input("Ticker", value="RELIANCE.NS",
                                   help="e.g. RELIANCE.NS, TCS.NS, ^NSEI, AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2019-01-01"))
end_date   = st.sidebar.date_input("End Date",   value=pd.to_datetime("2024-12-31"))
horizon    = st.sidebar.slider("Forecast Horizon (days)", 5, 60, 10)
run_backtest = st.sidebar.checkbox("Run VaR Backtest", value=True)
var_alpha  = st.sidebar.select_slider(
    "VaR Confidence", options=[0.90, 0.95, 0.99], value=0.95
)
run_btn = st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📈 GARCH Volatility Intelligence Platform")
st.caption("Model • Forecast • Quantify Risk — powered by GARCH family models")

if not run_btn:
    st.info("👈 Configure parameters and click **Run Analysis**")
    with st.expander("ℹ️ About this Platform"):
        st.markdown("""
        This platform provides a **complete volatility modeling workflow**:

        - 📊 **Data ingestion** from Yahoo Finance
        - 🧪 **Statistical testing** (ADF, ARCH-LM)
        - 🏆 **Auto model selection** across ARCH / GARCH / EGARCH / GJR-GARCH
        - 🔮 **Multi-step forecasting** with simulation
        - ⚠️ **Risk metrics**: VaR, CVaR (historical + parametric)
        - 🎯 **Rolling backtest** with Kupiec POF test
        """)
    st.stop()

# ── Pipeline ─────────────────────────────────────────────────────────────────
try:
    with st.spinner("Fetching data..."):
        df = DataLoader(ticker, str(start_date), str(end_date)).fetch_data()
    st.success(f"✅ Loaded {len(df)} rows for **{ticker}**")

    pre     = Preprocessor(df)
    df      = pre.compute_log_returns()
    returns = df["Return_Pct"]
    stats   = pre.summary_stats(returns)
    adf     = pre.adf_test(returns)
    arch    = pre.arch_effect_test(returns)

    # ── KPIs ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observations", stats["Observations"])
    c2.metric("Mean Return (%)", f"{stats['Mean']:.4f}")
    c3.metric("Std Dev (%)", f"{stats['Std Dev']:.4f}")
    c4.metric("Kurtosis", f"{stats['Kurtosis']:.2f}",
              "Fat tails" if stats["Kurtosis"] > 3 else "Normal")

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["📊 Data", "🧪 Tests", "🏆 Models", "🔮 Forecast", "⚠️ Risk", "🎯 Backtest"]
    )

    viz = Visualizer()

    with tab1:
        st.subheader("Price History")
        st.plotly_chart(viz.price_chart(df, ticker), use_container_width=True)
        st.subheader("Log Returns")
        st.plotly_chart(viz.returns_chart(returns, ticker), use_container_width=True)
        st.plotly_chart(viz.returns_distribution(returns, ticker), use_container_width=True)

    with tab2:
        st.subheader("Stationarity & Heteroskedasticity Tests")
        tc1, tc2 = st.columns(2)
        with tc1:
            st.markdown("### ADF Test")
            st.metric("ADF Statistic", f"{adf['ADF Statistic']:.4f}")
            st.metric("p-value", f"{adf['p-value']:.6f}")
            st.success("✅ Stationary") if adf["Stationary"] else st.error("❌ Non-stationary")
        with tc2:
            st.markdown("### ARCH-LM Test")
            st.metric("LM Statistic", f"{arch['LM Statistic']:.4f}")
            st.metric("p-value", f"{arch['LM p-value']:.6f}")
            if arch["ARCH Effects Present"]:
                st.success("✅ ARCH effects present")
            else:
                st.warning("❌ No ARCH effects detected")

    # Model selection
    with st.spinner("Fitting GARCH models..."):
        selector   = ModelSelector(returns)
        comparison = selector.compare()
        result, best_name, best_cfg = selector.best_model("AIC")

    with tab3:
        st.subheader("Model Comparison")
        st.dataframe(comparison, use_container_width=True)
        st.plotly_chart(viz.model_comparison_chart(comparison), use_container_width=True)
        st.success(f"🏆 Best Model: **{best_name}**")
        st.code(str(result.summary()))

    # Forecast
    cond_vol    = result.conditional_volatility
    forecaster  = VolatilityForecaster(result)
    forecast_df = forecaster.forecast(horizon=horizon)

    with tab4:
        st.subheader("Conditional Volatility")
        st.plotly_chart(viz.volatility_chart(returns, cond_vol, ticker),
                        use_container_width=True)
        st.subheader(f"{horizon}-Day Forecast")
        st.plotly_chart(viz.forecast_chart(cond_vol, forecast_df, ticker),
                        use_container_width=True)
        st.dataframe(forecast_df.round(4), use_container_width=True)
        st.download_button(
            "💾 Download Forecast CSV",
            forecast_df.to_csv(),
            f"{ticker}_forecast.csv",
            "text/csv",
        )

    with tab5:
        risk   = RiskAnalyzer(returns, cond_vol, forecast_df["Forecast_Volatility"])
        report = risk.risk_report()
        st.subheader("Risk Metrics")
        rc1, rc2 = st.columns(2)
        with rc1:
            st.metric(f"Historical VaR ({int(var_alpha*100)}%)",
                      f"{risk.historical_var(var_alpha):.4f}%")
            st.metric(f"Historical CVaR ({int(var_alpha*100)}%)",
                      f"{risk.historical_cvar(var_alpha):.4f}%")
        with rc2:
            st.metric(f"Parametric VaR ({int(var_alpha*100)}%)",
                      f"{risk.parametric_var(var_alpha):.4f}%")
            st.metric(f"Parametric CVaR ({int(var_alpha*100)}%)",
                      f"{risk.parametric_cvar(var_alpha):.4f}%")
        st.markdown("### Full Risk Report")
        st.dataframe(
            pd.DataFrame(report.items(), columns=["Metric", "Value"]),
            use_container_width=True,
        )

    with tab6:
        if run_backtest:
            with st.spinner("Running rolling VaR backtest (this may take ~30 s)..."):
                bt    = Backtester(returns, window=500)
                bt_df = bt.rolling_var_backtest(
                    alpha=var_alpha, refit_every=20,
                    vol_model=best_cfg.get("vol", "GARCH"),
                    p=best_cfg.get("p", 1),
                    q=best_cfg.get("q", 1),
                    dist=best_cfg.get("dist", "normal"),
                )
                kupiec = bt.kupiec_test(bt_df, alpha=var_alpha)

            st.subheader(f"Rolling VaR Backtest ({int(var_alpha*100)}%)")
            st.plotly_chart(viz.var_backtest_chart(bt_df, ticker, var_alpha),
                            use_container_width=True)

            kc1, kc2, kc3 = st.columns(3)
            kc1.metric("Total Days",   kupiec["N"])
            kc2.metric("Breaches",     kupiec["Breaches"])
            kc3.metric("Breach Rate",  f"{kupiec['Breach Rate']:.2%}")

            if kupiec.get("Model Adequate"):
                st.success(f"✅ Kupiec POF test PASSED (p={kupiec['p_value']:.4f})")
            else:
                st.warning(f"⚠️ Model may be miscalibrated (p={kupiec.get('p_value')})")
        else:
            st.info("Enable backtest in the sidebar to run rolling validation.")

    st.balloons()

except Exception as e:
    st.error(f"❌ Error: {e}")
    st.exception(e)
