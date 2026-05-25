"""CLI Entry Point — Runs Full Analysis Pipeline"""
import click
import warnings
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src import (
    CONFIG, DataLoader, Preprocessor, ModelSelector,
    VolatilityForecaster, RiskAnalyzer, Backtester,
    Visualizer, ModelEvaluator, ReportGenerator,
)

warnings.filterwarnings("ignore")
console = Console()


@click.command()
@click.option("--ticker",   default=CONFIG.DEFAULT_TICKER,  help="Stock ticker symbol")
@click.option("--start",    default=CONFIG.DEFAULT_START,   help="Start date YYYY-MM-DD")
@click.option("--end",      default=CONFIG.DEFAULT_END,     help="End date YYYY-MM-DD")
@click.option("--horizon",  default=CONFIG.DEFAULT_HORIZON, help="Forecast horizon (days)")
@click.option("--backtest/--no-backtest", default=True,     help="Run VaR backtest")
@click.option("--synthetic", is_flag=True, default=False,
              help="Use synthetic data if Yahoo Finance is unreachable")
def run(ticker, start, end, horizon, backtest, synthetic):
    console.print(Panel.fit(
        f"[bold cyan]GARCH Volatility Intelligence Platform[/bold cyan]\n"
        f"Ticker: [yellow]{ticker}[/yellow] | Period: {start} → {end}",
        border_style="cyan",
    ))

    # ── 1. Load data ──────────────────────────────────────────────────────
    if synthetic:
        console.print("[yellow]⚠️  --synthetic flag set: using generated data[/yellow]")
    with console.status("[cyan]Fetching data..."):
        df = DataLoader(ticker, start, end,
                        allow_synthetic=synthetic).fetch_data()
    console.print(f"✅ Loaded [green]{len(df)}[/green] rows\n")

    # ── 2. Preprocess ─────────────────────────────────────────────────────
    pre = Preprocessor(df)
    df = pre.compute_log_returns()
    returns = df["Return_Pct"]

    stats = pre.summary_stats(returns)
    adf   = pre.adf_test(returns)
    arch  = pre.arch_effect_test(returns)

    t = Table(title="📊 Statistical Tests", show_header=True)
    t.add_column("Test"); t.add_column("Stat"); t.add_column("p-val"); t.add_column("Result")
    t.add_row("ADF (Stationarity)",
              f"{adf['ADF Statistic']:.3f}", f"{adf['p-value']:.4f}",
              "✅ Stationary" if adf["Stationary"] else "❌")
    t.add_row("ARCH-LM",
              f"{arch['LM Statistic']:.3f}", f"{arch['LM p-value']:.4f}",
              "✅ ARCH effects" if arch["ARCH Effects Present"] else "❌")
    console.print(t)

    # ── 3. Model selection ────────────────────────────────────────────────
    console.print("\n[bold]🔍 Comparing Models...[/bold]")
    selector   = ModelSelector(returns)
    comparison = selector.compare()
    console.print(comparison.to_string(index=False))

    result, best_name, best_cfg = selector.best_model(criterion="AIC")
    console.print(f"\n🏆 Best Model: [bold green]{best_name}[/bold green]")

    # ── 4. Conditional volatility & forecast ──────────────────────────────
    cond_vol    = result.conditional_volatility
    forecaster  = VolatilityForecaster(result)
    forecast_df = forecaster.forecast(horizon=horizon)
    console.print(f"\n[bold]🔮 {horizon}-Day Forecast:[/bold]")
    console.print(forecast_df.round(4).to_string())

    # ── 5. Risk analysis ──────────────────────────────────────────────────
    risk        = RiskAnalyzer(returns, cond_vol, forecast_df["Forecast_Volatility"])
    risk_report = risk.risk_report()

    rt = Table(title="⚠️  Risk Report", show_header=True)
    rt.add_column("Metric"); rt.add_column("Value")
    for k, v in risk_report.items():
        rt.add_row(k, f"{v:.4f}")
    console.print(rt)

    # ── 6. Backtest ───────────────────────────────────────────────────────
    if backtest:
        console.print("\n[bold]🎯 Running VaR Backtest...[/bold]")
        bt    = Backtester(returns, window=500)
        bt_df = bt.rolling_var_backtest(
            alpha=0.95, refit_every=20,
            vol_model=best_cfg.get("vol", "GARCH"),
            p=best_cfg.get("p", 1),
            q=best_cfg.get("q", 1),
            dist=best_cfg.get("dist", "normal"),
        )
        kupiec = bt.kupiec_test(bt_df, alpha=0.95)

        kt = Table(title="Kupiec POF Test (95% VaR)", show_header=True)
        kt.add_column("Metric"); kt.add_column("Value")
        for k, v in kupiec.items():
            kt.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
        console.print(kt)
    else:
        bt_df = None

    # ── 7. Visualizations ─────────────────────────────────────────────────
    console.print("\n[bold]🎨 Generating plots...[/bold]")
    viz    = Visualizer()
    charts = {
        "price":       viz.price_chart(df, ticker),
        "returns":     viz.returns_chart(returns, ticker),
        "distribution":viz.returns_distribution(returns, ticker),
        "volatility":  viz.volatility_chart(returns, cond_vol, ticker),
        "forecast":    viz.forecast_chart(cond_vol, forecast_df, ticker),
        "comparison":  viz.model_comparison_chart(comparison),
    }
    if bt_df is not None:
        charts["backtest"] = viz.var_backtest_chart(bt_df, ticker, 0.95)

    for name, fig in charts.items():
        viz._save(fig, f"{ticker}_{name}.html")

    # ── 8. Report ─────────────────────────────────────────────────────────
    evaluator   = ModelEvaluator(result)
    key_metrics = {
        "Log-Likelihood": evaluator.metrics()["Log-Likelihood"],
        "AIC":            evaluator.metrics()["AIC"],
        "BIC":            evaluator.metrics()["BIC"],
        "QLIKE Loss":     evaluator.qlike_loss(returns),
        "Current Vol (%)":float(cond_vol.iloc[-1]),
    }
    report_path = ReportGenerator().generate(
        ticker=ticker, start_date=start, end_date=end,
        best_model=best_name, summary_stats=stats,
        adf_result=adf, arch_result=arch, comparison_df=comparison,
        risk_report=risk_report, forecast_df=forecast_df,
        key_metrics=key_metrics,
    )

    console.print(f"\n✅ [bold green]HTML Report:[/bold green] {report_path}")
    console.print(f"✅ [bold green]Charts saved in:[/bold green] {CONFIG.PLOTS_DIR}/")
    console.print(Panel.fit("[bold green]🎉 Pipeline Complete![/bold green]",
                            border_style="green"))


if __name__ == "__main__":
    run()
