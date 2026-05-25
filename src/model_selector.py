"""Automatic Model Selection via AIC/BIC"""
import pandas as pd
from arch import arch_model
from src.config import CONFIG


class ModelSelector:
    def __init__(self, returns: pd.Series):
        self.returns = returns.dropna()
        self.results = []

    def compare(self, candidates=None) -> pd.DataFrame:
        candidates = candidates or CONFIG.MODEL_CANDIDATES
        self.results = []

        for c in candidates:
            try:
                kwargs = {k: v for k, v in c.items() if k != "name"}
                model = arch_model(self.returns, mean="Constant", **kwargs)
                res = model.fit(disp="off", show_warning=False)
                self.results.append({
                    "Model": c["name"],
                    "LogLik": res.loglikelihood,
                    "AIC": res.aic,
                    "BIC": res.bic,
                    "Converged": res.convergence_flag == 0,
                    "_result": res,
                    "_config": c,
                })
            except Exception as e:
                self.results.append({
                    "Model": c["name"], "LogLik": None, "AIC": None, "BIC": None,
                    "Converged": False, "_result": None, "_config": c, "Error": str(e),
                })

        df = pd.DataFrame(self.results)
        df_display = df.drop(columns=["_result", "_config"], errors="ignore")
        df_display = df_display.sort_values("AIC", na_position="last").reset_index(drop=True)
        return df_display

    def best_model(self, criterion: str = "AIC"):
        if not self.results:
            self.compare()
        valid = [r for r in self.results if r["AIC"] is not None and r["Converged"]]
        if not valid:
            raise RuntimeError("No model converged successfully")
        best = min(valid, key=lambda r: r[criterion])
        return best["_result"], best["Model"], best["_config"]
