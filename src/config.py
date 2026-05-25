"""Global Configuration"""
from dataclasses import dataclass, field
from typing import List
import os


@dataclass
class Config:
    # Directories
    DATA_DIR: str = "data"
    OUTPUT_DIR: str = "outputs"
    PLOTS_DIR: str = "outputs/plots"
    REPORTS_DIR: str = "outputs/reports"

    # Defaults
    DEFAULT_TICKER: str = "RELIANCE.NS"
    DEFAULT_START: str = "2019-01-01"
    DEFAULT_END: str = "2024-12-31"
    DEFAULT_HORIZON: int = 10

    # Models to compare
    MODEL_CANDIDATES: List[dict] = field(default_factory=lambda: [
        {"name": "ARCH(1)",        "vol": "ARCH",  "p": 1, "q": 0, "dist": "normal"},
        {"name": "GARCH(1,1)",     "vol": "GARCH", "p": 1, "q": 1, "dist": "normal"},
        {"name": "GARCH(1,1)-t",   "vol": "GARCH", "p": 1, "q": 1, "dist": "t"},
        {"name": "EGARCH(1,1)",    "vol": "EGARCH","p": 1, "q": 1, "dist": "normal"},
        {"name": "GJR-GARCH(1,1)", "vol": "GARCH", "p": 1, "o": 1, "q": 1, "dist": "normal"},
    ])

    # Risk
    VAR_CONFIDENCE: List[float] = field(default_factory=lambda: [0.95, 0.99])

    def ensure_dirs(self):
        for d in [self.DATA_DIR, self.OUTPUT_DIR, self.PLOTS_DIR, self.REPORTS_DIR]:
            os.makedirs(d, exist_ok=True)


CONFIG = Config()
CONFIG.ensure_dirs()
