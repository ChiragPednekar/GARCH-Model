"""Data Loader: Yahoo Finance with crumb auth, retry, and synthetic fallback."""
from __future__ import annotations
import os
import time
import json
import random
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_YF_COOKIE_URL = "https://finance.yahoo.com"
_YF_CRUMB_URL  = "https://query1.finance.yahoo.com/v1/test/getcrumb"
_CHART_URL     = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
_CHART_URL2    = "https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://finance.yahoo.com/",
}


# ---------------------------------------------------------------------------
# Session + crumb
# ---------------------------------------------------------------------------

def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(_HEADERS)
    return s


def _get_crumb(session: requests.Session) -> str | None:
    try:
        session.get(_YF_COOKIE_URL, timeout=10)
        time.sleep(random.uniform(0.5, 1.5))
        r = session.get(_YF_CRUMB_URL, timeout=10)
        if r.status_code == 200 and r.text.strip():
            return r.text.strip()
    except Exception:
        pass
    return None


def _chart_api(ticker: str, start: str, end: str,
               session: requests.Session, crumb: str | None,
               base_url: str = _CHART_URL) -> pd.DataFrame | None:
    """Call Yahoo v8 chart endpoint directly."""
    params: dict = {
        "period1":  int(pd.Timestamp(start).timestamp()),
        "period2":  int(pd.Timestamp(end).timestamp()) + 86400,
        "interval": "1d",
        "events":   "history",
        "includeAdjustedClose": "true",
    }
    if crumb:
        params["crumb"] = crumb

    url = base_url.format(ticker=ticker)
    try:
        r = session.get(url, params=params, timeout=15)
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", 8))
            print(f"[WARN] Rate limited — waiting {wait}s …")
            time.sleep(wait)
            r = session.get(url, params=params, timeout=15)
        if r.status_code != 200 or not r.text.strip():
            return None
        data   = r.json()
        result = data.get("chart", {}).get("result") or []
        if not result:
            return None
        res    = result[0]
        q      = res["indicators"]["quote"][0]
        adj    = (res["indicators"].get("adjclose") or [{}])[0].get(
                     "adjclose", q["close"])
        index  = pd.to_datetime(res["timestamp"], unit="s", utc=True
                                ).tz_convert("UTC").tz_localize(None).normalize()
        df = pd.DataFrame(
            {"Open": q["open"], "High": q["high"],
             "Low":  q["low"],  "Close": adj, "Volume": q["volume"]},
            index=index,
        )
        df.index.name = "Date"
        return df.dropna(how="all")
    except Exception as e:
        print(f"[WARN] Chart API ({base_url[:30]}…) failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Synthetic data generator (fallback for offline / rate-limited environments)
# ---------------------------------------------------------------------------

def _synthetic_ohlcv(ticker: str, start: str, end: str,
                     seed: int | None = None) -> pd.DataFrame:
    """Generate realistic-looking synthetic OHLCV data for development."""
    rng   = np.random.default_rng(seed or abs(hash(ticker)) % 2**31)
    index = pd.bdate_range(start, end)
    n     = len(index)

    # GBM price path
    mu, sigma = 0.0003, 0.015
    log_ret   = rng.normal(mu, sigma, n)
    close     = 1000.0 * np.exp(np.cumsum(log_ret))

    noise = rng.uniform(0.002, 0.012, n)
    high  = close * (1 + noise)
    low   = close * (1 - noise)
    open_ = np.roll(close, 1); open_[0] = close[0]
    vol   = rng.integers(500_000, 5_000_000, n)

    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=index,
    )
    df.index.name = "Date"
    return df


# ---------------------------------------------------------------------------
# Public DataLoader class
# ---------------------------------------------------------------------------

class DataLoader:
    def __init__(self, ticker: str, start_date: str, end_date: str,
                 data_dir: str = "data", allow_synthetic: bool = False):
        """
        Parameters
        ----------
        allow_synthetic : bool
            If True, fall back to synthetic data when Yahoo Finance is
            unreachable (useful for demos / offline dev).
            Set via: DataLoader(..., allow_synthetic=True)
        """
        self.ticker          = ticker
        self.start_date      = start_date
        self.end_date        = end_date
        self.data_dir        = data_dir
        self.allow_synthetic = allow_synthetic
        os.makedirs(self.data_dir, exist_ok=True)

    def _cache_path(self) -> str:
        safe = self.ticker.replace("^", "IDX_").replace("/", "_")
        return os.path.join(self.data_dir,
                            f"{safe}_{self.start_date}_{self.end_date}.csv")

    # ------------------------------------------------------------------
    def fetch_data(self, use_cache: bool = True, save: bool = True) -> pd.DataFrame:
        path = self._cache_path()

        # ── 1. Cache ───────────────────────────────────────────────────────
        if use_cache and os.path.exists(path):
            age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))
            if age < timedelta(hours=12):
                print(f"[CACHE] Loading {self.ticker} from cache")
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                if not df.empty:
                    return df

        print(f"[FETCH] Downloading {self.ticker} ({self.start_date} → {self.end_date})")

        # ── 2. Direct Yahoo API (query1) ───────────────────────────────────
        session = _make_session()
        crumb   = _get_crumb(session)
        df      = _chart_api(self.ticker, self.start_date, self.end_date,
                              session, crumb, _CHART_URL)

        # ── 3. Direct Yahoo API (query2 fallback) ──────────────────────────
        if df is None or df.empty:
            print("[RETRY] Trying query2.finance.yahoo.com …")
            time.sleep(2)
            df = _chart_api(self.ticker, self.start_date, self.end_date,
                             session, crumb, _CHART_URL2)

        # ── 4. yfinance library fallback ───────────────────────────────────
        if df is None or df.empty:
            print("[RETRY] Falling back to yfinance library …")
            time.sleep(2)
            try:
                raw = yf.download(self.ticker, start=self.start_date,
                                  end=self.end_date, progress=False,
                                  auto_adjust=True)
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                df = raw if not raw.empty else None
            except Exception as e:
                print(f"[WARN] yfinance failed: {e}")
                df = None

        # ── 5. Synthetic fallback ──────────────────────────────────────────
        if (df is None or df.empty) and self.allow_synthetic:
            print(
                f"[SYNTHETIC] ⚠️  Yahoo Finance unreachable. "
                f"Generating synthetic data for {self.ticker} (demo mode)."
            )
            df = _synthetic_ohlcv(self.ticker, self.start_date, self.end_date)

        # ── 6. Hard failure ────────────────────────────────────────────────
        if df is None or df.empty:
            raise ValueError(
                f"❌ Could not download data for '{self.ticker}'.\n\n"
                "Yahoo Finance appears to be rate-limiting your IP.\n"
                "Options:\n"
                "  1. Wait 1–2 minutes and retry\n"
                "  2. Use a VPN / different network\n"
                "  3. Run with allow_synthetic=True for offline demo:\n"
                "       DataLoader(ticker, start, end, allow_synthetic=True)\n"
                "     or via CLI: python main.py --synthetic\n"
            )

        # ── Keep OHLCV ─────────────────────────────────────────────────────
        cols = [c for c in ["Open", "High", "Low", "Close", "Volume"]
                if c in df.columns]
        df = df[cols].copy().dropna(how="all")

        if save:
            df.to_csv(path)
            print(f"[CACHE] Saved → {path}")

        print(f"[OK] {len(df)} rows loaded for {self.ticker}")
        return df

    # ------------------------------------------------------------------
    @staticmethod
    def validate_ticker(ticker: str) -> bool:
        try:
            s, crumb = _make_session(), _get_crumb(_make_session())
            end   = datetime.now().strftime("%Y-%m-%d")
            start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            df    = _chart_api(ticker, start, end, s, crumb)
            return df is not None and not df.empty
        except Exception:
            return False
