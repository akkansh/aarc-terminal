"""Market data, context indicators, and RSS news ingestion with caching and deduplication."""

import datetime
import hashlib
import re
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

import feedparser
import numpy as np
import pandas as pd
import yfinance as yf

from .cache import get_disk_cache


def _clean_series(s: pd.Series) -> pd.Series:
    """Sanitize inf/nan values in a pandas Series."""
    return s.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)


class MarketDataFetcher:
    """Unified data fetcher for equities, market benchmarks, VIX, and news RSS."""

    def __init__(self, cache_dir: str = ".cache", cache_ttl: int = 3600):
        self.cache = get_disk_cache(cache_dir)
        self.cache_ttl = cache_ttl

    def is_indian_ticker(self, ticker: str) -> bool:
        """Check if ticker belongs to Indian markets (NSE/BSE)."""
        clean = ticker.strip().upper()
        return clean.endswith(".NS") or clean.endswith(".BO")

    def get_benchmark_ticker(self, ticker: str) -> Tuple[str, str]:
        """Return (benchmark_index, volatility_index) appropriate for the ticker."""
        if self.is_indian_ticker(ticker):
            return "^NSEI", "^INDIAVIX"
        return "^GSPC", "^VIX"

    def fetch_ohlcv(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "2y",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Download historical OHLCV data.
        Flattens MultiIndex, standardizes column names, removes timezone offsets.
        """
        clean_ticker = ticker.strip().upper()
        cache_key = f"ohlcv:{clean_ticker}:{start_date}:{end_date}:{period}"

        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None and isinstance(cached, pd.DataFrame) and not cached.empty:
                return cached.copy()

        # Download from yfinance
        try:
            if start_date and end_date:
                df = yf.download(clean_ticker, start=start_date, end=end_date, progress=False)
            elif start_date:
                df = yf.download(clean_ticker, start=start_date, progress=False)
            else:
                df = yf.download(clean_ticker, period=period, progress=False)
        except Exception:
            df = pd.DataFrame()

        # Fallback to Ticker.history if download returned empty
        if df is None or df.empty:
            try:
                t_obj = yf.Ticker(clean_ticker)
                if start_date and end_date:
                    df = t_obj.history(start=start_date, end=end_date)
                elif start_date:
                    df = t_obj.history(start=start_date)
                else:
                    df = t_obj.history(period=period)
            except Exception:
                df = pd.DataFrame()

        if df is None or df.empty:
            raise ValueError(f"No price data available for ticker '{clean_ticker}'")

        # Flatten MultiIndex columns immediately
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Standardize column naming
        col_map = {c: str(c).capitalize() for c in df.columns}
        df = df.rename(columns=col_map)

        # Ensure datetime index is timezone-naive
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Ensure index is sorted ascending
        df = df.sort_index()

        # Ensure standard OHLCV columns exist
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col not in df.columns:
                if col == "Volume":
                    df[col] = 0.0
                elif "Close" in df.columns:
                    df[col] = df["Close"]

        # Drop rows with NaN close
        df = df.dropna(subset=["Close"]).copy()

        # Numeric sanitization
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = _clean_series(df[col])

        # Cache result
        if use_cache and not df.empty:
            self.cache.set(cache_key, df, ttl_seconds=self.cache_ttl)

        return df

    def fetch_market_context(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "2y",
    ) -> Dict[str, pd.DataFrame]:
        """Fetch market benchmark index and volatility index (VIX) corresponding to the ticker."""
        benchmark_sym, vix_sym = self.get_benchmark_ticker(ticker)

        try:
            bench_df = self.fetch_ohlcv(benchmark_sym, start_date=start_date, end_date=end_date, period=period)
        except Exception:
            bench_df = pd.DataFrame()

        try:
            vix_df = self.fetch_ohlcv(vix_sym, start_date=start_date, end_date=end_date, period=period)
        except Exception:
            vix_df = pd.DataFrame()

        return {
            "benchmark_symbol": benchmark_sym,
            "benchmark_df": bench_df,
            "vix_symbol": vix_sym,
            "vix_df": vix_df,
        }

    def fetch_news_rss(
        self,
        ticker: str,
        limit: int = 15,
        cutoff_date: Optional[str] = None,
        use_cache: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Scrape top headlines from Google News RSS.
        Deduplicates syndicated stories, extracts normalized dates,
        and enforces cutoff_date filter to strictly prevent look-ahead bias.
        """
        clean_ticker = ticker.split(".")[0].strip() if "." in ticker else ticker.strip()
        cache_key = f"news_rss:{clean_ticker}:{limit}"

        entries_raw = None
        if use_cache:
            entries_raw = self.cache.get(cache_key)

        if entries_raw is None:
            encoded = urllib.parse.quote_plus(clean_ticker)
            is_in = self.is_indian_ticker(ticker)
            hl_gl = "hl=en-IN&gl=IN&ceid=IN:en" if is_in else "hl=en-US&gl=US&ceid=US:en"
            rss_url = f"https://news.google.com/rss/search?q={encoded}+stock+finance&{hl_gl}"

            feed = feedparser.parse(rss_url)
            entries_raw = []
            if hasattr(feed, "entries") and feed.entries:
                for e in feed.entries:
                    title = getattr(e, "title", "")
                    if not title and isinstance(e, dict):
                        title = e.get("title", "")

                    pub = (
                        getattr(e, "published", "")
                        or getattr(e, "pubDate", "")
                        or getattr(e, "updated", "")
                    )
                    if not pub and isinstance(e, dict):
                        pub = e.get("published", "") or e.get("pubDate", "") or e.get("updated", "")

                    link = getattr(e, "link", "")
                    source = ""
                    if hasattr(e, "source") and getattr(e.source, "title", None):
                        source = e.source.title

                    if title:
                        entries_raw.append({
                            "title": str(title).strip(),
                            "published_raw": str(pub).strip() if pub else "",
                            "link": str(link).strip() if link else "",
                            "source": str(source).strip() if source else "",
                        })

            if use_cache and entries_raw:
                self.cache.set(cache_key, entries_raw, ttl_seconds=1800)  # 30 min cache

        # Process, parse timestamps, and deduplicate
        processed: List[Dict[str, Any]] = []
        seen_titles = set()

        cutoff_dt = None
        if cutoff_date:
            try:
                cutoff_dt = pd.to_datetime(cutoff_date).tz_localize(None)
            except Exception:
                cutoff_dt = None

        for item in entries_raw:
            raw_title = item.get("title", "")
            # Deduplication key: lower, strip publisher suffix (e.g. " - The Economic Times")
            normalized_title = re.sub(r"\s+-\s+[^-]+$", "", raw_title).strip().lower()
            title_hash = hashlib.md5(normalized_title.encode("utf-8")).hexdigest()

            if title_hash in seen_titles:
                continue
            seen_titles.add(title_hash)

            # Parse publish datetime
            pub_str = item.get("published_raw", "")
            pub_dt = None
            if pub_str:
                try:
                    pub_dt = pd.to_datetime(pub_str).tz_localize(None)
                except Exception:
                    pub_dt = None

            # Look-ahead prevention: discard news published strictly AFTER cutoff_date
            if cutoff_dt is not None and pub_dt is not None:
                if pub_dt.date() > cutoff_dt.date():
                    continue

            date_str = pub_dt.strftime("%Y-%m-%d") if pub_dt is not None else ""

            processed.append({
                "ticker": ticker.upper(),
                "title": raw_title,
                "clean_title": normalized_title,
                "published_at": pub_str,
                "published_date": date_str,
                "published_dt": pub_dt,
                "source": item.get("source", ""),
                "link": item.get("link", ""),
            })

            if len(processed) >= limit:
                break

        return processed

    def fetch_fundamentals(self, ticker: str) -> Dict[str, Any]:
        """Fetch fundamental valuation ratios and earnings dates if available."""
        clean_ticker = ticker.strip().upper()
        cache_key = f"fundamentals:{clean_ticker}"

        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        data: Dict[str, Any] = {
            "pe_ratio": 0.0,
            "forward_pe": 0.0,
            "price_to_book": 0.0,
            "revenue_growth": 0.0,
            "profit_margins": 0.0,
            "market_cap": 0.0,
            "has_fundamentals": False,
        }

        try:
            t_obj = yf.Ticker(clean_ticker)
            info = getattr(t_obj, "info", {}) or {}
            if info:
                data["pe_ratio"] = float(info.get("trailingPE", 0.0) or 0.0)
                data["forward_pe"] = float(info.get("forwardPE", 0.0) or 0.0)
                data["price_to_book"] = float(info.get("priceToBook", 0.0) or 0.0)
                data["revenue_growth"] = float(info.get("revenueGrowth", 0.0) or 0.0)
                data["profit_margins"] = float(info.get("profitMargins", 0.0) or 0.0)
                data["market_cap"] = float(info.get("marketCap", 0.0) or 0.0)
                data["has_fundamentals"] = True
        except Exception:
            pass

        self.cache.set(cache_key, data, ttl_seconds=86400)  # 24h cache
        return data
