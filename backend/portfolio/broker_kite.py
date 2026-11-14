"""
Zerodha Kite Connect API Integration & Broker Gateway for AARC Quant Terminal.
Provides OAuth token generation, automated session exchange, session persistence,
and live holdings retrieval with enterprise error handling and sample demo telemetry.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger("aarc.portfolio.kite")

try:
    from kiteconnect import KiteConnect
    from kiteconnect import exceptions as kite_exceptions
    KITE_AVAILABLE = True
except ImportError:
    KiteConnect = None
    kite_exceptions = None
    KITE_AVAILABLE = False

SESSION_FILE = Path(__file__).resolve().parent.parent / "broker_session.json"


class PortfolioHolding(BaseModel):
    ticker: str = Field(description="Normalized ticker symbol (e.g. RELIANCE.NS)")
    tradingsymbol: str = Field(description="Broker trading symbol (e.g. RELIANCE)")
    exchange: str = Field(default="NSE", description="Exchange: NSE / BSE")
    quantity: int = Field(description="Total held quantity")
    average_price: float = Field(description="Average purchase price")
    last_price: float = Field(description="Current market price")
    value: float = Field(description="Current total market value (quantity * last_price)")
    invested_value: float = Field(description="Total invested amount (quantity * average_price)")
    pnl: float = Field(description="Total unrealized profit/loss in currency")
    pnl_percentage: float = Field(description="Unrealized return percentage")
    day_change_percentage: float = Field(default=0.0, description="1-day price change percentage")
    isin: Optional[str] = Field(default=None, description="ISIN security code")


class BrokerAuthRequest(BaseModel):
    api_key: Optional[str] = Field(default=None, description="Zerodha Kite API Key")
    request_token: Optional[str] = Field(default=None, description="OAuth Request Token from Kite redirect")
    api_secret: Optional[str] = Field(default=None, description="Zerodha Kite API Secret")
    access_token: Optional[str] = Field(default=None, description="Existing valid Kite access token")


class BrokerCredentialsRequest(BaseModel):
    api_key: str = Field(description="Zerodha Kite API Key")
    api_secret: str = Field(description="Zerodha Kite API Secret")
    redirect_uri: Optional[str] = Field(default="http://localhost:3000/portfolio", description="Configured Redirect URI in Kite dashboard")


class BrokerSessionResponse(BaseModel):
    status: str = Field(description="'connected' or 'error'")
    is_authenticated: bool = Field(default=False, description="True if valid live session exists")
    access_token: Optional[str] = Field(default=None, description="Active Kite access token")
    user_id: Optional[str] = Field(default=None, description="Zerodha user ID")
    user_name: Optional[str] = Field(default=None, description="Account holder name")
    broker: str = Field(default="Zerodha Kite Connect", description="Broker identity")
    login_url: Optional[str] = Field(default=None, description="OAuth login URL")
    message: Optional[str] = Field(default=None, description="Status or error message")
    has_credentials: bool = Field(default=False, description="True if API Key and Secret are configured")
    api_key_masked: Optional[str] = Field(default=None, description="Masked API key for UI display")


class KiteBrokerClient:
    """
    Client for interacting with Zerodha Kite Connect API.
    Handles login URL generation, token exchange, and live equity holdings retrieval.
    """

    def __init__(self, api_key: Optional[str] = None, access_token: Optional[str] = None):
        self.api_key = api_key or ""
        self.access_token = access_token
        self._kite: Optional[Any] = None

        if KITE_AVAILABLE and self.api_key:
            self._kite = KiteConnect(api_key=self.api_key)
            if self.access_token:
                self._kite.set_access_token(self.access_token)

    def get_login_url(self) -> str:
        """Generate the Kite Connect OAuth 2.0 login URL."""
        if not KITE_AVAILABLE:
            raise RuntimeError("kiteconnect package is not installed. Run 'pip install kiteconnect'.")
        if not self.api_key:
            raise ValueError("api_key is required to generate a Kite Connect login URL.")
        kite = KiteConnect(api_key=self.api_key)
        return kite.login_url()

    def generate_session(self, request_token: str, api_secret: str) -> Dict[str, Any]:
        """
        Exchange an OAuth request token and API secret for an access token.
        """
        if not KITE_AVAILABLE:
            raise RuntimeError("kiteconnect package is not installed.")
        if not self.api_key or not api_secret or not request_token:
            raise ValueError("api_key, api_secret, and request_token are required for session generation.")

        kite = KiteConnect(api_key=self.api_key)
        try:
            session_data = kite.generate_session(request_token=request_token, api_secret=api_secret)
            self.access_token = session_data.get("access_token")
            self._kite = kite
            self._kite.set_access_token(self.access_token)
            return session_data
        except Exception as exc:
            logger.error("Kite session generation failed: %s", exc)
            self._handle_kite_exception(exc)
            raise

    def fetch_holdings(self) -> List[PortfolioHolding]:
        """
        Fetch long equity holdings from Kite Connect and normalize them into PortfolioHolding objects.
        """
        if not KITE_AVAILABLE:
            raise RuntimeError("kiteconnect package is not installed.")
        if not self._kite or not self.access_token:
            raise ValueError("Broker client is not authenticated. Please supply a valid api_key and access_token.")

        try:
            raw_holdings = self._kite.holdings()
            if not raw_holdings:
                logger.info("Kite returned empty holdings list.")
                return []

            normalized: List[PortfolioHolding] = []
            for item in raw_holdings:
                tsym = item.get("tradingsymbol", "").strip()
                exchange = item.get("exchange", "NSE").strip().upper()
                ticker = f"{tsym}.NS" if exchange == "NSE" and not tsym.endswith((".NS", ".BO")) else tsym

                qty = int(item.get("quantity", 0))
                if qty <= 0:
                    continue

                avg_price = float(item.get("average_price", 0.0))
                last_price = float(item.get("last_price", avg_price))
                curr_val = round(qty * last_price, 2)
                inv_val = round(qty * avg_price, 2)
                pnl = float(item.get("pnl", curr_val - inv_val))
                pnl_pct = round(((curr_val - inv_val) / inv_val * 100.0) if inv_val > 0 else 0.0, 2)
                day_change = float(item.get("day_change_percentage", 0.0))

                normalized.append(
                    PortfolioHolding(
                        ticker=ticker,
                        tradingsymbol=tsym,
                        exchange=exchange,
                        quantity=qty,
                        average_price=round(avg_price, 2),
                        last_price=round(last_price, 2),
                        value=curr_val,
                        invested_value=inv_val,
                        pnl=round(pnl, 2),
                        pnl_percentage=pnl_pct,
                        day_change_percentage=round(day_change, 2),
                        isin=item.get("isin"),
                    )
                )
            return normalized

        except Exception as exc:
            logger.error("Kite holdings fetch failed: %s", exc)
            self._handle_kite_exception(exc)
            raise

    @staticmethod
    def _handle_kite_exception(exc: Exception) -> None:
        """Translate Kite API exceptions into actionable errors."""
        if kite_exceptions:
            if isinstance(exc, kite_exceptions.TokenException):
                raise PermissionError("Zerodha Kite access token has expired or is invalid. Please re-authenticate.") from exc
            if isinstance(exc, kite_exceptions.PermissionException):
                raise PermissionError("Invalid API key or insufficient permissions for Zerodha Kite API.") from exc
            if isinstance(exc, kite_exceptions.NetworkException):
                raise ConnectionError("Network failure communicating with Zerodha Kite servers.") from exc
        raise RuntimeError(f"Zerodha Kite API error: {str(exc)}") from exc

    @staticmethod
    def get_demo_holdings() -> List[PortfolioHolding]:
        """
        Generate a realistic demo equity portfolio showcasing an over-concentrated
        asset allocation (Financials & Energy overweight > 20%, 0% FMCG/Healthcare)
        to demonstrate the Least-Risk Optimization algorithm.
        """
        demo_data = [
            {
                "ticker": "RELIANCE.NS",
                "tradingsymbol": "RELIANCE",
                "exchange": "NSE",
                "quantity": 100,
                "average_price": 1280.0,
                "last_price": 1339.7,
                "day_change_percentage": 0.85,
                "isin": "INE002A01018",
            },
            {
                "ticker": "HDFCBANK.NS",
                "tradingsymbol": "HDFCBANK",
                "exchange": "NSE",
                "quantity": 80,
                "average_price": 1640.0,
                "last_price": 1720.5,
                "day_change_percentage": -0.32,
                "isin": "INE040A01034",
            },
            {
                "ticker": "ICICIBANK.NS",
                "tradingsymbol": "ICICIBANK",
                "exchange": "NSE",
                "quantity": 65,
                "average_price": 1180.0,
                "last_price": 1254.0,
                "day_change_percentage": 1.15,
                "isin": "INE090A01021",
            },
            {
                "ticker": "TCS.NS",
                "tradingsymbol": "TCS",
                "exchange": "NSE",
                "quantity": 25,
                "average_price": 4100.0,
                "last_price": 3950.0,
                "day_change_percentage": -1.05,
                "isin": "INE467B01029",
            },
            {
                "ticker": "INFY.NS",
                "tradingsymbol": "INFY",
                "exchange": "NSE",
                "quantity": 40,
                "average_price": 1620.0,
                "last_price": 1680.0,
                "day_change_percentage": 0.45,
                "isin": "INE009A01021",
            },
            {
                "ticker": "ETERNAL.NS",
                "tradingsymbol": "ETERNAL",
                "exchange": "NSE",
                "quantity": 150,
                "average_price": 240.0,
                "last_price": 260.0,
                "day_change_percentage": 2.10,
                "isin": "INE758T01015",
            },
            {
                "ticker": "TATAMOTORS.NS",
                "tradingsymbol": "TATAMOTORS",
                "exchange": "NSE",
                "quantity": 30,
                "average_price": 610.0,
                "last_price": 642.0,
                "day_change_percentage": 0.70,
                "isin": "INE155A01022",
            },
        ]

        holdings: List[PortfolioHolding] = []
        for item in demo_data:
            qty = item["quantity"]
            avg = item["average_price"]
            last = item["last_price"]
            curr_val = round(qty * last, 2)
            inv_val = round(qty * avg, 2)
            pnl = round(curr_val - inv_val, 2)
            pnl_pct = round(((curr_val - inv_val) / inv_val * 100.0) if inv_val > 0 else 0.0, 2)

            holdings.append(
                PortfolioHolding(
                    ticker=item["ticker"],
                    tradingsymbol=item["tradingsymbol"],
                    exchange=item["exchange"],
                    quantity=qty,
                    average_price=avg,
                    last_price=last,
                    value=curr_val,
                    invested_value=inv_val,
                    pnl=pnl,
                    pnl_percentage=pnl_pct,
                    day_change_percentage=item["day_change_percentage"],
                    isin=item["isin"],
                )
            )
        return holdings


class BrokerSessionManager:
    """
    Singleton session manager for Zerodha Kite Connect.
    Maintains API credentials, active OAuth tokens, and cached client instances.
    Enables seamless 1-click authentication and automated callback processing.
    """

    _instance: Optional["BrokerSessionManager"] = None

    def __new__(cls) -> "BrokerSessionManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_store()
        return cls._instance

    def _init_store(self) -> None:
        self.api_key: str = os.getenv("KITE_API_KEY", "")
        self.api_secret: str = os.getenv("KITE_API_SECRET", "")
        self.access_token: Optional[str] = os.getenv("KITE_ACCESS_TOKEN", None)
        self.user_id: Optional[str] = None
        self.user_name: Optional[str] = None
        self.login_time: Optional[str] = None
        self._client: Optional[KiteBrokerClient] = None

        # Load persisted session if available
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        if SESSION_FILE.exists():
            try:
                with open(SESSION_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if not self.api_key and data.get("api_key"):
                        self.api_key = data["api_key"]
                    if not self.api_secret and data.get("api_secret"):
                        self.api_secret = data["api_secret"]
                    if not self.access_token and data.get("access_token"):
                        self.access_token = data["access_token"]
                    self.user_id = data.get("user_id", self.user_id)
                    self.user_name = data.get("user_name", self.user_name)
                    self.login_time = data.get("login_time", self.login_time)
            except Exception as e:
                logger.warning("Failed to load broker session file: %s", e)

        if self.api_key and self.access_token:
            self._client = KiteBrokerClient(api_key=self.api_key, access_token=self.access_token)

    def _save_to_disk(self) -> None:
        try:
            data = {
                "api_key": self.api_key,
                "api_secret": self.api_secret,
                "access_token": self.access_token,
                "user_id": self.user_id,
                "user_name": self.user_name,
                "login_time": self.login_time,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            with open(SESSION_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error("Failed to persist broker session: %s", e)

    def set_credentials(self, api_key: str, api_secret: str) -> None:
        self.api_key = api_key.strip()
        self.api_secret = api_secret.strip()
        self._save_to_disk()

    def get_login_url(self, api_key: Optional[str] = None) -> str:
        effective_key = api_key or self.api_key
        if not effective_key:
            raise ValueError("Kite API Key is not configured. Please supply an API Key.")
        client = KiteBrokerClient(api_key=effective_key)
        return client.get_login_url()

    def exchange_request_token(
        self,
        request_token: str,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ) -> Dict[str, Any]:
        effective_key = api_key or self.api_key
        effective_secret = api_secret or self.api_secret

        if not effective_key or not effective_secret:
            raise ValueError(
                "Zerodha Kite API Key and API Secret are required for token exchange. "
                "Configure them in settings or environment."
            )

        client = KiteBrokerClient(api_key=effective_key)
        session_data = client.generate_session(request_token=request_token, api_secret=effective_secret)

        self.api_key = effective_key
        self.api_secret = effective_secret
        self.access_token = session_data.get("access_token")
        self.user_id = session_data.get("user_id")
        self.user_name = session_data.get("user_name")
        self.login_time = datetime.now(timezone.utc).isoformat()
        self._client = client

        self._save_to_disk()
        return session_data

    def get_client(self) -> Optional[KiteBrokerClient]:
        if self._client and self.access_token:
            return self._client
        if self.api_key and self.access_token:
            self._client = KiteBrokerClient(api_key=self.api_key, access_token=self.access_token)
            return self._client
        return None

    def is_authenticated(self) -> bool:
        return bool(self.api_key and self.access_token)

    def clear_session(self) -> None:
        self.access_token = None
        self.user_id = None
        self.user_name = None
        self.login_time = None
        self._client = None
        self._save_to_disk()

    def get_status(self) -> BrokerSessionResponse:
        masked_key = (
            f"{self.api_key[:4]}••••{self.api_key[-4:]}"
            if self.api_key and len(self.api_key) > 8
            else ("••••" if self.api_key else None)
        )
        return BrokerSessionResponse(
            status="connected" if self.is_authenticated() else "disconnected",
            is_authenticated=self.is_authenticated(),
            access_token=self.access_token,
            user_id=self.user_id,
            user_name=self.user_name,
            broker="Zerodha Kite Connect",
            has_credentials=bool(self.api_key and self.api_secret),
            api_key_masked=masked_key,
            message="Authenticated" if self.is_authenticated() else "Not Authenticated",
        )


def get_broker_session_manager() -> BrokerSessionManager:
    return BrokerSessionManager()
