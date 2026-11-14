"""
FastAPI Router for Portfolio Risk & Diversification Engine.
Exposes Kite Connect OAuth endpoints, automated callback handlers,
live equity holdings ingestion, and quantitative least-risk optimization telemetry.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus
from fastapi import APIRouter, HTTPException, Query, Request, Response, status
from starlette.responses import RedirectResponse
from pydantic import BaseModel, Field

from backend.portfolio.broker_kite import (
    BrokerAuthRequest,
    BrokerCredentialsRequest,
    BrokerSessionResponse,
    KiteBrokerClient,
    PortfolioHolding,
    get_broker_session_manager,
    KITE_AVAILABLE,
)
from backend.portfolio.sector_optimizer import (
    LeastRiskPortfolioOptimizer,
    PortfolioRiskTelemetry,
    SECTOR_INFO,
)

logger = logging.getLogger("aarc.portfolio.router")

portfolio_router = APIRouter()


class CustomHoldingsAnalysisRequest(BaseModel):
    holdings: List[Dict[str, Any]] = Field(description="List of holding dicts containing ticker, quantity, average_price, last_price")
    threshold_pct: Optional[float] = Field(default=20.0, description="Concentration threshold ceiling (default: 20%)")


class OAuthCallbackRequest(BaseModel):
    request_token: str = Field(description="OAuth request token received from Zerodha redirect")
    api_key: Optional[str] = Field(default=None, description="Optional override for API Key")
    api_secret: Optional[str] = Field(default=None, description="Optional override for API Secret")
    threshold_pct: Optional[float] = Field(default=20.0, description="Concentration threshold ceiling (default: 20%)")


class AutomatedOAuthResponse(BaseModel):
    status: str = Field(description="'connected' or 'error'")
    user_id: Optional[str] = None
    user_name: Optional[str] = None
    broker: str = "Zerodha Kite Connect"
    telemetry: PortfolioRiskTelemetry


@portfolio_router.get("/login-url", response_model=Dict[str, Any])
def get_kite_login_url(
    api_key: Optional[str] = Query(default=None, description="Optional Zerodha Kite API Key override"),
):
    """
    Generate the OAuth 2.0 login URL for Zerodha Kite Connect.
    Directs the user to log in and authorize the terminal.
    """
    if not KITE_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="kiteconnect package is not installed on the server.",
        )
    session_mgr = get_broker_session_manager()
    effective_key = api_key or session_mgr.api_key
    if not effective_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Zerodha Kite API Key is not configured. Please supply an API Key in settings.",
        )

    try:
        login_url = session_mgr.get_login_url(api_key=effective_key)
        return {
            "status": "success",
            "login_url": login_url,
            "api_key_configured": bool(effective_key),
            "broker": "Zerodha Kite Connect",
        }
    except Exception as exc:
        logger.error("Failed to generate Kite login URL: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to generate login URL: {str(exc)}",
        )


@portfolio_router.post("/credentials", response_model=BrokerSessionResponse)
def save_broker_credentials(request: BrokerCredentialsRequest):
    """
    Securely store Zerodha Kite API Key and Secret for automated token exchange.
    """
    session_mgr = get_broker_session_manager()
    session_mgr.set_credentials(api_key=request.api_key, api_secret=request.api_secret)
    return session_mgr.get_status()


@portfolio_router.get("/session/status", response_model=BrokerSessionResponse)
def get_broker_session_status():
    """
    Check current broker session status (connected vs disconnected).
    """
    session_mgr = get_broker_session_manager()
    return session_mgr.get_status()


@portfolio_router.post("/disconnect", response_model=BrokerSessionResponse)
def disconnect_broker_session():
    """
    Disconnect active Zerodha Kite session and invalidate access tokens.
    """
    session_mgr = get_broker_session_manager()
    session_mgr.clear_session()
    return session_mgr.get_status()


@portfolio_router.post("/oauth/callback", response_model=AutomatedOAuthResponse)
def process_oauth_callback(request: OAuthCallbackRequest):
    """
    Automated OAuth callback handler called by Next.js when request_token is detected in URL.
    Instantly exchanges the request_token for an access_token, fetches live holdings,
    and returns full quantitative least-risk optimization telemetry in a single round-trip.
    """
    session_mgr = get_broker_session_manager()

    try:
        session_data = session_mgr.exchange_request_token(
            request_token=request.request_token,
            api_key=request.api_key,
            api_secret=request.api_secret,
        )
    except Exception as exc:
        logger.error("OAuth token exchange failure: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Failed to exchange Kite request token: {str(exc)}",
        )

    # Ingest live equity holdings immediately
    client = session_mgr.get_client()
    if not client:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Broker client initialization failed post-exchange.",
        )

    try:
        holdings = client.fetch_holdings()
    except Exception as exc:
        logger.error("Live holdings fetch failed after exchange: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to fetch live holdings from Zerodha Kite: {str(exc)}",
        )

    optimizer = LeastRiskPortfolioOptimizer(concentration_threshold_pct=request.threshold_pct or 20.0)
    telemetry = optimizer.analyze_portfolio(holdings)

    return AutomatedOAuthResponse(
        status="connected",
        user_id=session_data.get("user_id"),
        user_name=session_data.get("user_name"),
        broker="Zerodha Kite Connect",
        telemetry=telemetry,
    )


@portfolio_router.get("/callback")
@portfolio_router.get("/oauth/callback")
def direct_backend_oauth_callback(
    request: Request,
    request_token: Optional[str] = Query(default=None, description="Zerodha Kite OAuth request token"),
    status_param: Optional[str] = Query(default=None, alias="status", description="OAuth status from Zerodha"),
    action: Optional[str] = Query(default=None, description="OAuth action"),
):
    """
    Automated Zerodha Kite Connect OAuth Callback Endpoint.
    1. Captures the incoming request_token parameter from Zerodha query string.
    2. Automatically executes session exchange (kite.generate_session(request_token, api_secret=API_SECRET)) to obtain access_token.
    3. Issues an explicit HTTP 307 Redirect (starlette.responses.RedirectResponse) to immediately bounce browser to:
       http://localhost:3000/portfolio?token={access_token}&success=true
    4. Sets HttpOnly session cookie and handles errors gracefully.
    """
    frontend_url = "http://localhost:3000/portfolio"

    if status_param == "cancelled":
        logger.info("Zerodha OAuth authorization cancelled by user.")
        return RedirectResponse(
            url=f"{frontend_url}?status=cancelled",
            status_code=status.HTTP_307_TEMPORARY_REDIRECT,
        )

    if not request_token:
        logger.warning("OAuth callback invoked without request_token parameter.")
        return RedirectResponse(
            url=f"{frontend_url}?status=error&error_msg=Missing+request_token+parameter",
            status_code=status.HTTP_307_TEMPORARY_REDIRECT,
        )

    session_mgr = get_broker_session_manager()
    try:
        session_data = session_mgr.exchange_request_token(request_token=request_token)
        access_token = session_data.get("access_token", "")
        user_id = session_data.get("user_id", "")

        logger.info("Successfully exchanged Kite request_token for user %s. Redirecting HTTP 307.", user_id)

        # Explicit HTTP 307 Redirect passing generated access token and success status
        redirect_url = f"{frontend_url}?token={access_token}&success=true&status=connected&user_id={user_id}"
        response = RedirectResponse(
            url=redirect_url,
            status_code=status.HTTP_307_TEMPORARY_REDIRECT,
        )

        # Securely store daily access token in session cookie
        if access_token:
            response.set_cookie(
                key="kite_access_token",
                value=access_token,
                httponly=True,
                samesite="lax",
                max_age=86400,  # 24 hours (daily trading session lifespan)
                path="/",
            )

        return response

    except Exception as exc:
        logger.error("Backend redirect OAuth exchange failed: %s", exc)
        safe_msg = quote_plus(str(exc))
        return RedirectResponse(
            url=f"{frontend_url}?status=error&error_msg={safe_msg}",
            status_code=status.HTTP_307_TEMPORARY_REDIRECT,
        )


@portfolio_router.get("/holdings", response_model=PortfolioRiskTelemetry)
def get_portfolio_holdings_get(
    threshold_pct: float = Query(default=20.0, description="Concentration ceiling threshold percentage"),
):
    """
    GET endpoint for retrieving portfolio risk telemetry.
    If authenticated Kite session is active, pulls live holdings.
    Otherwise gracefully serves simulated demo portfolio.
    """
    session_mgr = get_broker_session_manager()
    optimizer = LeastRiskPortfolioOptimizer(concentration_threshold_pct=threshold_pct)

    if session_mgr.is_authenticated():
        client = session_mgr.get_client()
        if client:
            try:
                holdings = client.fetch_holdings()
                return optimizer.analyze_portfolio(holdings)
            except Exception as exc:
                logger.warning("Live fetch failed on active session, falling back to demo: %s", exc)

    # Demo fallback
    demo_holdings = KiteBrokerClient.get_demo_holdings()
    return optimizer.analyze_portfolio(demo_holdings)


@portfolio_router.post("/holdings", response_model=PortfolioRiskTelemetry)
def get_portfolio_holdings_and_risk(
    request: Optional[BrokerAuthRequest] = None,
    threshold_pct: float = Query(default=20.0, description="Concentration ceiling threshold percentage"),
):
    """
    Fetch equity holdings, compute sector exposures, weighted portfolio beta,
    concentration HHI, and least-risk rebalancing advice.
    """
    session_mgr = get_broker_session_manager()
    optimizer = LeastRiskPortfolioOptimizer(concentration_threshold_pct=threshold_pct)

    # 1. Check if explicit live credentials provided in request
    req = request or BrokerAuthRequest()
    if req.api_key and req.access_token and req.api_key.upper() != "DEMO":
        try:
            client = KiteBrokerClient(api_key=req.api_key, access_token=req.access_token)
            holdings = client.fetch_holdings()
            return optimizer.analyze_portfolio(holdings)
        except Exception as exc:
            logger.error("Error retrieving portfolio holdings: %s", exc)
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc))

    # 2. Check if active session exists in session manager
    if session_mgr.is_authenticated():
        client = session_mgr.get_client()
        if client:
            try:
                holdings = client.fetch_holdings()
                return optimizer.analyze_portfolio(holdings)
            except Exception as exc:
                logger.warning("Active session fetch error: %s", exc)

    # 3. Graceful fallback to demo portfolio
    demo_holdings = KiteBrokerClient.get_demo_holdings()
    return optimizer.analyze_portfolio(demo_holdings)


@portfolio_router.get("/demo")
def get_demo_portfolio_risk(
    request: Request,
    request_token: Optional[str] = Query(default=None, description="Zerodha Kite OAuth request token"),
    status_param: Optional[str] = Query(default=None, alias="status", description="OAuth status from Zerodha"),
    action: Optional[str] = Query(default=None, description="OAuth action"),
    threshold_pct: float = Query(default=20.0, description="Concentration ceiling threshold percentage"),
):
    """
    If invoked as an OAuth redirect callback containing request_token (e.g. app redirect URI was set to /api/portfolio/demo),
    captures request_token, exchanges session, and issues HTTP 307 Redirect to http://localhost:3000/portfolio?token=...
    Otherwise, returns pre-calculated demo portfolio telemetry.
    """
    if request_token or status_param == "cancelled":
        return direct_backend_oauth_callback(
            request=request,
            request_token=request_token,
            status_param=status_param,
            action=action,
        )

    optimizer = LeastRiskPortfolioOptimizer(concentration_threshold_pct=threshold_pct)
    demo_holdings = KiteBrokerClient.get_demo_holdings()
    return optimizer.analyze_portfolio(demo_holdings)


@portfolio_router.post("/analyze", response_model=PortfolioRiskTelemetry)
def analyze_custom_portfolio(request: CustomHoldingsAnalysisRequest):
    """
    Accepts arbitrary user holdings payload and computes full sector concentration,
    systematic beta, and least-risk diversification advice.
    """
    holdings: List[PortfolioHolding] = []
    for item in request.holdings:
        ticker = str(item.get("ticker", "UNKNOWN")).strip().upper()
        tsym = str(item.get("tradingsymbol", ticker.split(".")[0])).strip().upper()
        qty = int(item.get("quantity", 0))
        if qty <= 0:
            continue
        avg = float(item.get("average_price", 0.0))
        last = float(item.get("last_price", avg))
        curr_val = round(qty * last, 2)
        inv_val = round(qty * avg, 2)
        pnl = round(curr_val - inv_val, 2)
        pnl_pct = round(((curr_val - inv_val) / inv_val * 100.0) if inv_val > 0 else 0.0, 2)

        holdings.append(
            PortfolioHolding(
                ticker=ticker,
                tradingsymbol=tsym,
                exchange=item.get("exchange", "NSE"),
                quantity=qty,
                average_price=avg,
                last_price=last,
                value=curr_val,
                invested_value=inv_val,
                pnl=pnl,
                pnl_percentage=pnl_pct,
                day_change_percentage=float(item.get("day_change_percentage", 0.0)),
                isin=item.get("isin"),
            )
        )

    optimizer = LeastRiskPortfolioOptimizer(concentration_threshold_pct=request.threshold_pct or 20.0)
    return optimizer.analyze_portfolio(holdings)


@portfolio_router.get("/sectors", response_model=Dict[str, Any])
def list_sectors():
    """Returns sector classification taxonomy, baseline systematic betas, and defensive status."""
    return {"status": "success", "sectors": SECTOR_INFO}
