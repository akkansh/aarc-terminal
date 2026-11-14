"""
Unit and integration tests for Portfolio Risk & Diversification Engine.
Tests Kite Connect broker integration, sector concentration analysis,
least-risk rebalancing optimization, and FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from backend.main import app
from backend.portfolio.broker_kite import (
    KiteBrokerClient,
    PortfolioHolding,
    BrokerAuthRequest,
)
from backend.portfolio.sector_optimizer import (
    LeastRiskPortfolioOptimizer,
    SECTOR_INFO,
)


@pytest.fixture
def test_client():
    return TestClient(app)


def test_sector_resolution_and_baseline_betas():
    optimizer = LeastRiskPortfolioOptimizer(concentration_threshold_pct=20.0)

    # Standard equities
    sec, beta = optimizer.resolve_sector_and_beta("RELIANCE.NS")
    assert sec == "Energy"
    assert beta == 1.05

    sec, beta = optimizer.resolve_sector_and_beta("HDFCBANK.NS")
    assert sec == "Financials"
    assert beta == 1.18

    sec, beta = optimizer.resolve_sector_and_beta("TCS.NS")
    assert sec == "Information Technology"
    assert beta == 1.05

    sec, beta = optimizer.resolve_sector_and_beta("ITC.NS")
    assert sec == "FMCG"
    assert beta == 0.62

    sec, beta = optimizer.resolve_sector_and_beta("SUNPHARMA.NS")
    assert sec == "Healthcare"
    assert beta == 0.68

    sec, beta = optimizer.resolve_sector_and_beta("ETERNAL.NS")
    assert sec == "Consumer Services"
    assert beta == 1.32

    # Global equities
    sec, beta = optimizer.resolve_sector_and_beta("AAPL")
    assert sec == "Information Technology"
    assert beta == 1.08

    # Keyword heuristic fallback
    sec, beta = optimizer.resolve_sector_and_beta("KOTAKBANK")
    assert sec == "Financials"

    # Unknown fallback
    sec, beta = optimizer.resolve_sector_and_beta("UNKNOWN_XYZ.NS")
    assert sec == "Other / Diversified"
    assert beta == 1.0


def test_least_risk_optimization_on_concentrated_portfolio():
    optimizer = LeastRiskPortfolioOptimizer(concentration_threshold_pct=20.0)
    demo_holdings = KiteBrokerClient.get_demo_holdings()

    telemetry = optimizer.analyze_portfolio(demo_holdings)

    # Total valuation checks
    assert telemetry.total_value > 0
    assert telemetry.holdings_count == len(demo_holdings)

    # Check overweight sectors detection (Financials > 20%, Energy > 20%)
    assert telemetry.overweight_sectors_count >= 1
    assert telemetry.concentration_posture in ["MODERATE", "HIGH_RISK_CONCENTRATED"]
    assert telemetry.hhi_concentration_index > 1500

    # Sector exposures verification
    sectors_map = {s.sector: s for s in telemetry.sectors}
    assert "Financials" in sectors_map
    assert "Energy" in sectors_map
    assert sectors_map["Financials"].is_overweight is True
    assert sectors_map["Financials"].excess_pct > 0
    assert sectors_map["Financials"].excess_value > 0

    # Rebalance recommendations verification
    recs = telemetry.recommendations
    assert len(recs) > 0

    trim_actions = [r for r in recs if r.action == "TRIM"]
    assert len(trim_actions) >= 1
    assert any(r.sector == "Financials" for r in trim_actions)

    defensive_actions = [r for r in recs if r.action == "ALLOCATE_DEFENSIVE"]
    assert len(defensive_actions) >= 1
    assert any(r.sector in ["FMCG", "Healthcare"] for r in defensive_actions)

    # Projected beta reduction
    assert telemetry.projected_rebalanced_beta <= telemetry.portfolio_weighted_beta
    assert telemetry.beta_reduction_pct >= 0


def test_least_risk_optimization_on_well_diversified_portfolio():
    optimizer = LeastRiskPortfolioOptimizer(concentration_threshold_pct=20.0)

    # Balanced portfolio with 6 different sectors each ~16.6%
    balanced_holdings = [
        PortfolioHolding(ticker="RELIANCE.NS", tradingsymbol="RELIANCE", exchange="NSE", quantity=10, average_price=1000, last_price=1000, value=10000, invested_value=10000, pnl=0, pnl_percentage=0),
        PortfolioHolding(ticker="HDFCBANK.NS", tradingsymbol="HDFCBANK", exchange="NSE", quantity=10, average_price=1000, last_price=1000, value=10000, invested_value=10000, pnl=0, pnl_percentage=0),
        PortfolioHolding(ticker="TCS.NS", tradingsymbol="TCS", exchange="NSE", quantity=10, average_price=1000, last_price=1000, value=10000, invested_value=10000, pnl=0, pnl_percentage=0),
        PortfolioHolding(ticker="ITC.NS", tradingsymbol="ITC", exchange="NSE", quantity=10, average_price=1000, last_price=1000, value=10000, invested_value=10000, pnl=0, pnl_percentage=0),
        PortfolioHolding(ticker="SUNPHARMA.NS", tradingsymbol="SUNPHARMA", exchange="NSE", quantity=10, average_price=1000, last_price=1000, value=10000, invested_value=10000, pnl=0, pnl_percentage=0),
        PortfolioHolding(ticker="TATAMOTORS.NS", tradingsymbol="TATAMOTORS", exchange="NSE", quantity=10, average_price=1000, last_price=1000, value=10000, invested_value=10000, pnl=0, pnl_percentage=0),
    ]

    telemetry = optimizer.analyze_portfolio(balanced_holdings)
    assert telemetry.overweight_sectors_count == 0
    assert telemetry.concentration_posture == "DIVERSIFIED"
    assert telemetry.hhi_concentration_index < 2000

    hold_recs = [r for r in telemetry.recommendations if r.action == "HOLD"]
    assert len(hold_recs) == 1


def test_empty_portfolio_handling():
    optimizer = LeastRiskPortfolioOptimizer()
    telemetry = optimizer.analyze_portfolio([])
    assert telemetry.total_value == 0.0
    assert telemetry.holdings_count == 0
    assert telemetry.sectors == []
    assert telemetry.recommendations == []


def test_api_portfolio_demo_endpoint(test_client):
    response = test_client.get("/api/portfolio/demo")
    assert response.status_code == 200
    data = response.json()
    assert "total_value" in data
    assert "portfolio_weighted_beta" in data
    assert "sectors" in data
    assert "recommendations" in data
    assert len(data["sectors"]) > 0


def test_api_portfolio_sectors_endpoint(test_client):
    response = test_client.get("/api/portfolio/sectors")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "Financials" in data["sectors"]
    assert "FMCG" in data["sectors"]
    assert data["sectors"]["FMCG"]["is_defensive"] is True


def test_api_portfolio_analyze_custom(test_client):
    payload = {
        "holdings": [
            {"ticker": "INFY.NS", "quantity": 100, "average_price": 1500.0, "last_price": 1600.0},
            {"ticker": "ITC.NS", "quantity": 200, "average_price": 400.0, "last_price": 450.0},
        ],
        "threshold_pct": 25.0,
    }
    response = test_client.post("/api/portfolio/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["holdings_count"] == 2
    assert data["total_value"] == (100 * 1600.0) + (200 * 450.0)
    assert len(data["sectors"]) == 2


def test_api_portfolio_holdings_demo_fallback(test_client):
    # When api_key is 'DEMO' or empty, returns simulated demo telemetry
    payload = {"api_key": "DEMO"}
    response = test_client.post("/api/portfolio/holdings", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_value"] > 0
    assert len(data["holdings"]) > 0


def test_api_portfolio_login_url(test_client):
    response = test_client.get("/api/portfolio/login-url?api_key=test_api_key_123")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "login_url" in data
    assert "test_api_key_123" in data["login_url"]


def test_broker_session_manager_and_status():
    from backend.portfolio.broker_kite import get_broker_session_manager
    session_mgr = get_broker_session_manager()
    session_mgr.set_credentials("test_api_key_abc", "test_api_secret_xyz")

    status = session_mgr.get_status()
    assert status.has_credentials is True
    assert status.api_key_masked is not None
    assert "test" in status.api_key_masked

    session_mgr.clear_session()
    status_cleared = session_mgr.get_status()
    assert status_cleared.is_authenticated is False


def test_api_portfolio_credentials_and_session_status(test_client):
    # Set credentials
    cred_payload = {
        "api_key": "kite_key_9988",
        "api_secret": "kite_secret_9988",
        "redirect_uri": "http://localhost:3000/portfolio"
    }
    res = test_client.post("/api/portfolio/credentials", json=cred_payload)
    assert res.status_code == 200
    data = res.json()
    assert data["has_credentials"] is True

    # Check status endpoint
    status_res = test_client.get("/api/portfolio/session/status")
    assert status_res.status_code == 200
    status_data = status_res.json()
    assert status_data["broker"] == "Zerodha Kite Connect"
    assert status_data["has_credentials"] is True

    # Disconnect
    disc_res = test_client.post("/api/portfolio/disconnect")
    assert disc_res.status_code == 200
    assert disc_res.json()["is_authenticated"] is False


def test_api_portfolio_holdings_get_endpoint(test_client):
    # GET /api/portfolio/holdings should return demo or live telemetry
    res = test_client.get("/api/portfolio/holdings?threshold_pct=20.0")
    assert res.status_code == 200
    data = res.json()
    assert "total_value" in data
    assert len(data["holdings"]) > 0


@patch("backend.portfolio.broker_kite.KiteConnect")
def test_api_portfolio_direct_backend_callback(mock_kite_cls, test_client):
    # Mock KiteConnect instance methods
    mock_instance = MagicMock()
    mock_instance.generate_session.return_value = {
        "access_token": "mock_daily_access_token_987",
        "user_id": "XY9988",
        "user_name": "Redirect Trader",
    }
    mock_kite_cls.return_value = mock_instance

    # 1. Cancelled status redirect -> HTTP 307
    res_cancel = test_client.get("/api/portfolio/callback?status=cancelled", follow_redirects=False)
    assert res_cancel.status_code == 307
    assert "status=cancelled" in res_cancel.headers["location"]

    # 2. Successful authorization -> HTTP 307 redirect to frontend with token=... and cookie
    res_success = test_client.get("/api/portfolio/callback?request_token=valid_request_token_abc&status=success", follow_redirects=False)
    assert res_success.status_code == 307
    assert "http://localhost:3000/portfolio?token=mock_daily_access_token_987&success=true" in res_success.headers["location"]
    assert "set-cookie" in res_success.headers
    assert "kite_access_token" in res_success.headers["set-cookie"]

    # 3. If user configured redirect URI to /api/portfolio/demo -> HTTP 307 redirect to frontend
    res_demo_redirect = test_client.get("/api/portfolio/demo?request_token=valid_request_token_abc&status=success", follow_redirects=False)
    assert res_demo_redirect.status_code == 307
    assert "http://localhost:3000/portfolio?token=mock_daily_access_token_987&success=true" in res_demo_redirect.headers["location"]

    # 4. Root /callback route also handles HTTP 307 redirect
    res_root = test_client.get("/callback?status=cancelled", follow_redirects=False)
    assert res_root.status_code == 307
    assert "status=cancelled" in res_root.headers["location"]


@patch("backend.portfolio.broker_kite.KiteConnect")
def test_api_portfolio_oauth_callback_flow(mock_kite_cls, test_client):
    # Mock KiteConnect instance methods
    mock_instance = MagicMock()
    mock_instance.generate_session.return_value = {
        "access_token": "mock_access_token_12345",
        "user_id": "AB1234",
        "user_name": "Test Trader"
    }
    mock_instance.holdings.return_value = [
        {
            "tradingsymbol": "INFY",
            "exchange": "NSE",
            "quantity": 50,
            "average_price": 1500.0,
            "last_price": 1600.0,
            "pnl": 5000.0,
            "day_change_percentage": 1.2
        },
        {
            "tradingsymbol": "ITC",
            "exchange": "NSE",
            "quantity": 100,
            "average_price": 400.0,
            "last_price": 440.0,
            "pnl": 4000.0,
            "day_change_percentage": 0.8
        }
    ]
    mock_kite_cls.return_value = mock_instance

    callback_payload = {
        "request_token": "mock_request_token_abc",
        "api_key": "test_key",
        "api_secret": "test_secret",
        "threshold_pct": 20.0
    }
    res = test_client.post("/api/portfolio/oauth/callback", json=callback_payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "connected"
    assert data["user_id"] == "AB1234"
    assert data["telemetry"]["holdings_count"] == 2
    assert data["telemetry"]["total_value"] == (50 * 1600.0) + (100 * 440.0)
