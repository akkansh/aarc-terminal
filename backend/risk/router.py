"""
FastAPI Router for Adversarial Stress Testing & Macro Market Shock Simulations.
Exposes endpoints to simulate historical crises and synthetic market shocks against
live Zerodha holdings or simulated demo portfolios.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from backend.portfolio.broker_kite import (
    KiteBrokerClient,
    PortfolioHolding,
    get_broker_session_manager,
)
from backend.risk.stress_tester import (
    AdversarialStressTestingEngine,
    MonteCarloSimulationResult,
    StressScenarioDefinition,
    StressTestResult,
)

logger = logging.getLogger("aarc.risk.router")

stress_router = APIRouter()
engine = AdversarialStressTestingEngine()


# ============================================================================
# REQUEST SCHEMAS
# ============================================================================

class CustomHoldingsStressRequest(BaseModel):
    scenario_id: Optional[str] = Field(default="covid_2020", description="Target scenario identifier")
    holdings: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Optional explicit holding list. If omitted, uses active Zerodha session or demo holdings.",
    )
    custom_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional custom parameter dictionary for 'custom' scenario.",
    )


class MonteCarloSimulationRequest(BaseModel):
    holdings: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Optional explicit holding list. If omitted, uses active Zerodha session or demo holdings.",
    )
    num_simulations: Optional[int] = Field(default=2500, description="Number of Monte Carlo paths (default: 2500)")
    time_horizon_days: Optional[int] = Field(default=21, description="Simulation horizon in trading days (default: 21)")
    df_t: Optional[int] = Field(default=4, description="Student-t degrees of freedom for fat-tail shock thickness")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _resolve_effective_holdings(raw_holdings: Optional[List[Dict[str, Any]]]) -> List[PortfolioHolding]:
    """
    Resolves holdings:
    1. If explicit holding list passed, normalizes it.
    2. Else if active Zerodha session is authenticated, fetches live holdings.
    3. Else falls back to standard demo portfolio holdings.
    """
    if raw_holdings and len(raw_holdings) > 0:
        parsed: List[PortfolioHolding] = []
        for item in raw_holdings:
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

            parsed.append(
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
        if parsed:
            return parsed

    # Check active broker session
    session_mgr = get_broker_session_manager()
    if session_mgr.is_authenticated():
        client = session_mgr.get_client()
        if client:
            try:
                live_holdings = client.fetch_holdings()
                if live_holdings:
                    return live_holdings
            except Exception as exc:
                logger.warning("Failed to fetch live broker holdings for stress test: %s", exc)

    # Fallback demo holdings
    return KiteBrokerClient.get_demo_holdings()


# ============================================================================
# ENDPOINTS
# ============================================================================

@stress_router.get("/scenarios", response_model=List[Dict[str, Any]])
def list_available_scenarios():
    """Returns all registered historical, macro, and sector stress testing scenarios."""
    return engine.get_available_scenarios()


@stress_router.get("/scenarios/{scenario_id}", response_model=StressScenarioDefinition)
def get_scenario_detail(scenario_id: str):
    """Retrieve complete definition and shock parameters for a specific scenario."""
    if scenario_id not in engine.scenarios:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stress scenario '{scenario_id}' not found.",
        )
    return engine.scenarios[scenario_id]


@stress_router.post("/run", response_model=StressTestResult)
@stress_router.post("", response_model=StressTestResult)
def run_stress_test_simulation(request: Optional[CustomHoldingsStressRequest] = None):
    """
    Execute multi-scenario shock simulation against user holdings.
    Accepts custom holdings or pulls live holdings from the active Zerodha session.
    """
    req = request or CustomHoldingsStressRequest()
    holdings = _resolve_effective_holdings(req.holdings)
    try:
        result = engine.run_scenario_stress_test(
            holdings=holdings,
            scenario_id=req.scenario_id or "covid_2020",
            custom_params=req.custom_params,
        )
        return result
    except Exception as exc:
        logger.error("Error executing stress test: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Stress test execution failed: {str(exc)}",
        )


@stress_router.get("/run", response_model=StressTestResult)
@stress_router.get("", response_model=StressTestResult)
def run_stress_test_get(
    scenario_id: str = Query(default="covid_2020", description="Target scenario identifier"),
):
    """
    GET convenience endpoint to run stress test with specified scenario against
    active Zerodha session or demo holdings.
    """
    holdings = _resolve_effective_holdings(None)
    try:
        return engine.run_scenario_stress_test(holdings=holdings, scenario_id=scenario_id)
    except Exception as exc:
        logger.error("Error executing GET stress test: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Stress test simulation failed: {str(exc)}",
        )


@stress_router.post("/monte-carlo", response_model=MonteCarloSimulationResult)
def run_monte_carlo_simulation(request: Optional[MonteCarloSimulationRequest] = None):
    """
    Execute probabilistic fat-tailed Student-t Monte Carlo simulation.
    Generates distribution percentiles, VaR 95/99, Expected Shortfall (CVaR),
    and 16-bin histogram for interactive chart visualization.
    """
    req = request or MonteCarloSimulationRequest()
    holdings = _resolve_effective_holdings(req.holdings)
    try:
        return engine.run_monte_carlo_stress_test(
            holdings=holdings,
            num_simulations=req.num_simulations or 2500,
            time_horizon_days=req.time_horizon_days or 21,
            df_t=req.df_t or 4,
        )
    except Exception as exc:
        logger.error("Error running Monte Carlo simulation: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Monte Carlo simulation failed: {str(exc)}",
        )


@stress_router.get("/monte-carlo", response_model=MonteCarloSimulationResult)
def run_monte_carlo_simulation_get(
    num_simulations: int = Query(default=2500, description="Total simulation paths"),
    time_horizon_days: int = Query(default=21, description="Horizon in trading days"),
    df_t: int = Query(default=4, description="Student-t degrees of freedom"),
):
    """GET convenience endpoint to run Monte Carlo on active session/demo holdings."""
    holdings = _resolve_effective_holdings(None)
    try:
        return engine.run_monte_carlo_stress_test(
            holdings=holdings,
            num_simulations=num_simulations,
            time_horizon_days=time_horizon_days,
            df_t=df_t,
        )
    except Exception as exc:
        logger.error("Error running GET Monte Carlo simulation: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Monte Carlo simulation failed: {str(exc)}",
        )
