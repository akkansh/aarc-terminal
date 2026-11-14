"""
Unit & Mathematical Invariance Tests for Adversarial Stress Testing Engine.
Verifies multi-scenario shock simulation, loss attribution invariance,
recovery timeline calibration, and Monte Carlo tail distribution guarantees.
"""

import pytest
from fastapi.testclient import TestClient

import backend.main
from backend.portfolio.broker_kite import PortfolioHolding, KiteBrokerClient
from backend.risk.stress_tester import (
    DEFAULT_SCENARIOS,
    AdversarialStressTestingEngine,
)


@pytest.fixture
def sample_holdings():
    return [
        PortfolioHolding(
            ticker="RELIANCE.NS",
            tradingsymbol="RELIANCE",
            exchange="NSE",
            quantity=100,
            average_price=1200.0,
            last_price=1300.0,
            value=130000.0,
            invested_value=120000.0,
            pnl=10000.0,
            pnl_percentage=8.33,
            day_change_percentage=0.5,
        ),
        PortfolioHolding(
            ticker="HDFCBANK.NS",
            tradingsymbol="HDFCBANK",
            exchange="NSE",
            quantity=50,
            average_price=1500.0,
            last_price=1600.0,
            value=80000.0,
            invested_value=75000.0,
            pnl=5000.0,
            pnl_percentage=6.67,
            day_change_percentage=-0.2,
        ),
        PortfolioHolding(
            ticker="ITC.NS",
            tradingsymbol="ITC",
            exchange="NSE",
            quantity=200,
            average_price=420.0,
            last_price=460.0,
            value=92000.0,
            invested_value=84000.0,
            pnl=8000.0,
            pnl_percentage=9.52,
            day_change_percentage=0.8,
        ),
    ]


@pytest.fixture
def engine():
    return AdversarialStressTestingEngine()


@pytest.fixture
def client():
    return TestClient(backend.main.app)


def test_scenario_catalogue_completeness(engine):
    """Ensure all core scenario categories and required shock keys exist."""
    scenarios = engine.get_available_scenarios()
    assert len(scenarios) >= 8
    scenario_ids = [s["scenario_id"] for s in scenarios]
    assert "covid_2020" in scenario_ids
    assert "gfc_2008" in scenario_ids
    assert "nifty_crash_20" in scenario_ids
    assert "crude_spike_130" in scenario_ids
    assert "rate_shock_150" in scenario_ids
    assert "mid_small_liquidity_freeze" in scenario_ids
    assert "it_global_recession" in scenario_ids


def test_stress_test_loss_invariance(engine, sample_holdings):
    """Verify mathematical identity: portfolio_stressed_value + drawdown_inr == portfolio_current_value."""
    result = engine.run_scenario_stress_test(sample_holdings, scenario_id="covid_2020")
    
    assert result.portfolio_current_value == pytest.approx(130000.0 + 80000.0 + 92000.0, rel=1e-3)
    assert (result.portfolio_stressed_value + result.drawdown_inr) == pytest.approx(
        result.portfolio_current_value, rel=1e-2
    )
    assert result.drawdown_pct > 0.0
    assert result.drawdown_pct <= 100.0
    assert result.weighted_beta > 0.0


def test_defensive_outperformance(engine, sample_holdings):
    """Verify that defensive FMCG (ITC) suffers lower percentage drawdown than high-beta cyclicals in COVID shock."""
    result = engine.run_scenario_stress_test(sample_holdings, scenario_id="covid_2020")
    
    itc_impact = next(h for h in result.worst_hit_holdings if h.tradingsymbol == "ITC")
    hdfc_impact = next(h for h in result.worst_hit_holdings if h.tradingsymbol == "HDFCBANK")
    
    assert itc_impact.is_defensive is True
    assert abs(itc_impact.shock_loss_pct) < abs(hdfc_impact.shock_loss_pct)


def test_crude_oil_spike_energy_behavior(engine, sample_holdings):
    """Under a crude oil spike scenario, energy exploration (Reliance) should perform better than oil-importing financials."""
    result = engine.run_scenario_stress_test(sample_holdings, scenario_id="crude_spike_130")
    
    rel_impact = next(h for h in result.worst_hit_holdings if h.tradingsymbol == "RELIANCE")
    hdfc_impact = next(h for h in result.worst_hit_holdings if h.tradingsymbol == "HDFCBANK")
    
    # Reliance is mapped to Energy which has positive / resilient shock under crude spike
    assert rel_impact.shock_loss_pct > hdfc_impact.shock_loss_pct


def test_recovery_timeline_bounds(engine, sample_holdings):
    """Recovery timeline should be bounded, positive, and specify required break-even CAGR."""
    result = engine.run_scenario_stress_test(sample_holdings, scenario_id="covid_2020")
    rec = result.recovery_timeline
    
    assert rec.estimated_months > 0.0
    assert rec.range_min_months <= rec.estimated_months
    assert rec.estimated_months <= rec.range_max_months
    assert rec.recovery_cagr_required_pct > 0.0
    assert len(rec.human_readable_timeline) > 0


def test_custom_scenario_execution(engine, sample_holdings):
    """User-supplied custom adversarial parameters should be properly applied."""
    custom = {
        "name": "Custom Armageddon",
        "description": "50% instant crash",
        "nifty_benchmark_shock_pct": -50.0,
        "vix_spike_pct": 200.0,
        "sector_shocks": {"Energy": -50.0, "Financials": -50.0, "FMCG": -30.0},
        "baseline_recovery_months": 18.0,
    }
    result = engine.run_scenario_stress_test(sample_holdings, scenario_id="custom", custom_params=custom)
    assert result.scenario.name == "Custom Armageddon"
    assert result.drawdown_pct >= 35.0


def test_monte_carlo_probabilistic_ordering(engine, sample_holdings):
    """Verify Monte Carlo tail risk properties: VaR 90% <= VaR 95% <= VaR 99% <= CVaR 99%."""
    sim = engine.run_monte_carlo_stress_test(
        sample_holdings, num_simulations=1000, time_horizon_days=21, df_t=4
    )
    assert sim.num_simulations == 1000
    assert sim.var_90_pct <= sim.var_95_pct
    assert sim.var_95_pct <= sim.var_99_pct
    assert sim.var_95_pct <= sim.cvar_95_pct
    assert sim.var_99_pct <= sim.cvar_99_pct
    assert sim.max_tail_drawdown_pct >= sim.cvar_99_pct
    
    # Histogram distribution validation
    assert len(sim.distribution_bins) == 16
    total_count = sum(b.count for b in sim.distribution_bins)
    assert total_count == 1000
    assert sim.distribution_bins[-1].cumulative_pct == pytest.approx(100.0, abs=0.5)


def test_empty_holdings_raises_error(engine):
    with pytest.raises(ValueError):
        engine.run_scenario_stress_test([])
    with pytest.raises(ValueError):
        engine.run_monte_carlo_stress_test([])


def test_fastapi_endpoints(client):
    """Integration test for all FastAPI stress testing endpoints."""
    # 1. Scenarios endpoint
    resp = client.get("/api/risk/stress-test/scenarios")
    assert resp.status_code == 200
    scenarios = resp.json()
    assert len(scenarios) >= 8

    # 2. Scenario detail
    resp_det = client.get("/api/risk/stress-test/scenarios/covid_2020")
    assert resp_det.status_code == 200
    assert resp_det.json()["scenario_id"] == "covid_2020"

    # 3. GET stress test run
    resp_run = client.get("/api/risk/stress-test?scenario_id=covid_2020")
    assert resp_run.status_code == 200
    data = resp_run.json()
    assert "portfolio_current_value" in data
    assert "drawdown_pct" in data
    assert "worst_hit_holdings" in data
    assert "adversarial_committee" in data

    # 4. POST stress test run
    payload = {
        "scenario_id": "nifty_crash_20",
        "holdings": [
            {
                "ticker": "TCS.NS",
                "tradingsymbol": "TCS",
                "quantity": 10,
                "average_price": 3800.0,
                "last_price": 4000.0,
            }
        ],
    }
    resp_post = client.post("/api/risk/stress-test", json=payload)
    assert resp_post.status_code == 200
    assert resp_post.json()["scenario"]["scenario_id"] == "nifty_crash_20"

    # 5. GET Monte Carlo
    resp_mc = client.get("/api/risk/stress-test/monte-carlo?num_simulations=500")
    assert resp_mc.status_code == 200
    mc_data = resp_mc.json()
    assert mc_data["num_simulations"] == 500
    assert "var_95_pct" in mc_data
    assert "distribution_bins" in mc_data
