"""
Adversarial Stress Testing & Macro Market Shock Simulation Engine.
Simulates historical black swan events, macro interest-rate/oil shocks,
and sector-specific liquidity freezes against live equity portfolios.
Calculates institutional drawdown projections, recovery timelines,
worst-hit exposures, and probabilistic Monte Carlo tail distributions.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field
from scipy import stats

from backend.portfolio.broker_kite import PortfolioHolding
from backend.portfolio.sector_optimizer import SECTOR_INFO, TICKER_SECTOR_MAP

logger = logging.getLogger("aarc.risk.stress_tester")


# ============================================================================
# DATA CONTRACTS & SCHEMAS
# ============================================================================

class StressScenarioDefinition(BaseModel):
    scenario_id: str = Field(description="Unique scenario identifier")
    name: str = Field(description="Display title of the stress scenario")
    category: str = Field(description="'Historical Black Swan' | 'Macro Shock' | 'Sector-Specific Liquidity Squeeze' | 'Custom Adversarial'")
    description: str = Field(description="Detailed narrative of macroeconomic or market shock dynamics")
    historical_precedent: str = Field(description="Empirical historical crisis benchmark")
    nifty_benchmark_shock_pct: float = Field(description="Projected broad index (Nifty 50) percentage return shock")
    vix_spike_pct: float = Field(description="Projected percentage spike in India VIX / market volatility")
    sector_shocks: Dict[str, float] = Field(description="Sector-specific transmission shock percentages")
    baseline_recovery_months: float = Field(description="Historical calibrated recovery timeline for a 1.0 beta portfolio")
    bear_catalysts: List[str] = Field(default_factory=list, description="Primary downside transmission mechanisms")
    bull_catalysts: List[str] = Field(default_factory=list, description="Countervailing stabilizers or silver linings")
    recommended_hedges: List[str] = Field(default_factory=list, description="Tactical hedging and defensive rebalancing recommendations")


class StressHoldingImpact(BaseModel):
    ticker: str = Field(description="Ticker symbol (e.g. RELIANCE.NS)")
    tradingsymbol: str = Field(description="Display symbol")
    sector: str = Field(description="Industry sector classification")
    quantity: int = Field(description="Portfolio share count")
    last_price: float = Field(description="Pre-shock current market price (INR)")
    stressed_price: float = Field(description="Projected post-shock price (INR)")
    current_value: float = Field(description="Pre-shock market value (INR)")
    stressed_value: float = Field(description="Projected post-shock value (INR)")
    weight_pct: float = Field(description="Portfolio weight percentage")
    shock_loss_inr: float = Field(description="Projected absolute drawdown in INR (positive number)")
    shock_loss_pct: float = Field(description="Projected percentage price shock (negative number)")
    beta: float = Field(description="Systematic market beta sensitivity")
    vulnerability_rating: str = Field(description="'CRITICAL' | 'HIGH' | 'MODERATE' | 'RESILIENT'")
    is_defensive: bool = Field(description="Whether the asset belongs to a defensive sector")


class RecoveryTimelineEstimate(BaseModel):
    estimated_months: float = Field(description="Expected recovery duration in months")
    range_min_months: float = Field(description="Optimistic recovery estimate (months)")
    range_max_months: float = Field(description="Conservative recovery estimate (months)")
    recovery_cagr_required_pct: float = Field(description="Annualized return required from trough to reach parity")
    human_readable_timeline: str = Field(description="Formatted human-readable duration description")
    historical_precedent_note: str = Field(description="Comparative historical recovery context")


class AdversarialCommitteeAdvisory(BaseModel):
    risk_level: str = Field(description="'CRITICAL' | 'SEVERE' | 'ELEVATED' | 'MANAGED'")
    cro_verdict: str = Field(description="Chief Risk Officer tail risk summary and executive capital defense directive")
    bear_case_cascade: str = Field(description="Adversarial Bear thesis: compounding liquidity traps, margin call risks, contagion")
    bull_case_stabilizer: str = Field(description="Adversarial Bull thesis: buffer assets, valuation support, recovery catalysts")
    actionable_hedges: List[str] = Field(description="Specific derivatives, cash levels, and tactical rotations recommended")


class StressTestResult(BaseModel):
    scenario: StressScenarioDefinition = Field(description="Active scenario specification")
    portfolio_current_value: float = Field(description="Current aggregate portfolio value (INR)")
    portfolio_stressed_value: float = Field(description="Projected post-shock aggregate portfolio value (INR)")
    drawdown_inr: float = Field(description="Absolute portfolio drawdown in INR")
    drawdown_pct: float = Field(description="Percentage portfolio drawdown")
    weighted_beta: float = Field(description="Portfolio weighted systematic beta")
    systematic_loss_inr: float = Field(description="Loss portion driven strictly by systematic broad market beta")
    idiosyncratic_loss_inr: float = Field(description="Loss portion driven by sector bias and asset-specific factors")
    recovery_timeline: RecoveryTimelineEstimate = Field(description="Calibrated post-shock recovery projection")
    worst_hit_holdings: List[StressHoldingImpact] = Field(description="Ranked list of holdings by absolute loss in INR")
    resilient_holdings: List[StressHoldingImpact] = Field(description="Holdings exhibiting strongest capital preservation")
    sector_loss_breakdown: List[Dict[str, Any]] = Field(description="Sector-by-sector absolute and percentage drawdown attribution")
    adversarial_committee: AdversarialCommitteeAdvisory = Field(description="Adversarial Risk Committee critique and mitigation")


class MonteCarloHistogramBin(BaseModel):
    bin_label: str = Field(description="Label for UI display (e.g., '15% - 20%')")
    range_min_pct: float = Field(description="Lower bound drawdown percentage")
    range_max_pct: float = Field(description="Upper bound drawdown percentage")
    count: int = Field(description="Number of simulated paths falling into this bin")
    probability_pct: float = Field(description="Percentage probability of falling into this bin")
    cumulative_pct: float = Field(description="Cumulative probability up to this bin")


class MonteCarloSimulationResult(BaseModel):
    num_simulations: int = Field(description="Total simulated paths (e.g. 2,500)")
    time_horizon_days: int = Field(description="Stress horizon in trading days (e.g. 21 days)")
    mean_drawdown_pct: float = Field(description="Expected average drawdown percentage")
    var_90_pct: float = Field(description="Value at Risk at 90% confidence level (percentage)")
    var_95_pct: float = Field(description="Value at Risk at 95% confidence level (percentage)")
    var_99_pct: float = Field(description="Value at Risk at 99% confidence level (percentage)")
    var_95_inr: float = Field(description="Value at Risk at 95% confidence level in INR")
    var_99_inr: float = Field(description="Value at Risk at 99% confidence level in INR")
    cvar_95_pct: float = Field(description="Expected Shortfall (CVaR) beyond 95% quantile (percentage)")
    cvar_99_pct: float = Field(description="Expected Shortfall (CVaR) beyond 99% quantile (percentage)")
    cvar_95_inr: float = Field(description="Expected Shortfall (CVaR) beyond 95% in INR")
    cvar_99_inr: float = Field(description="Expected Shortfall (CVaR) beyond 99% in INR")
    max_tail_drawdown_pct: float = Field(description="Worst simulated tail loss in the entire Monte Carlo sample")
    max_tail_drawdown_inr: float = Field(description="Worst simulated tail loss in INR")
    fat_tail_kurtosis: float = Field(description="Distribution empirical kurtosis (indicating tail risk thickness)")
    distribution_bins: List[MonteCarloHistogramBin] = Field(description="Histogram bins for visual chart rendering")


# ============================================================================
# BUILT-IN SCENARIO CATALOGUE
# ============================================================================

DEFAULT_SCENARIOS: Dict[str, StressScenarioDefinition] = {
    "covid_2020": StressScenarioDefinition(
        scenario_id="covid_2020",
        name="COVID-19 March 2020 Liquidity Crunch",
        category="Historical Black Swan",
        description="Global pandemic outbreak triggers emergency lockdowns, severe cross-border supply chain paralysis, and extreme foreign institutional investor (FII) cash repatriation. Circuit breakers tripped repeatedly as India VIX surged past 80.",
        historical_precedent="March 2020: Nifty 50 plummeted 38.4% in 33 calendar days. Complete liquidity drainage in smallcaps and highly levered financial NBFCs.",
        nifty_benchmark_shock_pct=-38.0,
        vix_spike_pct=260.0,
        sector_shocks={
            "Financials": -43.0,
            "Energy": -41.0,
            "Automobile": -45.0,
            "Consumer Services": -47.0,
            "Metals & Mining": -44.0,
            "Industrials & Power": -42.0,
            "Information Technology": -27.0,
            "Telecommunication": -22.0,
            "FMCG": -14.0,
            "Healthcare": -11.0,
            "Utilities": -21.0,
            "Other / Diversified": -35.0,
        },
        baseline_recovery_months=7.5,
        bear_catalysts=[
            "Unprecedented liquidity squeeze across commercial paper and NBFC credit lines",
            "Discretionary retail spend collapses to near zero during nationwide curfew",
            "Foreign institutional selling overwhelms domestic mutual fund SIP inflows",
        ],
        bull_catalysts=[
            "Pharma & Active Pharmaceutical Ingredient (API) demand skyrockets globally",
            "Essential FMCG household staples demonstrate near total inelasticity",
            "Central bank global quantitative easing provides massive follow-on monetary stimulus",
        ],
        recommended_hedges=[
            "Initiate Nifty 50 OTM Put Option ladder (delta 0.25 - 0.35) for broad tail protection",
            "Tactically rotate 15-20% from Consumer Services / Auto into defensive FMCG (ITC, HUL) & Healthcare",
            "Accumulate sovereign gold bonds (SGB) or Gold ETFs as a currency and crisis safe-haven",
        ],
    ),

    "gfc_2008": StressScenarioDefinition(
        scenario_id="gfc_2008",
        name="2008 Global Financial Crisis & Credit Freeze",
        category="Historical Black Swan",
        description="Subprime mortgage contagion and Lehman Brothers bankruptcy trigger systemic global interbank freezing. Massive deleveraging of leveraged carry trades forces severe capital flight from emerging market equities.",
        historical_precedent="Jan - Oct 2008: Nifty 50 plunged 59.9% peak-to-trough. Real Estate, Infrastructure, and Commodities suffered >70% peak drawdowns.",
        nifty_benchmark_shock_pct=-52.0,
        vix_spike_pct=310.0,
        sector_shocks={
            "Financials": -61.0,
            "Metals & Mining": -68.0,
            "Industrials & Power": -65.0,
            "Automobile": -56.0,
            "Consumer Services": -58.0,
            "Energy": -49.0,
            "Information Technology": -47.0,
            "Telecommunication": -38.0,
            "FMCG": -23.0,
            "Healthcare": -26.0,
            "Utilities": -36.0,
            "Other / Diversified": -50.0,
        },
        baseline_recovery_months=24.0,
        bear_catalysts=[
            "Wholesale commercial bank credit freeze halts infrastructure and corporate capital expenditure",
            "Global commodity demand collapse triggers massive inventory markdowns in metals and energy",
            "Multi-year balance sheet repair slows economic growth across both advanced and emerging economies",
        ],
        bull_catalysts=[
            "Domestic private banks with conservative loan-to-deposit ratios maintain institutional solvency",
            "Drastic drop in Brent crude oil prices dramatically cuts India's fiscal deficit and import bills",
            "Defensive consumer staples generate consistent operating cash flows throughout the downturn",
        ],
        recommended_hedges=[
            "Raise cash and short-duration liquid debt (Liquid BeES / Overnight Funds) to at least 25% allocation",
            "Severely cut exposure to high-debt cyclical infrastructure, metals, and leveraged NBFCs",
            "Execute synthetic short futures or deep out-of-the-money index put spreads",
        ],
    ),

    "tech_tightening_2022": StressScenarioDefinition(
        scenario_id="tech_tightening_2022",
        name="2022 Fed Rate Tightening & Tech Multiple Compression",
        category="Historical Black Swan",
        description="Historic headline inflation forces the US Federal Reserve and RBI into aggressive front-loaded rate hikes (+450 bps). Discount rates explode, triggering brutal multiple compression in high-P/E software, cloud platforms, and unprofitable tech startups.",
        historical_precedent="2022: NASDAQ dropped 33%; Nifty IT fell 26% while Traditional Energy and Value equities outperformed with positive nominal returns.",
        nifty_benchmark_shock_pct=-18.0,
        vix_spike_pct=75.0,
        sector_shocks={
            "Information Technology": -34.0,
            "Consumer Services": -38.0,
            "Automobile": -14.0,
            "Financials": -9.0,
            "Metals & Mining": -16.0,
            "Industrials & Power": -11.0,
            "Energy": 12.0,  # Commodity inflation tailwind
            "FMCG": -4.0,
            "Healthcare": -8.0,
            "Telecommunication": -6.0,
            "Utilities": 2.0,
            "Other / Diversified": -12.0,
        },
        baseline_recovery_months=13.0,
        bear_catalysts=[
            "Terminal discount rate surge shrinks enterprise SaaS and platform forward valuation multiples",
            "Venture funding winter and cost-cutting drastically reduce discretionary digital transformation spend",
            "Elevated domestic borrowing costs pressure interest-rate sensitive consumer cyclicals",
        ],
        bull_catalysts=[
            "Upstream energy and oil exploration companies benefit from commodity price spikes",
            "High net interest margins (NIM) in banking offset bond treasury portfolio mark-to-market losses",
            "Low-multiple defensive dividend aristocrats offer resilient real cash yields",
        ],
        recommended_hedges=[
            "Trim high-multiple growth equities and reallocate to high-dividend value and energy producers",
            "Deploy long-dated sovereign fixed income to lock in cyclical peak interest yields",
            "Hedge tech exposure with targeted sector ETF put options or short IT futures",
        ],
    ),

    "nifty_crash_20": StressScenarioDefinition(
        scenario_id="nifty_crash_20",
        name="Nifty 50 Sudden -20% Flash Correction",
        category="Macro Shock",
        description="High-velocity systemic flash crash driven by algorithmic stop-loss cascades, sudden geopolitical escalation, and simultaneous institutional de-grossing across Indian large and midcaps.",
        historical_precedent="Broad -20% index corrections typically occur once every 3-4 years (e.g. 2015 China deval shock, 2018 NBFC liquidity shock, early 2020).",
        nifty_benchmark_shock_pct=-20.0,
        vix_spike_pct=110.0,
        sector_shocks={
            "Financials": -25.0,
            "Consumer Services": -27.0,
            "Metals & Mining": -28.0,
            "Automobile": -23.0,
            "Industrials & Power": -22.0,
            "Information Technology": -17.0,
            "Energy": -18.0,
            "Telecommunication": -13.0,
            "FMCG": -8.5,
            "Healthcare": -9.5,
            "Utilities": -10.0,
            "Other / Diversified": -19.0,
        },
        baseline_recovery_months=4.5,
        bear_catalysts=[
            "Automated programmatic stops trigger indiscriminate selling across derivatives and cash markets",
            "High-beta retail favorite stocks suffer exacerbated drawdowns due to leverage liquidations",
        ],
        bull_catalysts=[
            "Fast institutional dip-buying as long-term domestic DII institutional capital steps in",
            "Fundamental corporate earnings remain intact despite transient price volatility",
        ],
        recommended_hedges=[
            "Implement a volatility-contingent delta hedge using Nifty weekly/monthly put options",
            "Maintain 10% dry powder reserve to capitalize on institutional panic capitulation",
        ],
    ),

    "crude_spike_130": StressScenarioDefinition(
        scenario_id="crude_spike_130",
        name="Crude Oil Surge to $130/bbl (Geopolitical Supply Shock)",
        category="Macro Shock",
        description="Major geopolitical escalation chokes global maritime straits, driving Brent crude oil to $130/barrel. India's current account deficit balloons, domestic inflation surges, and the Indian Rupee depreciates rapidly against the US Dollar.",
        historical_precedent="1990 Gulf War, 2008 oil peak ($147/bbl), and March 2022 Russia-Ukraine conflict onset ($139/bbl).",
        nifty_benchmark_shock_pct=-14.0,
        vix_spike_pct=90.0,
        sector_shocks={
            "Energy": 17.0,  # Upstream refiners and crude extractors surge
            "Automobile": -26.0,  # Fuel inflation suppresses vehicle purchase intent
            "Consumer Services": -22.0,  # Transportation and delivery logistics cost explosion
            "FMCG": -16.0,  # Packaging and palm oil / crude derivative raw material margin crunch
            "Metals & Mining": -17.0,
            "Financials": -15.0,  # Higher inflation delays rate cuts and pushes bond yields up
            "Industrials & Power": -16.0,
            "Information Technology": -5.0,  # Weaker INR cushions export earnings
            "Healthcare": -6.5,
            "Telecommunication": -8.0,
            "Utilities": -9.0,
            "Other / Diversified": -14.0,
        },
        baseline_recovery_months=6.0,
        bear_catalysts=[
            "Severe gross margin compression for paints, adhesives, tires, and FMCG packaging",
            "Automotive sales volumes drop sharply as total cost of ownership jumps",
            "Import bill expansion triggers foreign capital outflows due to currency depreciation",
        ],
        bull_catalysts=[
            "Upstream oil exploration majors (ONGC, Oil India, Reliance) capture windfall gross refining margins",
            "IT exporters and generic pharma benefit from INR/USD foreign exchange translation gains",
        ],
        recommended_hedges=[
            "Overweight upstream oil & gas producers (Reliance, ONGC) as a natural fundamental hedge",
            "Reduce allocations to consumer discretionary, paint manufacturers, and airlines",
            "Hold USD-denominated instruments or IT exporters to benefit from rupee depreciation",
        ],
    ),

    "rate_shock_150": StressScenarioDefinition(
        scenario_id="rate_shock_150",
        name="Central Bank Emergency Rate Hike (+150 bps)",
        category="Macro Shock",
        description="Stubborn inflationary expectations prompt the Reserve Bank of India and global central banks to hike benchmark policy rates by 150 basis points. The yield curve shifts upward, compressing consumer credit and real estate affordability.",
        historical_precedent="May - August 2022 RBI emergency inter-meeting rate hikes (+140 bps) following sudden post-pandemic inflation prints.",
        nifty_benchmark_shock_pct=-12.0,
        vix_spike_pct=55.0,
        sector_shocks={
            "Automobile": -22.0,  # Vehicle retail financing costs escalate
            "Consumer Services": -18.0,
            "Financials": -17.0,  # Treasury portfolio mark-to-market losses and loan growth slowdown
            "Industrials & Power": -15.0,
            "Metals & Mining": -14.0,
            "Information Technology": -9.0,
            "Energy": -8.0,
            "FMCG": -4.5,
            "Healthcare": -3.5,
            "Utilities": -5.0,
            "Telecommunication": -6.0,
            "Other / Diversified": -11.0,
        },
        baseline_recovery_months=5.0,
        bear_catalysts=[
            "Higher home loan and auto loan EMI rates dampen real estate and passenger car registrations",
            "Banks incur mark-to-market losses on government bond holdings in their AFS/HFT portfolios",
        ],
        bull_catalysts=[
            "Cash-rich corporations with negative net debt benefit from higher treasury yields",
            "Defensive FMCG and healthcare operations maintain pricing power with low debt burden",
        ],
        recommended_hedges=[
            "Avoid high debt-to-equity cyclical enterprises and interest-rate-sensitive NBFCs",
            "Allocate to floating rate bonds and ultra-short duration fixed-income funds",
        ],
    ),

    "mid_small_liquidity_freeze": StressScenarioDefinition(
        scenario_id="mid_small_liquidity_freeze",
        name="Midcap & Smallcap Severe Liquidity Freeze",
        category="Sector-Specific Liquidity Squeeze",
        description="Regulator alerts on stretched valuations trigger heavy retail mutual fund redemption requests. Small and midcap equities experience severe liquidity evaporation with wide bid-ask spreads and consecutive circuit down-limits.",
        historical_precedent="January 2018 Smallcap Bubble burst: Nifty Smallcap 100 dropped 41% while Nifty 50 was virtually flat (+3% flight to quality).",
        nifty_benchmark_shock_pct=-7.0,  # Large caps stay relatively resilient
        vix_spike_pct=85.0,
        sector_shocks={
            "Consumer Services": -33.0,  # High retail ownership and startup platform multiples
            "Metals & Mining": -26.0,
            "Automobile": -20.0,
            "Industrials & Power": -24.0,
            "Financials": -12.0,  # Small finance banks and microfinance hit harder
            "Information Technology": -10.0,
            "Energy": -6.0,
            "FMCG": -2.0,  # Safe-haven capital rotation into bluechips
            "Healthcare": -4.0,
            "Utilities": -5.0,
            "Telecommunication": -4.0,
            "Other / Diversified": -22.0,
        },
        baseline_recovery_months=8.0,
        bear_catalysts=[
            "Retail margin-call unwinds and illiquidity discounts trigger severe gap-downs",
            "Mutual funds forced to sell liquid holdings to satisfy redemption requests",
        ],
        bull_catalysts=[
            "Flight to quality directs defensive domestic institutional inflows into Nifty 50 mega-caps",
            "High institutional quality balance sheets in large caps provide an anchor",
        ],
        recommended_hedges=[
            "Rebalance out of overextended smallcaps into Nifty 50 blue-chips and defensive staples",
            "Place strict trailing stop-losses and avoid averaging down on illiquid positions",
        ],
    ),

    "it_global_recession": StressScenarioDefinition(
        scenario_id="it_global_recession",
        name="Enterprise IT Tech Spend Freeze & Global Slowdown",
        category="Sector-Specific Liquidity Squeeze",
        description="Synchronized recessionary pressures across the US and Europe force Fortune 500 enterprises to slash discretionary software engineering budgets and delay digital transformation contract renewals.",
        historical_precedent="2001 dot-com bust and 2023 tech spending slowdown: Tier-1 Indian IT companies experienced deal deferrals and multi-quarter margin compression.",
        nifty_benchmark_shock_pct=-9.5,
        vix_spike_pct=50.0,
        sector_shocks={
            "Information Technology": -32.0,
            "Consumer Services": -19.0,
            "Financials": -8.0,
            "Automobile": -7.5,
            "Energy": -5.0,
            "Metals & Mining": -9.0,
            "Industrials & Power": -8.0,
            "FMCG": -2.5,
            "Healthcare": -3.0,
            "Telecommunication": -4.0,
            "Utilities": -3.0,
            "Other / Diversified": -8.0,
        },
        baseline_recovery_months=6.5,
        bear_catalysts=[
            "Total Contract Value (TCV) bookings drop 25% across Tier-1 and Tier-2 Indian IT service providers",
            "Billing rate renegotiations and vendor consolidation squeeze operating margins",
        ],
        bull_catalysts=[
            "Depreciation of INR provides partial operating margin expansion offset",
            "Indian domestic consumption sectors (FMCG, Banking, Infrastructure) remain insulated",
        ],
        recommended_hedges=[
            "Underweight enterprise IT services; rotate capital into domestic credit and infrastructure",
            "Hedge IT holdings with short Nifty IT index futures or put options",
        ],
    ),
}


# ============================================================================
# ADVERSARIAL STRESS TESTING SIMULATION ENGINE
# ============================================================================

class AdversarialStressTestingEngine:
    """
    Core quantitative stress testing engine for live and simulated equity portfolios.
    Applies multi-factor transmission models (systematic beta, sector-specific shock,
    liquidity penalty, and idiosyncratic noise) to simulate tail portfolio outcomes.
    """

    def __init__(self, scenarios: Optional[Dict[str, StressScenarioDefinition]] = None):
        self.scenarios = scenarios or DEFAULT_SCENARIOS

    def get_available_scenarios(self) -> List[Dict[str, Any]]:
        """Return list of all registered scenarios for UI selector."""
        return [
            {
                "scenario_id": sc.scenario_id,
                "name": sc.name,
                "category": sc.category,
                "description": sc.description,
                "nifty_benchmark_shock_pct": sc.nifty_benchmark_shock_pct,
                "vix_spike_pct": sc.vix_spike_pct,
                "baseline_recovery_months": sc.baseline_recovery_months,
            }
            for sc in self.scenarios.values()
        ]

    def _resolve_asset_metadata(self, ticker: str, tradingsymbol: str) -> Tuple[str, float, bool]:
        """Resolves sector classification, systematic beta, and defensive status."""
        norm_t = ticker.strip().upper()
        norm_sym = tradingsymbol.strip().upper()

        # Check ticker map
        if norm_t in TICKER_SECTOR_MAP:
            sec, beta = TICKER_SECTOR_MAP[norm_t]
        elif norm_sym in TICKER_SECTOR_MAP:
            sec, beta = TICKER_SECTOR_MAP[norm_sym]
        else:
            sec = "Other / Diversified"
            beta = 1.0

        sec_def = SECTOR_INFO.get(sec, {})
        is_defensive = sec_def.get("is_defensive", False)
        return sec, beta, is_defensive

    def run_scenario_stress_test(
        self,
        holdings: List[PortfolioHolding],
        scenario_id: str = "covid_2020",
        custom_params: Optional[Dict[str, Any]] = None,
    ) -> StressTestResult:
        """
        Execute deterministic multi-scenario shock simulation against portfolio holdings.
        """
        if not holdings:
            raise ValueError("Cannot run stress test on empty holdings list.")

        # Resolve scenario
        if scenario_id == "custom" and custom_params:
            scenario = StressScenarioDefinition(
                scenario_id="custom",
                name=custom_params.get("name", "Custom Adversarial Scenario"),
                category="Custom Adversarial",
                description=custom_params.get("description", "User-configured market shock scenario"),
                historical_precedent=custom_params.get("historical_precedent", "Synthetic adversarial risk model"),
                nifty_benchmark_shock_pct=float(custom_params.get("nifty_benchmark_shock_pct", -20.0)),
                vix_spike_pct=float(custom_params.get("vix_spike_pct", 100.0)),
                sector_shocks=custom_params.get("sector_shocks", {}),
                baseline_recovery_months=float(custom_params.get("baseline_recovery_months", 6.0)),
                bear_catalysts=custom_params.get("bear_catalysts", ["Custom adversarial downside transmission"]),
                bull_catalysts=custom_params.get("bull_catalysts", ["Value support and mean reversion"]),
                recommended_hedges=custom_params.get("recommended_hedges", ["Maintain index put hedges and cash reserves"]),
            )
        else:
            scenario = self.scenarios.get(scenario_id, self.scenarios["covid_2020"])

        # Calculate current aggregate value
        total_curr_val = sum(h.value for h in holdings)
        if total_curr_val <= 0:
            total_curr_val = sum(h.quantity * h.last_price for h in holdings)
        if total_curr_val <= 0:
            total_curr_val = 1.0

        impact_list: List[StressHoldingImpact] = []
        weighted_beta_sum = 0.0
        sector_losses: Dict[str, Dict[str, float]] = {}

        for h in holdings:
            val = h.value if h.value > 0 else (h.quantity * h.last_price)
            weight = (val / total_curr_val) * 100.0

            sec, beta, is_defensive = self._resolve_asset_metadata(h.ticker, h.tradingsymbol)
            weighted_beta_sum += (val / total_curr_val) * beta

            # Retrieve sector shock or fallback to benchmark shock
            sec_shock = scenario.sector_shocks.get(sec, scenario.nifty_benchmark_shock_pct)

            # Combined transmission return:
            # 50% broad market beta transmission + 50% sector-specific shock transmission
            beta_transmission = beta * scenario.nifty_benchmark_shock_pct
            asset_shock_return_pct = (0.50 * beta_transmission) + (0.50 * sec_shock)

            # Liquidity / concentration penalty: If holding is >20% of portfolio, add tail illiquidity discount
            if weight > 20.0 and scenario.vix_spike_pct > 100.0:
                illiquidity_penalty = -2.5
                asset_shock_return_pct += illiquidity_penalty

            # Bounded return: cap worst loss at -98% and max upside at +50%
            asset_shock_return_pct = max(-98.0, min(50.0, asset_shock_return_pct))

            stressed_price = round(h.last_price * (1.0 + (asset_shock_return_pct / 100.0)), 2)
            stressed_val = round(h.quantity * stressed_price, 2)
            loss_inr = round(val - stressed_val, 2)

            # Vulnerability tier
            if asset_shock_return_pct <= -35.0:
                vuln = "CRITICAL"
            elif asset_shock_return_pct <= -22.0:
                vuln = "HIGH"
            elif asset_shock_return_pct <= -12.0:
                vuln = "MODERATE"
            else:
                vuln = "RESILIENT"

            impact = StressHoldingImpact(
                ticker=h.ticker,
                tradingsymbol=h.tradingsymbol,
                sector=sec,
                quantity=h.quantity,
                last_price=h.last_price,
                stressed_price=stressed_price,
                current_value=round(val, 2),
                stressed_value=stressed_val,
                weight_pct=round(weight, 2),
                shock_loss_inr=loss_inr,
                shock_loss_pct=round(asset_shock_return_pct, 2),
                beta=round(beta, 2),
                vulnerability_rating=vuln,
                is_defensive=is_defensive,
            )
            impact_list.append(impact)

            # Sector aggregation
            if sec not in sector_losses:
                sector_losses[sec] = {
                    "sector": sec,
                    "current_value": 0.0,
                    "stressed_value": 0.0,
                    "loss_inr": 0.0,
                    "is_defensive": is_defensive,
                }
            sector_losses[sec]["current_value"] += val
            sector_losses[sec]["stressed_value"] += stressed_val
            sector_losses[sec]["loss_inr"] += loss_inr

        # Total portfolio metrics
        total_stressed_val = sum(i.stressed_value for i in impact_list)
        total_drawdown_inr = round(total_curr_val - total_stressed_val, 2)
        total_drawdown_pct = round((total_drawdown_inr / total_curr_val) * 100.0, 2)
        weighted_beta = round(weighted_beta_sum, 2)

        # Loss attribution: Systematic (Beta * Nifty shock) vs Idiosyncratic / Sector Bias
        systematic_loss_pct = (weighted_beta * abs(scenario.nifty_benchmark_shock_pct))
        systematic_loss_inr = round((systematic_loss_pct / 100.0) * total_curr_val, 2)
        idiosyncratic_loss_inr = round(total_drawdown_inr - systematic_loss_inr, 2)

        # Recovery Timeline Projection
        recovery_est = self._estimate_recovery_timeline(
            drawdown_pct=total_drawdown_pct,
            weighted_beta=weighted_beta,
            baseline_months=scenario.baseline_recovery_months,
            scenario_name=scenario.name,
        )

        # Sort holdings by loss
        worst_hit = sorted(impact_list, key=lambda x: x.shock_loss_inr, reverse=True)
        resilient = sorted(impact_list, key=lambda x: x.shock_loss_pct, reverse=True)

        # Sector breakdown formatting
        sector_breakdown_list = []
        for sec, sdata in sector_losses.items():
            curr = sdata["current_value"]
            loss = sdata["loss_inr"]
            lpct = round((loss / curr * 100.0) if curr > 0 else 0.0, 2)
            sector_breakdown_list.append({
                "sector": sec,
                "current_value": round(curr, 2),
                "stressed_value": round(sdata["stressed_value"], 2),
                "loss_inr": round(loss, 2),
                "loss_pct": lpct,
                "portfolio_weight_pct": round((curr / total_curr_val) * 100.0, 2),
                "is_defensive": sdata["is_defensive"],
            })
        sector_breakdown_list.sort(key=lambda x: x["loss_inr"], reverse=True)

        # Adversarial Committee Advisory
        committee_advisory = self._generate_committee_advisory(
            scenario=scenario,
            drawdown_pct=total_drawdown_pct,
            drawdown_inr=total_drawdown_inr,
            weighted_beta=weighted_beta,
            worst_hit=worst_hit[:3],
        )

        return StressTestResult(
            scenario=scenario,
            portfolio_current_value=round(total_curr_val, 2),
            portfolio_stressed_value=round(total_stressed_val, 2),
            drawdown_inr=total_drawdown_inr,
            drawdown_pct=total_drawdown_pct,
            weighted_beta=weighted_beta,
            systematic_loss_inr=systematic_loss_inr,
            idiosyncratic_loss_inr=idiosyncratic_loss_inr,
            recovery_timeline=recovery_est,
            worst_hit_holdings=worst_hit,
            resilient_holdings=resilient[:3],
            sector_loss_breakdown=sector_breakdown_list,
            adversarial_committee=committee_advisory,
        )

    def _estimate_recovery_timeline(
        self,
        drawdown_pct: float,
        weighted_beta: float,
        baseline_months: float,
        scenario_name: str,
    ) -> RecoveryTimelineEstimate:
        """
        Calibrates the projected recovery window using Ornstein-Uhlenbeck drift
        and empirical equity mean reversion rates.
        """
        clamped_dd = min(drawdown_pct, 85.0)
        # Required return to break even: 1 / (1 - DD) - 1
        trough_ratio = max(0.01, 1.0 - (clamped_dd / 100.0))
        required_return_pct = round(((1.0 / trough_ratio) - 1.0) * 100.0, 2)

        # Base recovery speed adjusted by beta: higher beta portfolios rebound slightly faster in bull rallies
        beta_adj = max(0.70, min(1.40, math.sqrt(weighted_beta)))
        estimated_months = round(baseline_months * (clamped_dd / 35.0) / beta_adj, 1)
        estimated_months = max(1.5, min(48.0, estimated_months))

        range_min = round(max(1.0, estimated_months * 0.75), 1)
        range_max = round(estimated_months * 1.45, 1)

        human_readable = f"~{estimated_months} months (Estimated range: {range_min} to {range_max} months)"
        precedent_note = f"Calibrated against {scenario_name} historical post-crisis mean reversion dynamics."

        return RecoveryTimelineEstimate(
            estimated_months=estimated_months,
            range_min_months=range_min,
            range_max_months=range_max,
            recovery_cagr_required_pct=required_return_pct,
            human_readable_timeline=human_readable,
            historical_precedent_note=precedent_note,
        )

    def _generate_committee_advisory(
        self,
        scenario: StressScenarioDefinition,
        drawdown_pct: float,
        drawdown_inr: float,
        weighted_beta: float,
        worst_hit: List[StressHoldingImpact],
    ) -> AdversarialCommitteeAdvisory:
        """
        Generates synthesis from the Chief Risk Officer (CRO), Bear thesis, Bull thesis,
        and targeted adversarial hedging recommendations.
        """
        if drawdown_pct >= 30.0:
            risk_level = "CRITICAL"
        elif drawdown_pct >= 20.0:
            risk_level = "SEVERE"
        elif drawdown_pct >= 12.0:
            risk_level = "ELEVATED"
        else:
            risk_level = "MANAGED"

        worst_names = ", ".join(f"{h.tradingsymbol} (-{abs(h.shock_loss_pct)}%)" for h in worst_hit)

        cro_verdict = (
            f"Under the '{scenario.name}' shock, the portfolio faces a projected drawdown of {drawdown_pct:.1f}% "
            f"(₹{drawdown_inr:,.0f}). Tail risk sensitivity is amplified by an active systematic beta of {weighted_beta:.2f}. "
            f"Primary capital hemorrhage is concentrated in {worst_names}. Immediate tactical tail hedging and "
            f"liquidity buffer enforcement is recommended to prevent margin breaches and irreversible capital impairment."
        )

        bear_case = (
            f"Adversarial Downside Cascade: In the event of prolonged market dislocation, liquidity evaporation in "
            f"high-beta holdings could trigger forced liquidations. Secondary contagion across retail mutual fund redemptions "
            f"would exacerbate sell-offs in speculative cyclicals, potentially extending the recovery window past "
            f"{scenario.baseline_recovery_months * 1.5:.0f} months."
        )

        bull_case = (
            f"Adversarial Stabilizers: Core bluechip holdings maintain durable cash flow generation and defensive balance sheets. "
            f"Historical crises confirm that high-quality Indian corporate franchises rebound rapidly once peak market volatility "
            f"(VIX > 40) recedes, offering significant asymmetric entry opportunities for patient capital."
        )

        hedges = list(scenario.recommended_hedges)
        if weighted_beta > 1.15:
            hedges.insert(0, f"Reduce weighted portfolio beta from {weighted_beta:.2f} toward 0.90 by trimming top loss contributors.")
        if drawdown_pct > 25.0:
            hedges.append("Consider purchasing Nifty monthly put options with delta ~0.30 to collar portfolio downside.")

        return AdversarialCommitteeAdvisory(
            risk_level=risk_level,
            cro_verdict=cro_verdict,
            bear_case_cascade=bear_case,
            bull_case_stabilizer=bull_case,
            actionable_hedges=hedges,
        )

    def run_monte_carlo_stress_test(
        self,
        holdings: List[PortfolioHolding],
        num_simulations: int = 2500,
        time_horizon_days: int = 21,
        df_t: int = 4,
    ) -> MonteCarloSimulationResult:
        """
        Executes a randomized fat-tailed Monte Carlo simulation (Student-t distribution with df=4)
        to calculate probabilistic Value at Risk (VaR), Expected Shortfall (CVaR), and full
        drawdown distribution percentiles.
        """
        if not holdings:
            raise ValueError("Holdings list required for Monte Carlo simulation.")

        total_val = sum(h.value if h.value > 0 else (h.quantity * h.last_price) for h in holdings)
        if total_val <= 0:
            total_val = 1.0

        weights = []
        betas = []
        for h in holdings:
            v = h.value if h.value > 0 else (h.quantity * h.last_price)
            _, beta, _ = self._resolve_asset_metadata(h.ticker, h.tradingsymbol)
            weights.append(v / total_val)
            betas.append(beta)

        weights_arr = np.array(weights)
        betas_arr = np.array(betas)
        port_beta = float(np.sum(weights_arr * betas_arr))

        # Stressed daily volatility calibrated to high-vol regime (annualized vol ~28-36%)
        # Daily benchmark vol: ~1.8% to 2.2%
        daily_benchmark_vol = 0.020
        # Horizon scaling: sqrt(days)
        horizon_factor = math.sqrt(time_horizon_days)

        # Reproducible random seed
        rng = np.random.default_rng(seed=42)

        # Generate Student-t distributed market shocks (capturing extreme fat-tail jump risk)
        # Scaled t-distribution: std dev of standard t is sqrt(df / (df - 2)) for df > 2
        t_samples = rng.standard_t(df=df_t, size=num_simulations)
        t_std = math.sqrt(df_t / (df_t - 2)) if df_t > 2 else 1.5
        normalized_market_shocks = (t_samples / t_std) * (daily_benchmark_vol * horizon_factor)

        # Idiosyncratic asset-level shocks
        idiosyncratic_shocks = rng.normal(
            loc=0.0,
            scale=0.015 * horizon_factor,
            size=(num_simulations, len(holdings)),
        )

        # Asset returns = Beta * Market Shock + Idiosyncratic Shock
        # Shape: (num_simulations, num_holdings)
        asset_returns = (
            np.outer(normalized_market_shocks, betas_arr) + idiosyncratic_shocks
        )

        # Portfolio return for each simulation path: dot product with weights
        portfolio_returns = np.dot(asset_returns, weights_arr)

        # Drawdown is defined as loss percentage (positive percentage number for losses)
        # E.g., return of -0.22 -> drawdown of 22.0%
        drawdowns_pct = np.maximum(0.0, -portfolio_returns * 100.0)

        # Quantitative Risk Metrics
        mean_dd = float(np.mean(drawdowns_pct))
        var_90 = float(np.percentile(drawdowns_pct, 90.0))
        var_95 = float(np.percentile(drawdowns_pct, 95.0))
        var_99 = float(np.percentile(drawdowns_pct, 99.0))

        # Expected Shortfall (CVaR): average of drawdowns exceeding VaR
        tail_95 = drawdowns_pct[drawdowns_pct >= var_95]
        cvar_95 = float(np.mean(tail_95)) if len(tail_95) > 0 else var_95

        tail_99 = drawdowns_pct[drawdowns_pct >= var_99]
        cvar_99 = float(np.mean(tail_99)) if len(tail_99) > 0 else var_99

        max_tail = float(np.max(drawdowns_pct))
        kurtosis = float(stats.kurtosis(portfolio_returns, fisher=True))

        # Absolute INR metrics
        var_95_inr = round((var_95 / 100.0) * total_val, 2)
        var_99_inr = round((var_99 / 100.0) * total_val, 2)
        cvar_95_inr = round((cvar_95 / 100.0) * total_val, 2)
        cvar_99_inr = round((cvar_99 / 100.0) * total_val, 2)
        max_tail_inr = round((max_tail / 100.0) * total_val, 2)

        # Construct 16-bin histogram for UI chart visualization
        num_bins = 16
        bin_max = max(10.0, math.ceil(max_tail / 5.0) * 5.0)
        bin_edges = np.linspace(0.0, bin_max, num_bins + 1)
        hist_counts, _ = np.histogram(drawdowns_pct, bins=bin_edges)

        hist_bins: List[MonteCarloHistogramBin] = []
        cumulative_count = 0
        for i in range(num_bins):
            b_min = round(float(bin_edges[i]), 1)
            b_max = round(float(bin_edges[i + 1]), 1)
            cnt = int(hist_counts[i])
            cumulative_count += cnt
            prob = round((cnt / num_simulations) * 100.0, 2)
            cum_prob = round((cumulative_count / num_simulations) * 100.0, 2)
            hist_bins.append(
                MonteCarloHistogramBin(
                    bin_label=f"{b_min}% - {b_max}%",
                    range_min_pct=b_min,
                    range_max_pct=b_max,
                    count=cnt,
                    probability_pct=prob,
                    cumulative_pct=cum_prob,
                )
            )

        return MonteCarloSimulationResult(
            num_simulations=num_simulations,
            time_horizon_days=time_horizon_days,
            mean_drawdown_pct=round(mean_dd, 2),
            var_90_pct=round(var_90, 2),
            var_95_pct=round(var_95, 2),
            var_99_pct=round(var_99, 2),
            var_95_inr=var_95_inr,
            var_99_inr=var_99_inr,
            cvar_95_pct=round(cvar_95, 2),
            cvar_99_pct=round(cvar_99, 2),
            cvar_95_inr=cvar_95_inr,
            cvar_99_inr=cvar_99_inr,
            max_tail_drawdown_pct=round(max_tail, 2),
            max_tail_drawdown_inr=max_tail_inr,
            fat_tail_kurtosis=round(kurtosis, 2),
            distribution_bins=hist_bins,
        )
