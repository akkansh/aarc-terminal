"""
Quantitative Sector Concentration & Least-Risk Diversification Engine.
Calculates sector exposures, portfolio weighted beta, Herfindahl-Hirschman Index (HHI),
and generates institutional least-risk rebalancing advice to dampen tail risk.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from backend.portfolio.broker_kite import PortfolioHolding

logger = logging.getLogger("aarc.portfolio.optimizer")


# Sector definitions, baseline systematic betas, and representative defensive proxies
SECTOR_INFO = {
    "Financials": {
        "description": "Banking, Insurance, and Non-Banking Financial Corporations",
        "baseline_beta": 1.22,
        "is_defensive": False,
        "target_max_pct": 20.0,
    },
    "Information Technology": {
        "description": "Enterprise IT Services, SaaS, and Cloud Infrastructure",
        "baseline_beta": 1.15,
        "is_defensive": False,
        "target_max_pct": 20.0,
    },
    "Energy": {
        "description": "Oil, Natural Gas, Refining, and Petrochemical Exploration",
        "baseline_beta": 1.05,
        "is_defensive": False,
        "target_max_pct": 20.0,
    },
    "Consumer Services": {
        "description": "Digital Platforms, Food Delivery, Quick Commerce, and Retail",
        "baseline_beta": 1.30,
        "is_defensive": False,
        "target_max_pct": 15.0,
    },
    "Automobile": {
        "description": "OEMs, Commercial Vehicles, EV Platforms, and Ancillaries",
        "baseline_beta": 1.20,
        "is_defensive": False,
        "target_max_pct": 15.0,
    },
    "Metals & Mining": {
        "description": "Ferrous & Non-Ferrous Metals, Mining, and Smelting",
        "baseline_beta": 1.38,
        "is_defensive": False,
        "target_max_pct": 12.0,
    },
    "Telecommunication": {
        "description": "Cellular Network Operators, Telecom Equipment, and Fiber",
        "baseline_beta": 0.95,
        "is_defensive": False,
        "target_max_pct": 15.0,
    },
    "Industrials & Power": {
        "description": "Engineering, Capital Goods, Defense, and Heavy Infrastructure",
        "baseline_beta": 1.10,
        "is_defensive": False,
        "target_max_pct": 15.0,
    },
    "FMCG": {
        "description": "Fast Moving Consumer Goods, Household Staples, and Tobacco",
        "baseline_beta": 0.65,
        "is_defensive": True,
        "defensive_proxies": ["ITC.NS", "HINDUNILVR.NS", "NESTLEIND.NS"],
        "target_max_pct": 20.0,
    },
    "Healthcare": {
        "description": "Pharmaceuticals, Active Ingredients, Biotech, and Diagnostics",
        "baseline_beta": 0.70,
        "is_defensive": True,
        "defensive_proxies": ["SUNPHARMA.NS", "CIPLA.NS", "DRREDDY.NS"],
        "target_max_pct": 20.0,
    },
    "Utilities": {
        "description": "Power Grid Transmission, Water, and City Gas Utilities",
        "baseline_beta": 0.60,
        "is_defensive": True,
        "defensive_proxies": ["POWERGRID.NS", "NTPC.NS"],
        "target_max_pct": 15.0,
    },
    "Other / Diversified": {
        "description": "Unclassified equities or multi-industry conglomerates",
        "baseline_beta": 1.00,
        "is_defensive": False,
        "target_max_pct": 15.0,
    },
}

# Extensive Ticker to Sector & Beta mapping (India Large/Mid Caps & Global Assets)
TICKER_SECTOR_MAP: Dict[str, Tuple[str, float]] = {
    # Energy
    "RELIANCE.NS": ("Energy", 1.05),
    "RELIANCE": ("Energy", 1.05),
    "ONGC.NS": ("Energy", 1.12),
    "BPCL.NS": ("Energy", 1.15),
    "IOC.NS": ("Energy", 1.08),
    "COALINDIA.NS": ("Energy", 0.98),
    "XOM": ("Energy", 1.02),
    "CVX": ("Energy", 1.04),

    # Financials
    "HDFCBANK.NS": ("Financials", 1.18),
    "HDFCBANK": ("Financials", 1.18),
    "ICICIBANK.NS": ("Financials", 1.25),
    "ICICIBANK": ("Financials", 1.25),
    "SBIN.NS": ("Financials", 1.30),
    "SBIN": ("Financials", 1.30),
    "KOTAKBANK.NS": ("Financials", 1.12),
    "AXISBANK.NS": ("Financials", 1.28),
    "BAJFINANCE.NS": ("Financials", 1.35),
    "BAJAJFINSV.NS": ("Financials", 1.28),
    "JPM": ("Financials", 1.15),
    "BAC": ("Financials", 1.22),

    # Information Technology
    "TCS.NS": ("Information Technology", 1.05),
    "TCS": ("Information Technology", 1.05),
    "INFY.NS": ("Information Technology", 1.18),
    "INFY": ("Information Technology", 1.18),
    "WIPRO.NS": ("Information Technology", 1.12),
    "HCLTECH.NS": ("Information Technology", 1.10),
    "TECHM.NS": ("Information Technology", 1.20),
    "LTIM.NS": ("Information Technology", 1.25),
    "NVDA": ("Information Technology", 1.65),
    "MSFT": ("Information Technology", 1.12),
    "AAPL": ("Information Technology", 1.08),
    "AVGO": ("Information Technology", 1.35),
    "GOOGL": ("Information Technology", 1.15),

    # Consumer Services / Quick Commerce
    "ETERNAL.NS": ("Consumer Services", 1.32),
    "ETERNAL": ("Consumer Services", 1.32),
    "ZOMATO.NS": ("Consumer Services", 1.32),
    "ZOMATO": ("Consumer Services", 1.32),
    "AMZN": ("Consumer Services", 1.22),
    "TSLA": ("Consumer Services", 1.85),

    # Automobile
    "TATAMOTORS.NS": ("Automobile", 1.35),
    "TATAMOTORS": ("Automobile", 1.35),
    "MARUTI.NS": ("Automobile", 1.02),
    "M&M.NS": ("Automobile", 1.18),
    "BAJAJ-AUTO.NS": ("Automobile", 0.92),
    "EICHERMOT.NS": ("Automobile", 1.08),

    # Metals & Mining
    "TATASTEEL.NS": ("Metals & Mining", 1.42),
    "JSWSTEEL.NS": ("Metals & Mining", 1.38),
    "HINDALCO.NS": ("Metals & Mining", 1.45),
    "VEDL.NS": ("Metals & Mining", 1.50),

    # Telecommunication
    "BHARTIARTL.NS": ("Telecommunication", 0.92),
    "BHARTIARTL": ("Telecommunication", 0.92),
    "IDEA.NS": ("Telecommunication", 1.70),

    # Industrials & Power
    "LT.NS": ("Industrials & Power", 1.12),
    "ADANIENT.NS": ("Industrials & Power", 1.65),
    "ADANIPORTS.NS": ("Industrials & Power", 1.35),
    "NTPC.NS": ("Industrials & Power", 0.88),
    "POWERGRID.NS": ("Industrials & Power", 0.72),

    # FMCG (Defensive)
    "ITC.NS": ("FMCG", 0.62),
    "ITC": ("FMCG", 0.62),
    "HINDUNILVR.NS": ("FMCG", 0.68),
    "HINDUNILVR": ("FMCG", 0.68),
    "NESTLEIND.NS": ("FMCG", 0.60),
    "BRITANNIA.NS": ("FMCG", 0.65),
    "DABUR.NS": ("FMCG", 0.64),
    "PG": ("FMCG", 0.58),
    "KO": ("FMCG", 0.62),

    # Healthcare (Defensive)
    "SUNPHARMA.NS": ("Healthcare", 0.68),
    "SUNPHARMA": ("Healthcare", 0.68),
    "CIPLA.NS": ("Healthcare", 0.70),
    "DRREDDY.NS": ("Healthcare", 0.72),
    "DIVISLAB.NS": ("Healthcare", 0.85),
    "APOLLOHOSP.NS": ("Healthcare", 0.90),
    "JNJ": ("Healthcare", 0.55),
    "PFE": ("Healthcare", 0.65),
}


class HoldingEnriched(BaseModel):
    ticker: str
    tradingsymbol: str
    exchange: str
    quantity: int
    average_price: float
    last_price: float
    value: float
    invested_value: float
    pnl: float
    pnl_percentage: float
    day_change_percentage: float
    weight_pct: float = Field(description="Weight as percentage of total portfolio")
    sector: str = Field(description="Classified industry sector")
    beta: float = Field(description="Systematic market beta coefficient")
    is_overconcentrated_asset: bool = Field(default=False, description="True if asset alone exceeds 15% of portfolio")


class SectorExposure(BaseModel):
    sector: str
    value: float = Field(description="Total market value allocated to this sector")
    weight_pct: float = Field(description="Weight as percentage of total portfolio")
    target_weight_pct: float = Field(description="Recommended least-risk target ceiling")
    is_overweight: bool = Field(description="True if weight_pct exceeds concentration threshold (>20%)")
    excess_pct: float = Field(description="Weight in excess of target ceiling")
    excess_value: float = Field(description="Currency value in excess of target ceiling")
    baseline_beta: float = Field(description="Sector systematic beta")
    is_defensive: bool = Field(description="True if sector is defensive (FMCG, Healthcare, Utilities)")
    holdings_count: int = Field(description="Number of assets held in this sector")
    assets: List[str] = Field(default_factory=list, description="List of tickers held in this sector")


class RebalanceRecommendation(BaseModel):
    action: str = Field(description="'TRIM', 'ALLOCATE_DEFENSIVE', or 'HOLD'")
    sector: str = Field(description="Target sector for rebalancing")
    current_weight_pct: float
    recommended_weight_pct: float
    amount_currency: float = Field(description="Recommended rupee/currency value to adjust")
    proxy_tickers: List[str] = Field(default_factory=list, description="Suggested instruments or assets to trim/buy")
    rationale: str = Field(description="Quantitative justification grounded in systemic beta and tail risk containment")


class PortfolioRiskTelemetry(BaseModel):
    total_value: float = Field(description="Total current market value")
    total_invested: float = Field(description="Total capital invested")
    total_pnl: float = Field(description="Total unrealized profit/loss")
    total_pnl_pct: float = Field(description="Total portfolio return percentage")
    holdings_count: int = Field(description="Number of distinct equity holdings")
    portfolio_weighted_beta: float = Field(description="Current market beta weighted across all holdings")
    projected_rebalanced_beta: float = Field(description="Estimated beta after executing least-risk rebalancing")
    beta_reduction_pct: float = Field(description="Percentage reduction in systemic equity beta")
    hhi_concentration_index: float = Field(description="Herfindahl-Hirschman Index for sector concentration (0-10000)")
    concentration_posture: str = Field(description="'DIVERSIFIED', 'MODERATE', or 'HIGH_RISK_CONCENTRATED'")
    overweight_sectors_count: int = Field(description="Count of sectors exceeding 20% concentration ceiling")
    sectors: List[SectorExposure] = Field(description="List of all sector exposures sorted by weight descending")
    holdings: List[HoldingEnriched] = Field(description="Enriched list of holdings with sector and beta")
    recommendations: List[RebalanceRecommendation] = Field(description="Actionable least-risk rebalancing directives")
    threshold_pct: float = Field(default=20.0, description="Concentration threshold applied")


class LeastRiskPortfolioOptimizer:
    """
    Analyzes portfolio holdings, calculates sector exposure weights,
    computes concentration risk metrics (HHI, Weighted Beta), and executes
    least-risk rebalancing to minimize tail-risk exposure during market shocks.
    """

    def __init__(self, concentration_threshold_pct: float = 20.0):
        self.threshold_pct = concentration_threshold_pct

    def resolve_sector_and_beta(self, ticker: str, tradingsymbol: Optional[str] = None) -> Tuple[str, float]:
        """Resolve ticker to sector name and baseline beta using comprehensive lookup table."""
        clean_ticker = ticker.strip().upper()
        if clean_ticker in TICKER_SECTOR_MAP:
            return TICKER_SECTOR_MAP[clean_ticker]

        # Check trading symbol without exchange suffix
        if tradingsymbol:
            clean_tsym = tradingsymbol.strip().upper()
            if clean_tsym in TICKER_SECTOR_MAP:
                return TICKER_SECTOR_MAP[clean_tsym]

        # Check base ticker without .NS or .BO
        base_ticker = clean_ticker.split(".")[0]
        if base_ticker in TICKER_SECTOR_MAP:
            return TICKER_SECTOR_MAP[base_ticker]

        # Keyword heuristics for common Indian names
        if any(w in clean_ticker for w in ["BANK", "FIN", "CAPITAL", "INSUR"]):
            return "Financials", 1.22
        if any(w in clean_ticker for w in ["TECH", "INFOSYS", "SOFT", "DATA"]):
            return "Information Technology", 1.15
        if any(w in clean_ticker for w in ["OIL", "PETRO", "GAS", "ENERGY"]):
            return "Energy", 1.05
        if any(w in clean_ticker for w in ["PHARMA", "LAB", "HEALTH", "DRUG"]):
            return "Healthcare", 0.70
        if any(w in clean_ticker for w in ["MOTORS", "AUTO", "TYRE"]):
            return "Automobile", 1.20
        if any(w in clean_ticker for w in ["STEEL", "METAL", "MINE", "ALUM"]):
            return "Metals & Mining", 1.38
        if any(w in clean_ticker for w in ["FMCG", "FOOD", "CONSUMER"]):
            return "FMCG", 0.65

        return "Other / Diversified", 1.00

    def analyze_portfolio(self, raw_holdings: List[PortfolioHolding]) -> PortfolioRiskTelemetry:
        """
        Execute full quantitative portfolio risk analysis and least-risk optimization.
        """
        if not raw_holdings:
            return PortfolioRiskTelemetry(
                total_value=0.0,
                total_invested=0.0,
                total_pnl=0.0,
                total_pnl_pct=0.0,
                holdings_count=0,
                portfolio_weighted_beta=1.0,
                projected_rebalanced_beta=1.0,
                beta_reduction_pct=0.0,
                hhi_concentration_index=0.0,
                concentration_posture="DIVERSIFIED",
                overweight_sectors_count=0,
                sectors=[],
                holdings=[],
                recommendations=[],
                threshold_pct=self.threshold_pct,
            )

        total_value = sum(h.value for h in raw_holdings)
        total_invested = sum(h.invested_value for h in raw_holdings)
        total_pnl = total_value - total_invested
        total_pnl_pct = (total_pnl / total_invested * 100.0) if total_invested > 0 else 0.0

        if total_value <= 0:
            total_value = 1.0  # Prevent division by zero

        # Enrich individual holdings
        enriched_holdings: List[HoldingEnriched] = []
        sector_accumulators: Dict[str, Dict[str, Any]] = {}

        weighted_beta_sum = 0.0

        for h in raw_holdings:
            weight_pct = round((h.value / total_value) * 100.0, 2)
            sector, beta = self.resolve_sector_and_beta(h.ticker, h.tradingsymbol)
            is_overconcentrated_asset = weight_pct > 15.0

            enriched = HoldingEnriched(
                ticker=h.ticker,
                tradingsymbol=h.tradingsymbol,
                exchange=h.exchange,
                quantity=h.quantity,
                average_price=h.average_price,
                last_price=h.last_price,
                value=h.value,
                invested_value=h.invested_value,
                pnl=h.pnl,
                pnl_percentage=h.pnl_percentage,
                day_change_percentage=h.day_change_percentage,
                weight_pct=weight_pct,
                sector=sector,
                beta=beta,
                is_overconcentrated_asset=is_overconcentrated_asset,
            )
            enriched_holdings.append(enriched)
            weighted_beta_sum += (h.value / total_value) * beta

            if sector not in sector_accumulators:
                sector_accumulators[sector] = {
                    "value": 0.0,
                    "assets": [],
                    "baseline_beta": SECTOR_INFO.get(sector, {}).get("baseline_beta", 1.0),
                    "is_defensive": SECTOR_INFO.get(sector, {}).get("is_defensive", False),
                    "target_max_pct": SECTOR_INFO.get(sector, {}).get("target_max_pct", self.threshold_pct),
                }
            sector_accumulators[sector]["value"] += h.value
            sector_accumulators[sector]["assets"].append(h.ticker)

        portfolio_weighted_beta = round(weighted_beta_sum, 2)

        # Build Sector Exposures & Calculate HHI
        sector_exposures: List[SectorExposure] = []
        hhi_sum = 0.0
        overweight_count = 0
        total_trim_value = 0.0

        for sector, data in sector_accumulators.items():
            sec_val = data["value"]
            sec_weight = round((sec_val / total_value) * 100.0, 2)
            hhi_sum += (sec_weight ** 2)

            target_max = min(self.threshold_pct, data["target_max_pct"])
            is_overweight = sec_weight > self.threshold_pct
            excess_pct = round(max(0.0, sec_weight - target_max), 2)
            excess_val = round((excess_pct / 100.0) * total_value, 2)

            if is_overweight:
                overweight_count += 1
                total_trim_value += excess_val

            sector_exposures.append(
                SectorExposure(
                    sector=sector,
                    value=round(sec_val, 2),
                    weight_pct=sec_weight,
                    target_weight_pct=target_max,
                    is_overweight=is_overweight,
                    excess_pct=excess_pct,
                    excess_value=excess_val,
                    baseline_beta=data["baseline_beta"],
                    is_defensive=data["is_defensive"],
                    holdings_count=len(data["assets"]),
                    assets=data["assets"],
                )
            )

        # Sort sectors by weight descending
        sector_exposures.sort(key=lambda s: s.weight_pct, reverse=True)
        hhi_index = round(hhi_sum, 1)

        # Determine concentration posture
        if hhi_index > 2500 or overweight_count >= 2:
            concentration_posture = "HIGH_RISK_CONCENTRATED"
        elif hhi_index > 1800 or overweight_count == 1:
            concentration_posture = "MODERATE"
        else:
            concentration_posture = "DIVERSIFIED"

        # Generate Least-Risk Optimization Recommendations
        recommendations: List[RebalanceRecommendation] = []

        # 1. Trim recommendations for overweight sectors
        for sec in sector_exposures:
            if sec.is_overweight:
                # Find largest holdings in this sector to suggest trimming
                sector_holdings = [h for h in enriched_holdings if h.sector == sec.sector]
                sector_holdings.sort(key=lambda h: h.value, reverse=True)
                top_tickers = [h.ticker for h in sector_holdings[:2]]

                recommendations.append(
                    RebalanceRecommendation(
                        action="TRIM",
                        sector=sec.sector,
                        current_weight_pct=sec.weight_pct,
                        recommended_weight_pct=sec.target_weight_pct,
                        amount_currency=sec.excess_value,
                        proxy_tickers=top_tickers,
                        rationale=(
                            f"{sec.sector} allocation ({sec.weight_pct}%) breaches the institutional {self.threshold_pct}% "
                            f"concentration limit by +{sec.excess_pct}%. Trim ₹{sec.excess_value:,.2f} across top holdings "
                            f"({', '.join(top_tickers)}) to mitigate systemic correlation and sector-specific drawdowns."
                        ),
                    )
                )

        # 2. Defensive reallocation recommendations
        # Allocate trimmed capital into defensive sectors with low beta (FMCG ~0.65, Healthcare ~0.70)
        defensive_sectors = ["FMCG", "Healthcare"]
        if total_trim_value > 0:
            alloc_per_sector = round(total_trim_value / len(defensive_sectors), 2)
            for def_sec in defensive_sectors:
                current_def_sec = next((s for s in sector_exposures if s.sector == def_sec), None)
                cur_weight = current_def_sec.weight_pct if current_def_sec else 0.0
                add_weight_pct = round((alloc_per_sector / total_value) * 100.0, 2)
                new_target = round(cur_weight + add_weight_pct, 2)
                proxies = SECTOR_INFO.get(def_sec, {}).get("defensive_proxies", [])

                recommendations.append(
                    RebalanceRecommendation(
                        action="ALLOCATE_DEFENSIVE",
                        sector=def_sec,
                        current_weight_pct=cur_weight,
                        recommended_weight_pct=new_target,
                        amount_currency=alloc_per_sector,
                        proxy_tickers=proxies,
                        rationale=(
                            f"Deploy ₹{alloc_per_sector:,.2f} into defensive {def_sec} (target: {new_target}%) using "
                            f"low-beta anchors ({', '.join(proxies[:2])}, beta ~{SECTOR_INFO[def_sec]['baseline_beta']}). "
                            f"This acts as a systematic shock absorber and compresses overall portfolio tail risk."
                        ),
                    )
                )
        else:
            recommendations.append(
                RebalanceRecommendation(
                    action="HOLD",
                    sector="All Sectors",
                    current_weight_pct=100.0,
                    recommended_weight_pct=100.0,
                    amount_currency=0.0,
                    proxy_tickers=[],
                    rationale="Portfolio sector allocation conforms to institutional diversification guidelines (no sector exceeds 20%). Maintain current risk posture.",
                )
            )

        # Calculate Projected Rebalanced Beta
        # If overweight capital with beta > 1.15 is moved to defensive sectors with beta ~0.675:
        if total_trim_value > 0:
            trim_weight_fraction = total_trim_value / total_value
            overweight_avg_beta = sum(
                (s.excess_value / total_trim_value) * s.baseline_beta for s in sector_exposures if s.is_overweight
            )
            defensive_avg_beta = (SECTOR_INFO["FMCG"]["baseline_beta"] + SECTOR_INFO["Healthcare"]["baseline_beta"]) / 2.0
            beta_delta = trim_weight_fraction * (overweight_avg_beta - defensive_avg_beta)
            projected_rebalanced_beta = max(0.50, round(portfolio_weighted_beta - beta_delta, 2))
            beta_reduction_pct = round(((portfolio_weighted_beta - projected_rebalanced_beta) / portfolio_weighted_beta) * 100.0, 1)
        else:
            projected_rebalanced_beta = portfolio_weighted_beta
            beta_reduction_pct = 0.0

        return PortfolioRiskTelemetry(
            total_value=round(total_value, 2),
            total_invested=round(total_invested, 2),
            total_pnl=round(total_pnl, 2),
            total_pnl_pct=round(total_pnl_pct, 2),
            holdings_count=len(enriched_holdings),
            portfolio_weighted_beta=portfolio_weighted_beta,
            projected_rebalanced_beta=projected_rebalanced_beta,
            beta_reduction_pct=beta_reduction_pct,
            hhi_concentration_index=hhi_index,
            concentration_posture=concentration_posture,
            overweight_sectors_count=overweight_count,
            sectors=sector_exposures,
            holdings=enriched_holdings,
            recommendations=recommendations,
            threshold_pct=self.threshold_pct,
        )
