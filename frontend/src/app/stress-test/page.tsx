'use client';

import React, { useState, useEffect, useMemo } from 'react';
import Link from 'next/link';
import {
  ShieldAlert,
  AlertTriangle,
  Activity,
  PieChart,
  RotateCcw,
  RefreshCw,
  Flame,
  TrendingDown,
  TrendingUp,
  Percent,
  Clock,
  Sliders,
  DollarSign,
  ChevronDown,
  ChevronRight,
  ShieldCheck,
  Zap,
  HelpCircle,
  ExternalLink,
  Layers,
  Sparkles,
  BarChart3,
  Dices,
  Lock,
  X,
  CheckCircle2,
  AlertCircle,
  Skull,
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
  ReferenceLine,
} from 'recharts';

// ============================================================================
// TYPES
// ============================================================================

interface StressScenario {
  scenario_id: string;
  name: string;
  category: string;
  description: string;
  nifty_benchmark_shock_pct: number;
  vix_spike_pct: number;
  baseline_recovery_months: number;
  historical_precedent?: string;
  bear_catalysts?: string[];
  bull_catalysts?: string[];
  recommended_hedges?: string[];
  sector_shocks?: Record<string, number>;
}

interface StressHoldingImpact {
  ticker: string;
  tradingsymbol: string;
  sector: string;
  quantity: number;
  last_price: number;
  stressed_price: number;
  current_value: number;
  stressed_value: number;
  weight_pct: number;
  shock_loss_inr: number;
  shock_loss_pct: number;
  beta: number;
  vulnerability_rating: 'CRITICAL' | 'HIGH' | 'MODERATE' | 'RESILIENT';
  is_defensive: boolean;
}

interface RecoveryTimeline {
  estimated_months: number;
  range_min_months: number;
  range_max_months: number;
  recovery_cagr_required_pct: number;
  human_readable_timeline: string;
  historical_precedent_note: string;
}

interface AdversarialAdvisory {
  risk_level: 'CRITICAL' | 'SEVERE' | 'ELEVATED' | 'MANAGED';
  cro_verdict: string;
  bear_case_cascade: string;
  bull_case_stabilizer: string;
  actionable_hedges: string[];
}

interface SectorLossBreakdown {
  sector: string;
  current_value: number;
  stressed_value: number;
  loss_inr: number;
  loss_pct: number;
  portfolio_weight_pct: number;
  is_defensive: boolean;
}

interface StressTestResult {
  scenario: StressScenario;
  portfolio_current_value: number;
  portfolio_stressed_value: number;
  drawdown_inr: number;
  drawdown_pct: number;
  weighted_beta: number;
  systematic_loss_inr: number;
  idiosyncratic_loss_inr: number;
  recovery_timeline: RecoveryTimeline;
  worst_hit_holdings: StressHoldingImpact[];
  resilient_holdings: StressHoldingImpact[];
  sector_loss_breakdown: SectorLossBreakdown[];
  adversarial_committee: AdversarialAdvisory;
}

interface MonteCarloHistogramBin {
  bin_label: string;
  range_min_pct: number;
  range_max_pct: number;
  count: number;
  probability_pct: number;
  cumulative_pct: number;
}

interface MonteCarloResult {
  num_simulations: number;
  time_horizon_days: number;
  mean_drawdown_pct: number;
  var_90_pct: number;
  var_95_pct: number;
  var_99_pct: number;
  var_95_inr: number;
  var_99_inr: number;
  cvar_95_pct: number;
  cvar_99_pct: number;
  cvar_95_inr: number;
  cvar_99_inr: number;
  max_tail_drawdown_pct: number;
  max_tail_drawdown_inr: number;
  fat_tail_kurtosis: number;
  distribution_bins: MonteCarloHistogramBin[];
}

// Fallback scenarios if API initial fetch is pending
const FALLBACK_SCENARIOS: StressScenario[] = [
  {
    scenario_id: 'covid_2020',
    name: 'COVID-19 March 2020 Liquidity Crunch',
    category: 'Historical Black Swan',
    description: 'Global pandemic emergency triggers nationwide lock-downs, FII liquidity drainage, and multi-asset circuit down-limits.',
    nifty_benchmark_shock_pct: -38.0,
    vix_spike_pct: 260.0,
    baseline_recovery_months: 7.5,
  },
  {
    scenario_id: 'gfc_2008',
    name: '2008 Global Financial Crisis & Credit Freeze',
    category: 'Historical Black Swan',
    description: 'Subprime mortgage failure and Lehman collapse freeze global interbank credit lines and liquidate emerging market holdings.',
    nifty_benchmark_shock_pct: -52.0,
    vix_spike_pct: 310.0,
    baseline_recovery_months: 24.0,
  },
  {
    scenario_id: 'tech_tightening_2022',
    name: '2022 Fed Rate Tightening & Tech Multiple Compression',
    category: 'Historical Black Swan',
    description: 'Historic inflation forces +450 bps global central bank rate hikes, triggering aggressive multiple contraction in growth platforms.',
    nifty_benchmark_shock_pct: -18.0,
    vix_spike_pct: 75.0,
    baseline_recovery_months: 13.0,
  },
  {
    scenario_id: 'nifty_crash_20',
    name: 'Nifty 50 Sudden -20% Flash Correction',
    category: 'Macro Shock',
    description: 'High-velocity algorithmic stop-loss cascade triggering broad index breakdown and leverage liquidations.',
    nifty_benchmark_shock_pct: -20.0,
    vix_spike_pct: 110.0,
    baseline_recovery_months: 4.5,
  },
  {
    scenario_id: 'crude_spike_130',
    name: 'Crude Oil Surge to $130/bbl (Geopolitical Supply Shock)',
    category: 'Macro Shock',
    description: 'Middle East conflict blocks strategic maritime straits, surging crude oil prices and expanding India CAD.',
    nifty_benchmark_shock_pct: -14.0,
    vix_spike_pct: 90.0,
    baseline_recovery_months: 6.0,
  },
  {
    scenario_id: 'rate_shock_150',
    name: 'Central Bank Emergency Rate Hike (+150 bps)',
    category: 'Macro Shock',
    description: 'Persistent inflation forces RBI to unexpectedly hike benchmark repo rate by 150 bps, flattening credit demand.',
    nifty_benchmark_shock_pct: -12.0,
    vix_spike_pct: 55.0,
    baseline_recovery_months: 5.0,
  },
  {
    scenario_id: 'mid_small_liquidity_freeze',
    name: 'Midcap & Smallcap Severe Liquidity Freeze',
    category: 'Sector-Specific Liquidity Squeeze',
    description: 'Valuation froth warning sparks heavy retail mutual fund redemptions, trapping non-Nifty50 equities in illiquid gap-downs.',
    nifty_benchmark_shock_pct: -7.0,
    vix_spike_pct: 85.0,
    baseline_recovery_months: 8.0,
  },
  {
    scenario_id: 'it_global_recession',
    name: 'Enterprise IT Tech Spend Freeze & Global Slowdown',
    category: 'Sector-Specific Liquidity Squeeze',
    description: 'Synchronized US/European recession drives Fortune 500 tech budget cuts and deal renewal deferrals.',
    nifty_benchmark_shock_pct: -9.5,
    vix_spike_pct: 50.0,
    baseline_recovery_months: 6.5,
  },
];

const API_BASE = 'http://localhost:8000';

// ============================================================================
// UTILITY HELPERS
// ============================================================================

function formatINR(val: number): string {
  const abs = Math.abs(val);
  const formatted = new Intl.NumberFormat('en-IN', {
    maximumFractionDigits: 0,
  }).format(abs);
  return `${val < 0 ? '-' : ''}₹${formatted}`;
}

function formatPct(val: number): string {
  const sign = val > 0 ? '+' : '';
  return `${sign}${val.toFixed(1)}%`;
}

// Custom Recharts Tooltip
interface CustomTooltipProps {
  active?: boolean;
  payload?: Array<{
    payload: MonteCarloHistogramBin;
  }>;
}

const CustomHistogramTooltip: React.FC<CustomTooltipProps> = ({ active, payload }) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    const isTail = data.range_min_pct >= 20;
    return (
      <div className="bg-[#090A0F]/95 border border-[#1E2230] p-3 rounded-lg shadow-xl text-xs font-mono space-y-1">
        <div className="text-white font-bold flex items-center justify-between gap-4">
          <span>Drawdown Range:</span>
          <span className={isTail ? 'text-[#F43F5E]' : 'text-[#10B981]'}>
            {data.bin_label}
          </span>
        </div>
        <div className="flex items-center justify-between gap-4 text-[#94A3B8]">
          <span>Path Count:</span>
          <span className="text-white font-medium">{data.count.toLocaleString()} paths</span>
        </div>
        <div className="flex items-center justify-between gap-4 text-[#94A3B8]">
          <span>Probability:</span>
          <span className="text-[#06B6D4] font-semibold">{data.probability_pct}%</span>
        </div>
        <div className="flex items-center justify-between gap-4 text-[#94A3B8]">
          <span>Cumulative Risk:</span>
          <span className="text-[#F59E0B] font-semibold">{data.cumulative_pct}%</span>
        </div>
      </div>
    );
  }
  return null;
};

// ============================================================================
// MAIN COMPONENT
// ============================================================================

export default function AdversarialStressTestPage() {
  const [scenarios, setScenarios] = useState<StressScenario[]>(FALLBACK_SCENARIOS);
  const [selectedScenarioId, setSelectedScenarioId] = useState<string>('covid_2020');
  const [stressResult, setStressResult] = useState<StressTestResult | null>(null);
  const [mcResult, setMcResult] = useState<MonteCarloResult | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [mcLoading, setMcLoading] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string>('');
  const [brokerStatus, setBrokerStatus] = useState<{ isConnected: boolean; userId?: string }>({
    isConnected: false,
  });
  const [activeTab, setActiveTab] = useState<'holdings' | 'sectors' | 'committee' | 'monte-carlo'>('holdings');
  const [customModalOpen, setCustomModalOpen] = useState<boolean>(false);

  // Custom scenario form state
  const [customBenchmarkShock, setCustomBenchmarkShock] = useState<number>(-25.0);
  const [customVixSpike, setCustomVixSpike] = useState<number>(120.0);
  const [customScenarioName, setCustomScenarioName] = useState<string>('Custom Adversarial Event');

  // Check broker session on mount
  useEffect(() => {
    async function checkBrokerSession() {
      try {
        const res = await fetch(`${API_BASE}/api/portfolio/session/status`);
        if (res.ok) {
          const data = await res.json();
          setBrokerStatus({
            isConnected: Boolean(data.is_authenticated),
            userId: data.user_id,
          });
        }
      } catch (err) {
        console.warn('Broker status check error:', err);
      }
    }
    checkBrokerSession();
  }, []);

  // Fetch scenarios catalogue
  useEffect(() => {
    async function fetchScenarios() {
      try {
        const res = await fetch(`${API_BASE}/api/risk/stress-test/scenarios`);
        if (res.ok) {
          const data = await res.json();
          if (Array.isArray(data) && data.length > 0) {
            setScenarios(data);
          }
        }
      } catch (err) {
        console.warn('Could not fetch scenarios from backend, using fallbacks:', err);
      }
    }
    fetchScenarios();
  }, []);

  // Run stress test simulation when selected scenario changes
  const runSimulation = async (scenarioId: string) => {
    setLoading(true);
    setErrorMsg('');
    try {
      const res = await fetch(`${API_BASE}/api/risk/stress-test?scenario_id=${encodeURIComponent(scenarioId)}`);
      if (!res.ok) {
        throw new Error(`Stress test failed (HTTP ${res.status})`);
      }
      const data: StressTestResult = await res.json();
      setStressResult(data);
    } catch (err: unknown) {
      console.error('Error running stress test:', err);
      const msg = err instanceof Error ? err.message : 'Failed to connect to simulation engine';
      setErrorMsg(msg);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    runSimulation(selectedScenarioId);
  }, [selectedScenarioId]);

  // Run Monte Carlo simulation
  const runMonteCarlo = async () => {
    setMcLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/risk/stress-test/monte-carlo?num_simulations=2500&time_horizon_days=21`);
      if (!res.ok) {
        throw new Error(`Monte Carlo simulation failed (HTTP ${res.status})`);
      }
      const data: MonteCarloResult = await res.json();
      setMcResult(data);
      setActiveTab('monte-carlo');
    } catch (err: unknown) {
      console.error('Monte Carlo error:', err);
      const msg = err instanceof Error ? err.message : 'Monte Carlo execution error';
      setErrorMsg(msg);
    } finally {
      setMcLoading(false);
    }
  };

  // Run custom scenario
  const handleRunCustom = async () => {
    setLoading(true);
    setCustomModalOpen(false);
    try {
      const payload = {
        scenario_id: 'custom',
        custom_params: {
          name: customScenarioName,
          category: 'Custom Adversarial',
          description: `User-defined macro shock with ${customBenchmarkShock}% Nifty correction and ${customVixSpike}% VIX spike.`,
          historical_precedent: 'Synthetic custom stress test parameters',
          nifty_benchmark_shock_pct: customBenchmarkShock,
          vix_spike_pct: customVixSpike,
          baseline_recovery_months: Math.round((Math.abs(customBenchmarkShock) / 5) * 10) / 10,
        },
      };
      const res = await fetch(`${API_BASE}/api/risk/stress-test`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error('Custom simulation failed');
      const data: StressTestResult = await res.json();
      setStressResult(data);
      setSelectedScenarioId('custom');
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Error executing custom shock';
      setErrorMsg(msg);
    } finally {
      setLoading(false);
    }
  };

  const activeScenario = useMemo(() => {
    if (stressResult?.scenario) return stressResult.scenario;
    return scenarios.find((s) => s.scenario_id === selectedScenarioId) || scenarios[0];
  }, [scenarios, selectedScenarioId, stressResult]);

  // Group scenarios by category
  const groupedScenarios = useMemo(() => {
    const groups: Record<string, StressScenario[]> = {};
    scenarios.forEach((sc) => {
      const cat = sc.category || 'Macro Shock';
      if (!groups[cat]) groups[cat] = [];
      groups[cat].push(sc);
    });
    return groups;
  }, [scenarios]);

  return (
    <div className="min-h-screen bg-[#08090D] text-[#E2E8F0] p-4 md:p-6 lg:p-8 space-y-6 max-w-[1600px] mx-auto">
      {/* ==================================================================== */}
      {/* 1. INSTITUTIONAL HEADER & NAVIGATION */}
      {/* ==================================================================== */}
      <header className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-[#1E2230] pb-5">
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-lg bg-[#12141C] border border-[#F43F5E]/40 flex items-center justify-center text-[#F43F5E] shadow-lg shadow-[#F43F5E]/15">
            <Flame className="h-5 w-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-bold tracking-tight text-white">ADVERSARIAL STRESS TESTING</h1>
              <span className="px-2 py-0.5 rounded text-[10px] font-semibold tracking-wider bg-[#F43F5E]/15 text-[#F43F5E] border border-[#F43F5E]/40">
                SHOCK ENGINE
              </span>
              {brokerStatus.isConnected ? (
                <span className="flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-semibold tracking-wider bg-[#10B981]/15 text-[#10B981] border border-[#10B981]/40">
                  <span className="h-1.5 w-1.5 rounded-full bg-[#10B981] animate-pulse"></span>
                  LIVE KITE ({brokerStatus.userId})
                </span>
              ) : (
                <span className="px-2 py-0.5 rounded text-[10px] font-semibold tracking-wider bg-[#F59E0B]/15 text-[#F59E0B] border border-[#F59E0B]/40">
                  DEMO HOLDINGS
                </span>
              )}
            </div>
            <p className="text-xs text-[#94A3B8]">
              Historical Black Swan Emulation • Multi-Scenario Macro Shocks • Fat-Tailed Student-t Monte Carlo VaR
            </p>
          </div>
        </div>

        {/* NAVIGATION TABS */}
        <div className="flex flex-wrap items-center gap-2 text-xs font-mono">
          <Link
            href="/"
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#12141C] border border-[#1E2230] text-[#94A3B8] hover:text-white hover:border-[#334155] transition-colors"
          >
            <Activity className="h-3.5 w-3.5 text-[#10B981]" />
            <span>Risk Committee</span>
          </Link>

          <Link
            href="/portfolio"
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#12141C] border border-[#1E2230] text-[#94A3B8] hover:text-white hover:border-[#334155] transition-colors"
          >
            <PieChart className="h-3.5 w-3.5 text-[#06B6D4]" />
            <span>Portfolio Risk</span>
          </Link>

          <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#F43F5E]/15 border border-[#F43F5E]/40 text-[#F43F5E] font-semibold">
            <Flame className="h-3.5 w-3.5" />
            <span>Stress Testing</span>
          </div>
        </div>
      </header>

      {/* ERROR BANNER */}
      {errorMsg && (
        <section
          role="alert"
          className="flex items-start justify-between gap-3 rounded-xl border border-[#F43F5E]/40 bg-[#F43F5E]/10 p-4 text-xs font-mono"
        >
          <div className="flex items-start gap-3">
            <AlertTriangle className="h-4 w-4 text-[#F43F5E] shrink-0 mt-0.5" />
            <div>
              <div className="font-bold uppercase tracking-wider text-[#F43F5E]">Simulation Warning</div>
              <p className="text-[#CBD5E1]">{errorMsg}</p>
            </div>
          </div>
          <button onClick={() => setErrorMsg('')} className="text-[#94A3B8] hover:text-white">
            <X className="h-4 w-4" />
          </button>
        </section>
      )}

      {/* ==================================================================== */}
      {/* 2. SCENARIO SELECTOR & CONTROL BAR */}
      {/* ==================================================================== */}
      <section className="bg-[#0D0F17] border border-[#1E2230] rounded-xl p-4 md:p-5 space-y-4">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <span className="text-xs uppercase font-mono tracking-wider text-[#94A3B8]">Active Scenario:</span>
              <span className="text-sm font-bold text-white">{activeScenario.name}</span>
              <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-[#1E2230] text-[#CBD5E1] border border-[#334155]">
                {activeScenario.category}
              </span>
            </div>
            <p className="text-xs text-[#94A3B8] max-w-3xl leading-relaxed">
              {activeScenario.description}
            </p>
          </div>

          {/* ACTION BUTTONS */}
          <div className="flex flex-wrap items-center gap-2.5">
            <button
              onClick={() => runSimulation(selectedScenarioId)}
              disabled={loading}
              className="flex items-center gap-1.5 px-3.5 py-2 rounded-lg text-xs font-mono bg-[#12141C] border border-[#1E2230] text-white hover:bg-[#1A1D27] hover:border-[#334155] transition-all disabled:opacity-50"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin text-[#06B6D4]' : 'text-[#94A3B8]'}`} />
              <span>{loading ? 'Simulating...' : 'Re-Run Shock'}</span>
            </button>

            <button
              onClick={runMonteCarlo}
              disabled={mcLoading}
              className="flex items-center gap-1.5 px-3.5 py-2 rounded-lg text-xs font-mono bg-[#8B5CF6]/15 border border-[#8B5CF6]/40 text-[#C4B5FD] hover:bg-[#8B5CF6]/25 transition-all disabled:opacity-50 font-semibold shadow-lg shadow-[#8B5CF6]/10"
            >
              <Dices className={`h-3.5 w-3.5 ${mcLoading ? 'animate-spin' : ''}`} />
              <span>{mcLoading ? 'Running Monte Carlo...' : 'Run Monte Carlo (2500 Paths)'}</span>
            </button>

            <button
              onClick={() => setCustomModalOpen(true)}
              className="flex items-center gap-1.5 px-3.5 py-2 rounded-lg text-xs font-mono bg-[#12141C] border border-[#1E2230] text-[#94A3B8] hover:text-white hover:border-[#334155] transition-all"
            >
              <Sliders className="h-3.5 w-3.5 text-[#F59E0B]" />
              <span>Custom Shock</span>
            </button>
          </div>
        </div>

        {/* SCENARIO SELECTOR CARDS */}
        <div className="pt-2 border-t border-[#1E2230]">
          <div className="text-[11px] font-mono uppercase tracking-wider text-[#94A3B8] mb-2 flex items-center justify-between">
            <span>Select Historical or Macro Stress Scenario</span>
            <span className="text-[10px] text-[#64748B]">Click card to shock portfolio</span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2.5">
            {scenarios.map((sc) => {
              const isSelected = selectedScenarioId === sc.scenario_id;
              const isBlackSwan = sc.category === 'Historical Black Swan';
              return (
                <button
                  key={sc.scenario_id}
                  onClick={() => setSelectedScenarioId(sc.scenario_id)}
                  className={`text-left p-3 rounded-lg border transition-all text-xs font-mono relative overflow-hidden flex flex-col justify-between ${
                    isSelected
                      ? 'bg-[#12141C] border-[#F43F5E] shadow-md shadow-[#F43F5E]/15'
                      : 'bg-[#090A0F]/60 border-[#1E2230] hover:border-[#334155] hover:bg-[#12141C]/50'
                  }`}
                >
                  {isSelected && (
                    <div className="absolute top-0 right-0 w-2 h-2 bg-[#F43F5E] rounded-bl"></div>
                  )}
                  <div className="space-y-1">
                    <div className="flex items-center justify-between gap-1">
                      <span className="text-[10px] font-semibold tracking-wider text-[#94A3B8] uppercase">
                        {sc.category}
                      </span>
                      <span
                        className={`text-[10px] font-bold px-1.5 py-0.2 rounded ${
                          sc.nifty_benchmark_shock_pct <= -30
                            ? 'bg-[#F43F5E]/20 text-[#F43F5E]'
                            : sc.nifty_benchmark_shock_pct <= -15
                            ? 'bg-[#F97316]/20 text-[#F97316]'
                            : 'bg-[#F59E0B]/20 text-[#F59E0B]'
                        }`}
                      >
                        Nifty {sc.nifty_benchmark_shock_pct}%
                      </span>
                    </div>
                    <div className="font-bold text-white text-xs leading-snug line-clamp-2">
                      {sc.name}
                    </div>
                  </div>

                  <div className="mt-2.5 pt-2 border-t border-[#1E2230]/70 flex items-center justify-between text-[10px] text-[#94A3B8]">
                    <span>VIX: +{sc.vix_spike_pct}%</span>
                    <span>Rec: ~{sc.baseline_recovery_months}m</span>
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      </section>

      {/* ==================================================================== */}
      {/* 3. KEY IMPACT KPI CARDS */}
      {/* ==================================================================== */}
      <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* KPI 1: PROJECTED PORTFOLIO VALUE */}
        <div className="bg-[#0D0F17] border border-[#1E2230] rounded-xl p-4 space-y-2 relative overflow-hidden">
          <div className="flex items-center justify-between text-xs font-mono text-[#94A3B8]">
            <span className="uppercase tracking-wider">Projected Portfolio Value</span>
            <Flame className="h-4 w-4 text-[#F43F5E]" />
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-bold font-mono text-white tracking-tight">
              {stressResult ? formatINR(stressResult.portfolio_stressed_value) : '---'}
            </span>
            {stressResult && (
              <span className="text-xs font-mono font-bold text-[#F43F5E] bg-[#F43F5E]/15 px-1.5 py-0.5 rounded">
                -{stressResult.drawdown_pct.toFixed(1)}%
              </span>
            )}
          </div>
          <div className="text-[11px] font-mono text-[#64748B] flex items-center justify-between">
            <span>Pre-Shock Value:</span>
            <span className="text-[#CBD5E1]">
              {stressResult ? formatINR(stressResult.portfolio_current_value) : '---'}
            </span>
          </div>
        </div>

        {/* KPI 2: ABSOLUTE VALUE LOSS */}
        <div className="bg-[#0D0F17] border border-[#1E2230] rounded-xl p-4 space-y-2">
          <div className="flex items-center justify-between text-xs font-mono text-[#94A3B8]">
            <span className="uppercase tracking-wider">Projected Capital Drawdown</span>
            <TrendingDown className="h-4 w-4 text-[#F43F5E]" />
          </div>
          <div className="text-2xl font-bold font-mono text-[#F43F5E] tracking-tight">
            {stressResult ? `-${formatINR(stressResult.drawdown_inr)}` : '---'}
          </div>
          <div className="text-[11px] font-mono text-[#64748B] flex items-center justify-between">
            <span>Systematic Loss:</span>
            <span className="text-[#CBD5E1]">
              {stressResult ? formatINR(stressResult.systematic_loss_inr) : '---'}
            </span>
          </div>
        </div>

        {/* KPI 3: RECOVERY TIMELINE */}
        <div className="bg-[#0D0F17] border border-[#1E2230] rounded-xl p-4 space-y-2">
          <div className="flex items-center justify-between text-xs font-mono text-[#94A3B8]">
            <span className="uppercase tracking-wider">Projected Recovery Time</span>
            <Clock className="h-4 w-4 text-[#06B6D4]" />
          </div>
          <div className="text-2xl font-bold font-mono text-[#06B6D4] tracking-tight">
            {stressResult ? `~${stressResult.recovery_timeline.estimated_months} mos` : '---'}
          </div>
          <div className="text-[11px] font-mono text-[#64748B] flex items-center justify-between">
            <span>Breakeven Return:</span>
            <span className="text-[#10B981] font-semibold">
              {stressResult ? `+${stressResult.recovery_timeline.recovery_cagr_required_pct}%` : '---'}
            </span>
          </div>
        </div>

        {/* KPI 4: SYSTEMATIC BETA & ADVERSARIAL RISK LEVEL */}
        <div className="bg-[#0D0F17] border border-[#1E2230] rounded-xl p-4 space-y-2">
          <div className="flex items-center justify-between text-xs font-mono text-[#94A3B8]">
            <span className="uppercase tracking-wider">Adversarial Risk Level</span>
            <ShieldAlert className="h-4 w-4 text-[#F59E0B]" />
          </div>
          <div className="flex items-baseline gap-2">
            <span
              className={`text-2xl font-bold font-mono tracking-tight ${
                stressResult?.adversarial_committee.risk_level === 'CRITICAL'
                  ? 'text-[#F43F5E]'
                  : stressResult?.adversarial_committee.risk_level === 'SEVERE'
                  ? 'text-[#F97316]'
                  : 'text-[#F59E0B]'
              }`}
            >
              {stressResult?.adversarial_committee.risk_level || 'ANALYZING'}
            </span>
            <span className="text-xs font-mono text-[#94A3B8]">
              (Beta: {stressResult?.weighted_beta.toFixed(2)})
            </span>
          </div>
          <div className="text-[11px] font-mono text-[#64748B] flex items-center justify-between">
            <span>VIX Sensitivity:</span>
            <span className="text-[#F43F5E]">+{activeScenario.vix_spike_pct}%</span>
          </div>
        </div>
      </section>

      {/* ==================================================================== */}
      {/* 4. ADVERSARIAL RISK COMMITTEE SYNTHESIS CARD */}
      {/* ==================================================================== */}
      {stressResult && (
        <section className="bg-[#0D0F17] border border-[#1E2230] rounded-xl p-5 space-y-4">
          <div className="flex items-center justify-between border-b border-[#1E2230] pb-3">
            <div className="flex items-center gap-2">
              <ShieldAlert className="h-5 w-5 text-[#F43F5E]" />
              <h2 className="text-sm font-bold tracking-tight text-white uppercase font-mono">
                ADVERSARIAL RISK COMMITTEE DIRECTIVE & HEDGES
              </h2>
            </div>
            <span
              className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border ${
                stressResult.adversarial_committee.risk_level === 'CRITICAL'
                  ? 'bg-[#F43F5E]/20 text-[#F43F5E] border-[#F43F5E]/40'
                  : 'bg-[#F59E0B]/20 text-[#F59E0B] border-[#F59E0B]/40'
              }`}
            >
              RISK STATUS: {stressResult.adversarial_committee.risk_level}
            </span>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 text-xs font-mono">
            {/* CRO VERDICT */}
            <div className="bg-[#12141C] border border-[#1E2230] rounded-lg p-3.5 space-y-2 lg:col-span-1">
              <div className="flex items-center gap-2 text-[#F43F5E] font-bold uppercase tracking-wider">
                <Skull className="h-4 w-4" />
                <span>Chief Risk Officer Verdict</span>
              </div>
              <p className="text-[#CBD5E1] leading-relaxed">
                {stressResult.adversarial_committee.cro_verdict}
              </p>
            </div>

            {/* BEAR VS BULL CASUALTY ARBITRATION */}
            <div className="bg-[#12141C] border border-[#1E2230] rounded-lg p-3.5 space-y-3 lg:col-span-1">
              <div>
                <div className="flex items-center gap-1.5 text-[#F97316] font-bold uppercase tracking-wider mb-1">
                  <TrendingDown className="h-3.5 w-3.5" />
                  <span>Bear Thesis (Downside Contagion)</span>
                </div>
                <p className="text-[#94A3B8] leading-relaxed text-[11px]">
                  {stressResult.adversarial_committee.bear_case_cascade}
                </p>
              </div>

              <div className="pt-2 border-t border-[#1E2230]">
                <div className="flex items-center gap-1.5 text-[#10B981] font-bold uppercase tracking-wider mb-1">
                  <TrendingUp className="h-3.5 w-3.5" />
                  <span>Bull Thesis (Core Resiliency)</span>
                </div>
                <p className="text-[#94A3B8] leading-relaxed text-[11px]">
                  {stressResult.adversarial_committee.bull_case_stabilizer}
                </p>
              </div>
            </div>

            {/* ACTIONABLE HEDGES */}
            <div className="bg-[#12141C] border border-[#1E2230] rounded-lg p-3.5 space-y-2.5 lg:col-span-1">
              <div className="flex items-center gap-2 text-[#06B6D4] font-bold uppercase tracking-wider">
                <ShieldCheck className="h-4 w-4" />
                <span>Tactical Mitigation & Hedges</span>
              </div>
              <ul className="space-y-2 text-[#CBD5E1]">
                {stressResult.adversarial_committee.actionable_hedges.map((hedge, idx) => (
                  <li key={idx} className="flex items-start gap-2 text-[11px] leading-relaxed">
                    <span className="text-[#06B6D4] font-bold mt-0.5">•</span>
                    <span>{hedge}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </section>
      )}

      {/* ==================================================================== */}
      {/* 5. TABBED DETAIL PANELS: WORST-HIT HOLDINGS / SECTORS / MONTE CARLO */}
      {/* ==================================================================== */}
      <section className="bg-[#0D0F17] border border-[#1E2230] rounded-xl p-5 space-y-5">
        {/* TAB CONTROLS */}
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#1E2230] pb-3">
          <div className="flex items-center gap-1.5">
            <button
              onClick={() => setActiveTab('holdings')}
              className={`px-3 py-1.5 rounded-lg text-xs font-mono transition-all flex items-center gap-1.5 ${
                activeTab === 'holdings'
                  ? 'bg-[#1E2230] text-white font-bold border border-[#334155]'
                  : 'text-[#94A3B8] hover:text-white hover:bg-[#12141C]'
              }`}
            >
              <Layers className="h-3.5 w-3.5 text-[#F43F5E]" />
              <span>Worst-Hit Holdings Breakdown</span>
            </button>

            <button
              onClick={() => setActiveTab('sectors')}
              className={`px-3 py-1.5 rounded-lg text-xs font-mono transition-all flex items-center gap-1.5 ${
                activeTab === 'sectors'
                  ? 'bg-[#1E2230] text-white font-bold border border-[#334155]'
                  : 'text-[#94A3B8] hover:text-white hover:bg-[#12141C]'
              }`}
            >
              <PieChart className="h-3.5 w-3.5 text-[#06B6D4]" />
              <span>Sector Drawdown Attribution</span>
            </button>

            <button
              onClick={() => {
                setActiveTab('monte-carlo');
                if (!mcResult && !mcLoading) {
                  runMonteCarlo();
                }
              }}
              className={`px-3 py-1.5 rounded-lg text-xs font-mono transition-all flex items-center gap-1.5 ${
                activeTab === 'monte-carlo'
                  ? 'bg-[#8B5CF6]/20 text-[#C4B5FD] font-bold border border-[#8B5CF6]/50'
                  : 'text-[#94A3B8] hover:text-[#C4B5FD] hover:bg-[#12141C]'
              }`}
            >
              <Dices className="h-3.5 w-3.5 text-[#8B5CF6]" />
              <span>Monte Carlo Tail Distribution</span>
              {mcResult && (
                <span className="px-1 py-0.2 rounded text-[9px] bg-[#8B5CF6]/30 text-[#C4B5FD]">
                  VaR 95: {mcResult.var_95_pct}%
                </span>
              )}
            </button>
          </div>

          <span className="text-[11px] font-mono text-[#64748B]">
            Scenario: <strong className="text-white">{activeScenario.name}</strong>
          </span>
        </div>

        {/* TAB 1: WORST-HIT HOLDINGS TABLE */}
        {activeTab === 'holdings' && stressResult && (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead>
                <tr className="border-b border-[#1E2230] text-[#94A3B8] uppercase tracking-wider text-[11px]">
                  <th className="py-2.5 px-3">Asset</th>
                  <th className="py-2.5 px-3">Sector</th>
                  <th className="py-2.5 px-3 text-right">Current Price</th>
                  <th className="py-2.5 px-3 text-right">Stressed Price</th>
                  <th className="py-2.5 px-3 text-right">Pre-Shock Value</th>
                  <th className="py-2.5 px-3 text-right">Stressed Value</th>
                  <th className="py-2.5 px-3 text-right">Shock Drawdown</th>
                  <th className="py-2.5 px-3 text-right">Loss (INR)</th>
                  <th className="py-2.5 px-3 text-center">Beta</th>
                  <th className="py-2.5 px-3 text-center">Vulnerability</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#1E2230]/60">
                {stressResult.worst_hit_holdings.map((h) => {
                  const isPositive = h.shock_loss_pct >= 0;
                  return (
                    <tr key={h.ticker} className="hover:bg-[#12141C]/50 transition-colors">
                      <td className="py-3 px-3">
                        <div className="font-bold text-white">{h.tradingsymbol}</div>
                        <div className="text-[10px] text-[#64748B]">{h.quantity} shares</div>
                      </td>
                      <td className="py-3 px-3">
                        <span className="text-[#CBD5E1]">{h.sector}</span>
                        {h.is_defensive && (
                          <span className="ml-1.5 px-1 py-0.2 rounded text-[9px] bg-[#10B981]/15 text-[#10B981] border border-[#10B981]/30">
                            DEF
                          </span>
                        )}
                      </td>
                      <td className="py-3 px-3 text-right text-white">
                        ₹{h.last_price.toFixed(2)}
                      </td>
                      <td className="py-3 px-3 text-right font-semibold text-[#CBD5E1]">
                        ₹{h.stressed_price.toFixed(2)}
                      </td>
                      <td className="py-3 px-3 text-right text-[#94A3B8]">
                        {formatINR(h.current_value)}
                      </td>
                      <td className="py-3 px-3 text-right font-medium text-white">
                        {formatINR(h.stressed_value)}
                      </td>
                      <td className="py-3 px-3 text-right font-bold">
                        <span className={isPositive ? 'text-[#10B981]' : 'text-[#F43F5E]'}>
                          {formatPct(h.shock_loss_pct)}
                        </span>
                      </td>
                      <td className="py-3 px-3 text-right font-bold text-[#F43F5E]">
                        -{formatINR(h.shock_loss_inr)}
                      </td>
                      <td className="py-3 px-3 text-center text-[#CBD5E1]">
                        {h.beta.toFixed(2)}
                      </td>
                      <td className="py-3 px-3 text-center">
                        <span
                          className={`px-2 py-0.5 rounded text-[10px] font-bold border ${
                            h.vulnerability_rating === 'CRITICAL'
                              ? 'bg-[#F43F5E]/20 text-[#F43F5E] border-[#F43F5E]/40'
                              : h.vulnerability_rating === 'HIGH'
                              ? 'bg-[#F97316]/20 text-[#F97316] border-[#F97316]/40'
                              : h.vulnerability_rating === 'MODERATE'
                              ? 'bg-[#F59E0B]/20 text-[#F59E0B] border-[#F59E0B]/40'
                              : 'bg-[#10B981]/20 text-[#10B981] border-[#10B981]/40'
                          }`}
                        >
                          {h.vulnerability_rating}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        {/* TAB 2: SECTOR DRAWDOWN ATTRIBUTION */}
        {activeTab === 'sectors' && stressResult && (
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {stressResult.sector_loss_breakdown.map((sec) => (
                <div
                  key={sec.sector}
                  className="bg-[#12141C] border border-[#1E2230] rounded-lg p-3.5 space-y-2 font-mono text-xs"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-white">{sec.sector}</span>
                    <span
                      className={`text-[10px] font-semibold px-1.5 py-0.2 rounded ${
                        sec.is_defensive
                          ? 'bg-[#10B981]/15 text-[#10B981] border border-[#10B981]/30'
                          : 'bg-[#06B6D4]/15 text-[#06B6D4] border border-[#06B6D4]/30'
                      }`}
                    >
                      {sec.is_defensive ? 'DEFENSIVE' : 'CYCLICAL'}
                    </span>
                  </div>

                  <div className="flex items-baseline justify-between">
                    <span className="text-[#94A3B8]">Loss Contribution:</span>
                    <span className="text-[#F43F5E] font-bold">-{formatINR(sec.loss_inr)}</span>
                  </div>

                  <div className="flex items-baseline justify-between text-[11px] text-[#64748B]">
                    <span>Drawdown %:</span>
                    <span className="text-[#CBD5E1] font-semibold">{sec.loss_pct.toFixed(1)}%</span>
                  </div>

                  {/* Drawdown progress visualizer */}
                  <div className="w-full bg-[#1E2230] h-1.5 rounded-full overflow-hidden">
                    <div
                      className={`h-full ${
                        sec.loss_pct >= 30 ? 'bg-[#F43F5E]' : sec.loss_pct >= 15 ? 'bg-[#F97316]' : 'bg-[#F59E0B]'
                      }`}
                      style={{ width: `${Math.min(100, Math.max(5, Math.abs(sec.loss_pct)))}%` }}
                    ></div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* TAB 3: PROBABILISTIC MONTE CARLO DRAWDOWN SIMULATION */}
        {activeTab === 'monte-carlo' && (
          <div className="space-y-5">
            {mcLoading && (
              <div className="flex items-center justify-center p-12 text-xs font-mono text-[#94A3B8] gap-3">
                <Dices className="h-5 w-5 animate-spin text-[#8B5CF6]" />
                <span>Simulating 2,500 fat-tailed Student-t market paths...</span>
              </div>
            )}

            {!mcLoading && !mcResult && (
              <div className="text-center p-10 bg-[#12141C] border border-[#1E2230] rounded-xl space-y-3 font-mono text-xs">
                <Dices className="h-8 w-8 text-[#8B5CF6] mx-auto opacity-70" />
                <div className="font-bold text-white">Probabilistic Monte Carlo Engine Not Yet Run</div>
                <p className="text-[#94A3B8] max-w-md mx-auto">
                  Execute 2,500 randomized fat-tailed simulation paths to reveal Value at Risk (VaR),
                  Conditional VaR (Expected Shortfall), and probability distribution percentiles.
                </p>
                <button
                  onClick={runMonteCarlo}
                  className="px-4 py-2 bg-[#8B5CF6] text-white rounded-lg font-bold hover:bg-[#7C3AED] transition-colors inline-flex items-center gap-2"
                >
                  <Zap className="h-3.5 w-3.5" />
                  <span>Execute 2,500 Monte Carlo Paths</span>
                </button>
              </div>
            )}

            {!mcLoading && mcResult && (
              <div className="space-y-5">
                {/* MONTE CARLO KPI TILES */}
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3 font-mono">
                  <div className="bg-[#12141C] border border-[#1E2230] rounded-lg p-3 space-y-1">
                    <span className="text-[10px] uppercase text-[#94A3B8] block">Mean Drawdown</span>
                    <span className="text-lg font-bold text-white">{mcResult.mean_drawdown_pct}%</span>
                  </div>

                  <div className="bg-[#12141C] border border-[#1E2230] rounded-lg p-3 space-y-1">
                    <span className="text-[10px] uppercase text-[#F59E0B] block">VaR 90%</span>
                    <span className="text-lg font-bold text-[#F59E0B]">{mcResult.var_90_pct}%</span>
                  </div>

                  <div className="bg-[#12141C] border border-[#1E2230] rounded-lg p-3 space-y-1">
                    <span className="text-[10px] uppercase text-[#F97316] block">VaR 95%</span>
                    <span className="text-lg font-bold text-[#F97316]">{mcResult.var_95_pct}%</span>
                    <span className="text-[10px] text-[#64748B] block">({formatINR(mcResult.var_95_inr)})</span>
                  </div>

                  <div className="bg-[#12141C] border border-[#1E2230] rounded-lg p-3 space-y-1">
                    <span className="text-[10px] uppercase text-[#F43F5E] block">VaR 99% (Extreme)</span>
                    <span className="text-lg font-bold text-[#F43F5E]">{mcResult.var_99_pct}%</span>
                    <span className="text-[10px] text-[#64748B] block">({formatINR(mcResult.var_99_inr)})</span>
                  </div>

                  <div className="bg-[#12141C] border border-[#1E2230] rounded-lg p-3 space-y-1">
                    <span className="text-[10px] uppercase text-[#C4B5FD] block">CVaR 95% (Expected Shortfall)</span>
                    <span className="text-lg font-bold text-[#C4B5FD]">{mcResult.cvar_95_pct}%</span>
                    <span className="text-[10px] text-[#64748B] block">({formatINR(mcResult.cvar_95_inr)})</span>
                  </div>

                  <div className="bg-[#12141C] border border-[#1E2230] rounded-lg p-3 space-y-1">
                    <span className="text-[10px] uppercase text-[#F43F5E] block">Max Tail Loss</span>
                    <span className="text-lg font-bold text-[#F43F5E]">{mcResult.max_tail_drawdown_pct}%</span>
                    <span className="text-[10px] text-[#64748B] block">Kurtosis: {mcResult.fat_tail_kurtosis}</span>
                  </div>
                </div>

                {/* DISTRIBUTION HISTOGRAM CHART */}
                <div className="bg-[#12141C] border border-[#1E2230] rounded-xl p-4 space-y-3">
                  <div className="flex items-center justify-between text-xs font-mono">
                    <div className="flex items-center gap-2">
                      <BarChart3 className="h-4 w-4 text-[#8B5CF6]" />
                      <span className="font-bold text-white uppercase tracking-wider">
                        Probabilistic Drawdown Distribution (21-Day Horizon)
                      </span>
                    </div>
                    <div className="flex items-center gap-4 text-[10px] text-[#94A3B8]">
                      <span className="flex items-center gap-1">
                        <span className="h-2 w-2 rounded-full bg-[#10B981]"></span>
                        <span>Normal (&lt;10%)</span>
                      </span>
                      <span className="flex items-center gap-1">
                        <span className="h-2 w-2 rounded-full bg-[#F59E0B]"></span>
                        <span>Elevated (10-20%)</span>
                      </span>
                      <span className="flex items-center gap-1">
                        <span className="h-2 w-2 rounded-full bg-[#F43F5E]"></span>
                        <span>Tail Risk (&gt;20%)</span>
                      </span>
                    </div>
                  </div>

                  <div className="h-72 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={mcResult.distribution_bins}
                        margin={{ top: 10, right: 10, left: -20, bottom: 25 }}
                      >
                        <XAxis
                          dataKey="bin_label"
                          tick={{ fill: '#94A3B8', fontSize: 10, fontFamily: 'monospace' }}
                          angle={-30}
                          textAnchor="end"
                          interval={0}
                          stroke="#1E2230"
                        />
                        <YAxis
                          tick={{ fill: '#94A3B8', fontSize: 10, fontFamily: 'monospace' }}
                          stroke="#1E2230"
                          tickFormatter={(v) => `${v}`}
                        />
                        <Tooltip content={<CustomHistogramTooltip />} />
                        <ReferenceLine
                          x={
                            mcResult.distribution_bins.find(
                              (b) => b.range_min_pct <= mcResult.var_95_pct && mcResult.var_95_pct < b.range_max_pct
                            )?.bin_label
                          }
                          stroke="#F97316"
                          strokeDasharray="3 3"
                          label={{
                            value: 'VaR 95%',
                            fill: '#F97316',
                            fontSize: 10,
                            position: 'top',
                            fontFamily: 'monospace',
                          }}
                        />
                        <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                          {mcResult.distribution_bins.map((entry, index) => {
                            let fillColor = '#10B981';
                            if (entry.range_min_pct >= 20.0) {
                              fillColor = '#F43F5E';
                            } else if (entry.range_min_pct >= 10.0) {
                              fillColor = '#F59E0B';
                            }
                            return <Cell key={`cell-${index}`} fill={fillColor} />;
                          })}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>

                  <div className="text-[11px] font-mono text-[#64748B] text-center pt-2 border-t border-[#1E2230]">
                    Simulation calibrated using Student-t fat-tailed returns (\u03bd = 4 degrees of freedom) capturing extreme kurtosis and Black Swan jump shocks.
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </section>

      {/* ==================================================================== */}
      {/* 6. CUSTOM ADVERSARIAL SHOCK CONFIGURATION MODAL */}
      {/* ==================================================================== */}
      {customModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in duration-200">
          <div className="bg-[#0D0F17] border border-[#1E2230] rounded-2xl max-w-lg w-full p-6 space-y-5 shadow-2xl font-mono text-xs">
            <div className="flex items-center justify-between border-b border-[#1E2230] pb-3">
              <div className="flex items-center gap-2">
                <Sliders className="h-4 w-4 text-[#F59E0B]" />
                <h3 className="font-bold text-white text-sm">CONFIGURE CUSTOM ADVERSARIAL SHOCK</h3>
              </div>
              <button
                onClick={() => setCustomModalOpen(false)}
                className="text-[#94A3B8] hover:text-white"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            <div className="space-y-4">
              <div>
                <label className="block text-[#94A3B8] mb-1 uppercase tracking-wider text-[11px]">
                  Scenario Title
                </label>
                <input
                  type="text"
                  value={customScenarioName}
                  onChange={(e) => setCustomScenarioName(e.target.value)}
                  className="w-full bg-[#12141C] border border-[#1E2230] rounded-lg px-3 py-2 text-white focus:outline-none focus:border-[#F59E0B]"
                />
              </div>

              <div>
                <div className="flex items-center justify-between mb-1">
                  <label className="text-[#94A3B8] uppercase tracking-wider text-[11px]">
                    Nifty 50 Benchmark Shock (%)
                  </label>
                  <span className="text-[#F43F5E] font-bold">{customBenchmarkShock}%</span>
                </div>
                <input
                  type="range"
                  min="-60"
                  max="-5"
                  step="1"
                  value={customBenchmarkShock}
                  onChange={(e) => setCustomBenchmarkShock(parseFloat(e.target.value))}
                  className="w-full accent-[#F43F5E]"
                />
                <div className="flex justify-between text-[10px] text-[#64748B]">
                  <span>-60% (Depression)</span>
                  <span>-30% (Severe)</span>
                  <span>-5% (Mild)</span>
                </div>
              </div>

              <div>
                <div className="flex items-center justify-between mb-1">
                  <label className="text-[#94A3B8] uppercase tracking-wider text-[11px]">
                    India VIX Volatility Spike (%)
                  </label>
                  <span className="text-[#F59E0B] font-bold">+{customVixSpike}%</span>
                </div>
                <input
                  type="range"
                  min="20"
                  max="350"
                  step="10"
                  value={customVixSpike}
                  onChange={(e) => setCustomVixSpike(parseFloat(e.target.value))}
                  className="w-full accent-[#F59E0B]"
                />
                <div className="flex justify-between text-[10px] text-[#64748B]">
                  <span>+20%</span>
                  <span>+150%</span>
                  <span>+350% (Panicked)</span>
                </div>
              </div>
            </div>

            <div className="flex items-center justify-end gap-3 pt-3 border-t border-[#1E2230]">
              <button
                onClick={() => setCustomModalOpen(false)}
                className="px-4 py-2 rounded-lg bg-[#12141C] text-[#94A3B8] hover:text-white border border-[#1E2230]"
              >
                Cancel
              </button>
              <button
                onClick={handleRunCustom}
                className="px-4 py-2 rounded-lg bg-[#F43F5E] text-white font-bold hover:bg-[#E11D48] transition-colors"
              >
                Execute Custom Shock
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
