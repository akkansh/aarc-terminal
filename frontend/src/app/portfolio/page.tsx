'use client';

import React, { useState, useEffect, useRef } from 'react';
import Link from 'next/link';
import {
  TrendingUp,
  TrendingDown,
  ShieldAlert,
  AlertTriangle,
  Activity,
  CheckCircle2,
  PieChart,
  Sliders,
  ExternalLink,
  RotateCcw,
  RefreshCw,
  Key,
  Layers,
  ArrowRight,
  ShieldCheck,
  Percent,
  DollarSign,
  Briefcase,
  ChevronRight,
  LogOut,
  Settings,
  X,
  Lock,
  Sparkles,
  Flame,
} from 'lucide-react';

interface HoldingEnriched {
  ticker: string;
  tradingsymbol: string;
  exchange: string;
  quantity: number;
  average_price: number;
  last_price: number;
  value: number;
  invested_value: number;
  pnl: number;
  pnl_percentage: number;
  day_change_percentage: number;
  weight_pct: number;
  sector: string;
  beta: number;
  is_overconcentrated_asset: boolean;
}

interface SectorExposure {
  sector: string;
  value: number;
  weight_pct: number;
  target_weight_pct: number;
  is_overweight: boolean;
  excess_pct: number;
  excess_value: number;
  baseline_beta: number;
  is_defensive: boolean;
  holdings_count: number;
  assets: string[];
}

interface RebalanceRecommendation {
  action: 'TRIM' | 'ALLOCATE_DEFENSIVE' | 'HOLD';
  sector: string;
  current_weight_pct: number;
  recommended_weight_pct: number;
  amount_currency: number;
  proxy_tickers: string[];
  rationale: string;
}

interface PortfolioRiskTelemetry {
  total_value: number;
  total_invested: number;
  total_pnl: number;
  total_pnl_pct: number;
  holdings_count: number;
  portfolio_weighted_beta: number;
  projected_rebalanced_beta: number;
  beta_reduction_pct: number;
  hhi_concentration_index: number;
  concentration_posture: 'DIVERSIFIED' | 'MODERATE' | 'HIGH_RISK_CONCENTRATED';
  overweight_sectors_count: number;
  sectors: SectorExposure[];
  holdings: HoldingEnriched[];
  recommendations: RebalanceRecommendation[];
  threshold_pct: number;
}

interface BrokerSessionStatus {
  status: 'connected' | 'disconnected' | 'error';
  is_authenticated: boolean;
  user_id?: string;
  user_name?: string;
  broker: string;
  has_credentials: boolean;
  api_key_masked?: string;
}

const SECTOR_COLORS: Record<string, string> = {
  Financials: '#3B82F6', // Blue
  'Information Technology': '#06B6D4', // Cyan
  Energy: '#F59E0B', // Amber
  'Consumer Services': '#EC4899', // Pink
  Automobile: '#8B5CF6', // Purple
  'Metals & Mining': '#E11D48', // Rose
  Telecommunication: '#6366F1', // Indigo
  'Industrials & Power': '#14B8A6', // Teal
  FMCG: '#10B981', // Emerald (Defensive)
  Healthcare: '#22C55E', // Green (Defensive)
  Utilities: '#84CC16', // Lime (Defensive)
  'Other / Diversified': '#94A3B8', // Slate
};

export default function PortfolioRiskPage() {
  const [telemetry, setTelemetry] = useState<PortfolioRiskTelemetry | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [loadingStage, setLoadingStage] = useState<string>('Initializing portfolio engine...');
  const [errorMsg, setErrorMsg] = useState<string>('');
  const [successBanner, setSuccessBanner] = useState<string>('');
  
  // Session & Auth state
  const [sessionStatus, setSessionStatus] = useState<BrokerSessionStatus | null>(null);
  const [portfolioMode, setPortfolioMode] = useState<'demo' | 'live'>('demo');
  const [settingsModalOpen, setSettingsModalOpen] = useState<boolean>(false);

  // Broker Credentials configuration
  const [apiKeyInput, setApiKeyInput] = useState<string>('');
  const [apiSecretInput, setApiSecretInput] = useState<string>('');
  const [thresholdPct, setThresholdPct] = useState<number>(20.0);
  const [sectorFilter, setSectorFilter] = useState<string>('ALL');

  const oauthHandledRef = useRef(false);

  // 1. Initial Load & OAuth URL Parameter Interception
  useEffect(() => {
    if (typeof window === 'undefined' || oauthHandledRef.current) return;
    oauthHandledRef.current = true;

    const urlParams = new URLSearchParams(window.location.search);
    const requestToken = urlParams.get('request_token');
    const authStatus = urlParams.get('status');
    const oauthFlag = urlParams.get('oauth');
    const successFlag = urlParams.get('success');
    const tokenParam = urlParams.get('token') || urlParams.get('access_token');
    const errorParam = urlParams.get('error_msg') || urlParams.get('error');

    // Case A: Redirected back from Zerodha with request_token
    if (requestToken && authStatus !== 'cancelled') {
      handleOAuthTokenExchange(requestToken);
      return;
    }

    // Case B: Cancelled authorization
    if (authStatus === 'cancelled') {
      setErrorMsg('Zerodha Kite authorization was cancelled. You can retry or use the Demo Portfolio.');
      window.history.replaceState({}, '', window.location.pathname);
      checkSessionAndLoad();
      return;
    }

    // Case C: Returned with error message
    if (authStatus === 'error' || errorParam) {
      setErrorMsg(errorParam || 'Zerodha Kite authorization encountered an error. Please retry.');
      window.history.replaceState({}, '', window.location.pathname);
      checkSessionAndLoad();
      return;
    }

    // Case D: Returned from FastAPI backend HTTP 307 redirect (?token=...&success=true)
    if (successFlag === 'true' || tokenParam || oauthFlag === 'success') {
      const userId = urlParams.get('user_id');
      setSuccessBanner(
        userId
          ? `Zerodha Kite connected for account ${userId}. Live holdings synced.`
          : 'Zerodha Kite Connect session active. Live holdings synced.'
      );
      window.history.replaceState({}, '', window.location.pathname);
    }
    checkSessionAndLoad();
  }, []);

  // 2. Automated OAuth Token Exchange & Instant Holdings Fetch
  const handleOAuthTokenExchange = async (requestToken: string) => {
    setLoading(true);
    setLoadingStage('Exchanging Zerodha Kite OAuth request token for live session...');
    setErrorMsg('');
    setSuccessBanner('');

    try {
      const res = await fetch('http://localhost:8000/api/portfolio/oauth/callback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          request_token: requestToken,
          threshold_pct: thresholdPct,
        }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `OAuth exchange failed with HTTP ${res.status}`);
      }

      const data = await res.json();
      setTelemetry(data.telemetry);
      setSessionStatus({
        status: 'connected',
        is_authenticated: true,
        user_id: data.user_id,
        user_name: data.user_name,
        broker: data.broker,
        has_credentials: true,
      });
      setPortfolioMode('live');
      setSuccessBanner(`Authenticated as ${data.user_id || 'Zerodha User'} (${data.user_name || 'Kite Client'}). Live holdings loaded.`);

      // Clean URL query params so request_token is never dangling
      window.history.replaceState({}, '', window.location.pathname);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setErrorMsg(`OAuth Token Exchange Error: ${msg}`);
      window.history.replaceState({}, '', window.location.pathname);
      // Fallback to demo portfolio so user can still interact
      fetchTelemetry('demo');
    } finally {
      setLoading(false);
    }
  };

  // 3. Check Session and Load Initial Telemetry
  const checkSessionAndLoad = async () => {
    setLoading(true);
    setLoadingStage('Connecting to portfolio risk backend...');
    try {
      const statusRes = await fetch('http://localhost:8000/api/portfolio/session/status');
      if (statusRes.ok) {
        const statusData: BrokerSessionStatus = await statusRes.json();
        setSessionStatus(statusData);

        if (statusData.is_authenticated) {
          setPortfolioMode('live');
          setLoadingStage('Ingesting live Zerodha holdings and calculating risk...');
          await fetchTelemetry('live');
          return;
        }
      }
      // If not authenticated, default to Demo Mode
      setPortfolioMode('demo');
      setLoadingStage('Loading institutional demo portfolio model...');
      await fetchTelemetry('demo');
    } catch {
      setPortfolioMode('demo');
      await fetchTelemetry('demo');
    } finally {
      setLoading(false);
    }
  };

  // 4. Fetch Telemetry Routine (Live or Demo)
  const fetchTelemetry = async (mode: 'demo' | 'live' = portfolioMode) => {
    setLoading(true);
    setErrorMsg('');
    try {
      if (mode === 'demo') {
        const res = await fetch(`http://localhost:8000/api/portfolio/demo?threshold_pct=${thresholdPct}`);
        if (!res.ok) throw new Error(`Demo fetch failed with HTTP ${res.status}`);
        const data: PortfolioRiskTelemetry = await res.json();
        setTelemetry(data);
      } else {
        const res = await fetch(`http://localhost:8000/api/portfolio/holdings?threshold_pct=${thresholdPct}`);
        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData.detail || `Live holdings fetch failed with status ${res.status}`);
        }
        const data: PortfolioRiskTelemetry = await res.json();
        setTelemetry(data);
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setErrorMsg(msg);
      // If live fetch fails (e.g. token expired), suggest switching to demo
      if (mode === 'live') {
        setPortfolioMode('demo');
        const demoRes = await fetch(`http://localhost:8000/api/portfolio/demo?threshold_pct=${thresholdPct}`);
        if (demoRes.ok) setTelemetry(await demoRes.json());
      }
    } finally {
      setLoading(false);
    }
  };

  // 5. Seamless 1-Click Kite Connect Button Action
  const handleLaunchKiteLogin = async () => {
    setErrorMsg('');
    try {
      // First check if API key is already configured on backend
      const res = await fetch('http://localhost:8000/api/portfolio/login-url');
      if (res.ok) {
        const data = await res.json();
        if (data.login_url) {
          // 1-Click redirect to Zerodha OAuth authorization
          window.location.href = data.login_url;
          return;
        }
      }
      // If credentials missing, prompt user with configuration modal
      setSettingsModalOpen(true);
    } catch {
      setSettingsModalOpen(true);
    }
  };

  // 6. Save Credentials & Immediately Launch Login
  const handleSaveCredentialsAndLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!apiKeyInput.trim() || !apiSecretInput.trim()) {
      setErrorMsg('Please supply both Zerodha Kite API Key and API Secret.');
      return;
    }

    setLoading(true);
    setLoadingStage('Saving broker credentials and generating OAuth authorization URL...');
    setErrorMsg('');

    try {
      const credRes = await fetch('http://localhost:8000/api/portfolio/credentials', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          api_key: apiKeyInput.trim(),
          api_secret: apiSecretInput.trim(),
          redirect_uri: 'http://localhost:3000/portfolio',
        }),
      });

      if (!credRes.ok) throw new Error('Failed to save Kite credentials.');

      const urlRes = await fetch(`http://localhost:8000/api/portfolio/login-url?api_key=${encodeURIComponent(apiKeyInput.trim())}`);
      const urlData = await urlRes.json();

      if (urlData.login_url) {
        setSettingsModalOpen(false);
        // Direct seamless redirect to Zerodha
        window.location.href = urlData.login_url;
      } else {
        throw new Error(urlData.detail || 'Failed to construct Kite login URL.');
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setErrorMsg(msg);
      setLoading(false);
    }
  };

  // 7. Disconnect Broker Session
  const handleDisconnect = async () => {
    try {
      await fetch('http://localhost:8000/api/portfolio/disconnect', { method: 'POST' });
      setSessionStatus(null);
      setPortfolioMode('demo');
      setSuccessBanner('Disconnected from Zerodha Kite. Switched to simulated demo portfolio.');
      fetchTelemetry('demo');
    } catch {
      setPortfolioMode('demo');
      fetchTelemetry('demo');
    }
  };

  const filteredHoldings = telemetry
    ? sectorFilter === 'ALL'
      ? telemetry.holdings
      : telemetry.holdings.filter((h) => h.sector === sectorFilter)
    : [];

  return (
    <div className="min-h-screen bg-[#090A0F] text-[#E2E8F0] p-4 md:p-6 space-y-6 font-sans">
      {/* INSTITUTIONAL HEADER & NAVIGATION */}
      <header className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-[#1E2230] pb-4">
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-lg bg-[#12141C] border border-[#1E2230] flex items-center justify-center text-[#10B981] shadow-lg shadow-[#10B981]/10">
            <PieChart className="h-5 w-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-bold tracking-tight text-white">PORTFOLIO RISK & DIVERSIFICATION</h1>
              <span className="px-2 py-0.5 rounded text-[10px] font-semibold tracking-wider bg-[#06B6D4]/10 text-[#06B6D4] border border-[#06B6D4]/30">
                LEAST-RISK MODEL
              </span>
            </div>
            <p className="text-xs text-[#94A3B8]">
              Automated Zerodha Kite Gateway • Quantitative Sector Concentration • Least-Risk Systematic Rebalancing
            </p>
          </div>
        </div>

        {/* NAVIGATION PILLS */}
        <div className="flex flex-wrap items-center gap-2">
          <Link
            href="/"
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-mono bg-[#12141C] border border-[#1E2230] text-[#94A3B8] hover:text-white hover:border-[#334155] transition-colors"
          >
            <Activity className="h-3.5 w-3.5 text-[#10B981]" />
            <span>Risk Committee Terminal</span>
          </Link>

          <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-mono bg-[#10B981]/15 border border-[#10B981]/40 text-[#10B981] font-semibold">
            <PieChart className="h-3.5 w-3.5" />
            <span>Portfolio Risk Dashboard</span>
          </div>

          <Link
            href="/stress-test"
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-mono bg-[#12141C] border border-[#1E2230] text-[#94A3B8] hover:text-white hover:border-[#334155] transition-colors"
          >
            <Flame className="h-3.5 w-3.5 text-[#F43F5E]" />
            <span>Stress Testing</span>
            <span className="px-1 py-0.2 rounded text-[9px] bg-[#F43F5E]/20 text-[#F43F5E] font-bold">
              NEW
            </span>
          </Link>
        </div>
      </header>

      {/* SUCCESS / ERROR NOTIFICATIONS */}
      {successBanner && (
        <section
          role="status"
          className="flex items-center justify-between gap-3 rounded-xl border border-[#10B981]/40 bg-[#10B981]/10 p-3.5 text-xs font-mono text-[#10B981]"
        >
          <div className="flex items-center gap-2">
            <CheckCircle2 className="h-4 w-4 shrink-0 text-[#10B981]" />
            <span>{successBanner}</span>
          </div>
          <button onClick={() => setSuccessBanner('')} className="text-[#94A3B8] hover:text-white">
            <X className="h-4 w-4" />
          </button>
        </section>
      )}

      {errorMsg && (
        <section
          role="alert"
          className="flex items-start justify-between gap-3 rounded-xl border border-[#F43F5E]/40 bg-[#F43F5E]/10 p-4 text-xs font-mono"
        >
          <div className="flex items-start gap-3">
            <AlertTriangle className="h-4 w-4 text-[#F43F5E] shrink-0 mt-0.5" />
            <div className="space-y-1">
              <div className="font-bold uppercase tracking-wider text-[#F43F5E]">Broker Integration Notice</div>
              <p className="text-[#E2E8F0]">{errorMsg}</p>
            </div>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            <button
              onClick={() => {
                setPortfolioMode('demo');
                fetchTelemetry('demo');
              }}
              className="px-3 py-1 rounded bg-[#12141C] hover:bg-[#1E2230] text-white font-medium border border-[#334155] text-xs"
            >
              Load Demo Mode
            </button>
            <button
              onClick={handleLaunchKiteLogin}
              className="px-3 py-1 rounded bg-[#06B6D4] hover:bg-[#0891B2] text-black font-semibold text-xs"
            >
              Retry Auth
            </button>
          </div>
        </section>
      )}

      {/* SEAMLESS 1-CLICK BROKER AUTHENTICATION CARD */}
      <section className="bg-[#12141C] border border-[#1E2230] rounded-xl p-4 sm:p-5 space-y-4">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div className="flex flex-wrap items-center gap-3">
            <div className="flex items-center gap-2">
              <span className="text-xs font-mono font-semibold uppercase tracking-wider text-white">
                BROKER GATEWAY:
              </span>

              {sessionStatus?.is_authenticated && portfolioMode === 'live' ? (
                <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-[#10B981]/15 border border-[#10B981]/40 text-[#10B981] text-xs font-mono font-semibold">
                  <span className="h-2 w-2 rounded-full bg-[#10B981] animate-pulse" />
                  <span>CONNECTED: ZERODHA KITE</span>
                  {sessionStatus.user_id && (
                    <span className="text-white/80 font-normal">({sessionStatus.user_id})</span>
                  )}
                </div>
              ) : (
                <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-[#F59E0B]/15 border border-[#F59E0B]/40 text-[#F59E0B] text-xs font-mono">
                  <span className="h-2 w-2 rounded-full bg-[#F59E0B]" />
                  <span>DEMO MODE: SIMULATED INSTITUTIONAL PORTFOLIO</span>
                </div>
              )}
            </div>

            {/* QUICK TOGGLE BUTTONS */}
            <div className="flex items-center bg-[#090A0F] border border-[#1E2230] rounded-lg p-0.5 text-xs font-mono">
              <button
                onClick={() => {
                  setPortfolioMode('demo');
                  fetchTelemetry('demo');
                }}
                className={`px-3 py-1 rounded-md transition-colors ${
                  portfolioMode === 'demo'
                    ? 'bg-[#10B981] text-black font-bold'
                    : 'text-[#94A3B8] hover:text-white'
                }`}
              >
                Demo Portfolio
              </button>

              <button
                onClick={() => {
                  if (sessionStatus?.is_authenticated) {
                    setPortfolioMode('live');
                    fetchTelemetry('live');
                  } else {
                    handleLaunchKiteLogin();
                  }
                }}
                className={`px-3 py-1 rounded-md transition-colors ${
                  portfolioMode === 'live'
                    ? 'bg-[#06B6D4] text-black font-bold'
                    : 'text-[#94A3B8] hover:text-white'
                }`}
              >
                Live Kite Connect
              </button>
            </div>
          </div>

          {/* ACTION BUTTONS & THRESHOLD SELECTOR */}
          <div className="flex flex-wrap items-center gap-2.5 text-xs font-mono">
            <div className="flex items-center gap-1.5 bg-[#090A0F] border border-[#1E2230] px-2.5 py-1 rounded-lg">
              <span className="text-[#94A3B8]">Concentration Ceiling:</span>
              <select
                value={thresholdPct}
                onChange={(e) => setThresholdPct(Number(e.target.value))}
                className="bg-transparent text-[#10B981] font-bold focus:outline-none cursor-pointer"
              >
                <option value={15} className="bg-[#12141C] text-white">15% (Strict)</option>
                <option value={20} className="bg-[#12141C] text-white">20% (Institutional)</option>
                <option value={25} className="bg-[#12141C] text-white">25% (Permissive)</option>
              </select>
            </div>

            {sessionStatus?.is_authenticated ? (
              <div className="flex items-center gap-2">
                <button
                  onClick={() => fetchTelemetry('live')}
                  disabled={loading}
                  className="flex items-center gap-1.5 px-3 py-1.5 bg-[#06B6D4]/15 hover:bg-[#06B6D4]/25 text-[#06B6D4] border border-[#06B6D4]/40 rounded-lg transition-colors font-semibold"
                >
                  <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} />
                  <span>Sync Holdings</span>
                </button>

                <button
                  onClick={handleDisconnect}
                  title="Disconnect Kite Session"
                  className="flex items-center gap-1 px-2.5 py-1.5 bg-[#12141C] hover:bg-[#F43F5E]/20 text-[#94A3B8] hover:text-[#F43F5E] border border-[#1E2230] hover:border-[#F43F5E]/40 rounded-lg transition-colors"
                >
                  <LogOut className="h-3.5 w-3.5" />
                  <span>Logout</span>
                </button>
              </div>
            ) : (
              <div className="flex items-center gap-2">
                <button
                  onClick={handleLaunchKiteLogin}
                  disabled={loading}
                  className="flex items-center gap-2 px-4 py-1.5 bg-[#F97316] hover:bg-[#EA580C] text-white font-semibold rounded-lg shadow-lg shadow-[#F97316]/20 transition-all"
                >
                  <Key className="h-3.5 w-3.5" />
                  <span>Connect Zerodha Kite</span>
                </button>

                <button
                  onClick={() => setSettingsModalOpen(true)}
                  title="Configure Kite API Credentials"
                  className="p-1.5 bg-[#090A0F] border border-[#1E2230] hover:border-[#334155] rounded-lg text-[#94A3B8] hover:text-white transition-colors"
                >
                  <Settings className="h-4 w-4" />
                </button>
              </div>
            )}
          </div>
        </div>
      </section>

      {/* CREDENTIALS CONFIGURATION MODAL */}
      {settingsModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm p-4">
          <div className="bg-[#12141C] border border-[#1E2230] rounded-xl max-w-md w-full p-6 space-y-4 shadow-2xl">
            <div className="flex items-center justify-between border-b border-[#1E2230] pb-3">
              <div className="flex items-center gap-2">
                <Lock className="h-4 w-4 text-[#06B6D4]" />
                <h3 className="text-sm font-bold font-mono text-white uppercase tracking-wider">
                  Zerodha Kite Connect Setup
                </h3>
              </div>
              <button
                onClick={() => setSettingsModalOpen(false)}
                className="text-[#94A3B8] hover:text-white"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            <p className="text-xs text-[#94A3B8] leading-relaxed">
              Enter your Zerodha Kite Connect developer credentials to enable automated 1-click OAuth login.
              Ensure your App Redirect URI in the Kite Developer Console is set to{' '}
              <code className="text-[#06B6D4] bg-[#090A0F] px-1.5 py-0.5 rounded border border-[#1E2230]">
                http://localhost:3000/portfolio
              </code>.
            </p>

            <form onSubmit={handleSaveCredentialsAndLogin} className="space-y-3 text-xs font-mono">
              <div>
                <label className="text-[#E2E8F0] block mb-1 font-semibold">Kite API Key</label>
                <input
                  type="text"
                  placeholder="e.g. 7q8abcdef1234"
                  value={apiKeyInput}
                  onChange={(e) => setApiKeyInput(e.target.value)}
                  className="w-full bg-[#090A0F] border border-[#1E2230] rounded-lg px-3 py-2 text-white placeholder-[#475569] focus:outline-none focus:border-[#06B6D4]"
                  required
                />
              </div>

              <div>
                <label className="text-[#E2E8F0] block mb-1 font-semibold">Kite API Secret</label>
                <input
                  type="password"
                  placeholder="e.g. 9z8yxwvu7654321"
                  value={apiSecretInput}
                  onChange={(e) => setApiSecretInput(e.target.value)}
                  className="w-full bg-[#090A0F] border border-[#1E2230] rounded-lg px-3 py-2 text-white placeholder-[#475569] focus:outline-none focus:border-[#06B6D4]"
                  required
                />
              </div>

              <div className="pt-2 flex items-center justify-between gap-3">
                <button
                  type="button"
                  onClick={() => setSettingsModalOpen(false)}
                  className="px-3 py-2 rounded-lg bg-[#090A0F] border border-[#1E2230] text-[#94A3B8] hover:text-white"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={loading}
                  className="flex-1 px-4 py-2 bg-[#F97316] hover:bg-[#EA580C] text-white font-bold rounded-lg transition-colors flex items-center justify-center gap-1.5"
                >
                  <Key className="h-3.5 w-3.5" />
                  <span>Save & Authenticate</span>
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* SKELETON LOADING STATE HUD */}
      {loading ? (
        <section className="space-y-6">
          {/* SKELETON RADAR HUD */}
          <div className="bg-[#12141C] border border-[#1E2230] rounded-xl p-8 flex flex-col items-center justify-center text-center space-y-3">
            <div className="relative">
              <div className="h-12 w-12 rounded-full border-2 border-[#10B981]/30 border-t-[#10B981] animate-spin" />
              <PieChart className="h-5 w-5 text-[#10B981] absolute inset-0 m-auto" />
            </div>
            <div className="space-y-1">
              <p className="text-sm font-mono font-semibold text-white animate-pulse">
                {loadingStage}
              </p>
              <p className="text-xs font-mono text-[#64748B]">
                Optimizing sector weights, computing systematic betas & Herfindahl Index...
              </p>
            </div>
          </div>

          {/* SKELETON CARDS */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
            {[1, 2, 3, 4, 5].map((i) => (
              <div key={i} className="bg-[#12141C] border border-[#1E2230] rounded-xl p-4 space-y-3 animate-pulse">
                <div className="h-3 w-24 bg-[#1E2230] rounded" />
                <div className="h-7 w-32 bg-[#1E2230] rounded" />
                <div className="h-2 w-full bg-[#1E2230] rounded" />
              </div>
            ))}
          </div>

          {/* SKELETON TABLE */}
          <div className="bg-[#12141C] border border-[#1E2230] rounded-xl p-5 space-y-3 animate-pulse">
            <div className="h-4 w-48 bg-[#1E2230] rounded" />
            <div className="h-4 w-full bg-[#1E2230] rounded" />
            <div className="space-y-2 pt-2">
              {[1, 2, 3, 4].map((j) => (
                <div key={j} className="h-10 w-full bg-[#090A0F] rounded" />
              ))}
            </div>
          </div>
        </section>
      ) : telemetry ? (
        <>
          {/* TOP SUMMARY TELEMETRY CARDS */}
          <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
            {/* CARD 1: TOTAL VALUE */}
            <div className="bg-[#12141C] border border-[#1E2230] rounded-xl p-4 space-y-1 shadow-sm">
              <span className="text-[11px] font-mono text-[#94A3B8] uppercase tracking-wider block">
                Total Portfolio Value
              </span>
              <div className="text-xl font-bold font-mono text-white">
                ₹{telemetry.total_value.toLocaleString('en-IN', { minimumFractionDigits: 2 })}
              </div>
              <div className="flex items-center gap-1.5 text-xs font-mono">
                <span className="text-[#64748B]">Invested: ₹{telemetry.total_invested.toLocaleString('en-IN', { maximumFractionDigits: 0 })}</span>
                <span
                  className={`font-semibold ml-auto flex items-center ${
                    telemetry.total_pnl >= 0 ? 'text-[#10B981]' : 'text-[#F43F5E]'
                  }`}
                >
                  {telemetry.total_pnl >= 0 ? '+' : ''}
                  {telemetry.total_pnl_pct.toFixed(2)}%
                </span>
              </div>
            </div>

            {/* CARD 2: RISK POSTURE */}
            <div className="bg-[#12141C] border border-[#1E2230] rounded-xl p-4 space-y-1 shadow-sm">
              <span className="text-[11px] font-mono text-[#94A3B8] uppercase tracking-wider block">
                Active Risk Posture
              </span>
              <div className="flex items-center gap-2">
                <span
                  className={`h-2.5 w-2.5 rounded-full ${
                    telemetry.concentration_posture === 'HIGH_RISK_CONCENTRATED'
                      ? 'bg-[#F43F5E] animate-pulse'
                      : telemetry.concentration_posture === 'MODERATE'
                      ? 'bg-[#F59E0B]'
                      : 'bg-[#10B981]'
                  }`}
                />
                <span
                  className={`text-sm font-bold font-mono ${
                    telemetry.concentration_posture === 'HIGH_RISK_CONCENTRATED'
                      ? 'text-[#F43F5E]'
                      : telemetry.concentration_posture === 'MODERATE'
                      ? 'text-[#F59E0B]'
                      : 'text-[#10B981]'
                  }`}
                >
                  {telemetry.concentration_posture.replace(/_/g, ' ')}
                </span>
              </div>
              <p className="text-[11px] font-mono text-[#64748B]">
                HHI Concentration Score: {telemetry.hhi_concentration_index.toFixed(0)}
              </p>
            </div>

            {/* CARD 3: OVERWEIGHT BREACHES */}
            <div className="bg-[#12141C] border border-[#1E2230] rounded-xl p-4 space-y-1 shadow-sm">
              <span className="text-[11px] font-mono text-[#94A3B8] uppercase tracking-wider block">
                Overweight Breaches
              </span>
              <div className="flex items-center gap-2">
                <span className="text-xl font-bold font-mono text-white">
                  {telemetry.overweight_sectors_count}
                </span>
                <span className="text-xs text-[#94A3B8]">
                  {telemetry.overweight_sectors_count === 1 ? 'Sector' : 'Sectors'} &gt; {telemetry.threshold_pct}%
                </span>
              </div>
              <p className="text-[11px] font-mono text-[#F59E0B]">
                {telemetry.overweight_sectors_count > 0
                  ? 'Rebalancing recommended'
                  : 'Conforms to guidelines'}
              </p>
            </div>

            {/* CARD 4: PORTFOLIO BETA */}
            <div className="bg-[#12141C] border border-[#1E2230] rounded-xl p-4 space-y-1 shadow-sm">
              <span className="text-[11px] font-mono text-[#94A3B8] uppercase tracking-wider block">
                Weighted Portfolio Beta
              </span>
              <div className="flex items-center gap-2">
                <span className="text-xl font-bold font-mono text-[#06B6D4]">
                  {telemetry.portfolio_weighted_beta.toFixed(2)}
                </span>
                <span className="text-xs font-mono text-[#94A3B8]">
                  {telemetry.portfolio_weighted_beta > 1.15
                    ? 'High Beta'
                    : telemetry.portfolio_weighted_beta < 0.85
                    ? 'Defensive'
                    : 'Balanced'}
                </span>
              </div>
              <p className="text-[11px] font-mono text-[#64748B]">
                Across {telemetry.holdings_count} equity assets
              </p>
            </div>

            {/* CARD 5: LEAST-RISK PROJECTED BETA */}
            <div className="bg-[#12141C] border border-[#1E2230] rounded-xl p-4 space-y-1 shadow-sm">
              <span className="text-[11px] font-mono text-[#94A3B8] uppercase tracking-wider block">
                Projected Least-Risk Beta
              </span>
              <div className="flex items-center gap-2">
                <span className="text-xl font-bold font-mono text-[#10B981]">
                  {telemetry.projected_rebalanced_beta.toFixed(2)}
                </span>
                {telemetry.beta_reduction_pct > 0 && (
                  <span className="px-1.5 py-0.5 rounded text-[10px] font-bold font-mono bg-[#10B981]/15 text-[#10B981] border border-[#10B981]/30">
                    -{telemetry.beta_reduction_pct.toFixed(1)}%
                  </span>
                )}
              </div>
              <p className="text-[11px] font-mono text-[#10B981]">
                Tail risk compression target
              </p>
            </div>
          </section>

          {/* SECTOR ALLOCATION BREAKDOWN (VISUAL BAR & CARDS) */}
          <section className="bg-[#12141C] border border-[#1E2230] rounded-xl p-5 space-y-5">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Layers className="h-4 w-4 text-[#10B981]" />
                <h2 className="text-sm font-bold font-mono uppercase tracking-wider text-white">
                  Sector Allocation Breakdown vs {telemetry.threshold_pct}% Risk Ceiling
                </h2>
              </div>
              <span className="text-xs font-mono text-[#94A3B8]">
                {telemetry.sectors.length} Sectors Represented
              </span>
            </div>

            {/* SEGMENTED PROGRESS BAR */}
            <div className="space-y-1.5">
              <div className="h-4 w-full rounded-full bg-[#090A0F] border border-[#1E2230] overflow-hidden flex">
                {telemetry.sectors.map((sec) => (
                  <div
                    key={sec.sector}
                    style={{
                      width: `${sec.weight_pct}%`,
                      backgroundColor: SECTOR_COLORS[sec.sector] || '#94A3B8',
                    }}
                    title={`${sec.sector}: ${sec.weight_pct}% (₹${sec.value.toLocaleString('en-IN')})`}
                    className="h-full transition-all duration-300 relative group cursor-pointer hover:opacity-85"
                  />
                ))}
              </div>

              {/* BAR LEGEND */}
              <div className="flex flex-wrap items-center gap-3 pt-1 text-xs font-mono">
                {telemetry.sectors.map((sec) => (
                  <div key={sec.sector} className="flex items-center gap-1.5">
                    <span
                      className="h-2.5 w-2.5 rounded-sm shrink-0"
                      style={{ backgroundColor: SECTOR_COLORS[sec.sector] || '#94A3B8' }}
                    />
                    <span className="text-[#E2E8F0]">{sec.sector}</span>
                    <span className="text-[#94A3B8] font-semibold">{sec.weight_pct}%</span>
                  </div>
                ))}
              </div>
            </div>

            {/* SECTOR CARDS GRID */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 pt-2">
              {telemetry.sectors.map((sec) => (
                <div
                  key={sec.sector}
                  className={`p-3.5 rounded-lg border transition-colors ${
                    sec.is_overweight
                      ? 'bg-[#F59E0B]/5 border-[#F59E0B]/40 hover:border-[#F59E0B]'
                      : sec.is_defensive
                      ? 'bg-[#10B981]/5 border-[#10B981]/30 hover:border-[#10B981]/60'
                      : 'bg-[#090A0F] border-[#1E2230] hover:border-[#334155]'
                  }`}
                >
                  <div className="flex items-start justify-between gap-2 mb-2">
                    <div className="flex items-center gap-2">
                      <span
                        className="h-3 w-3 rounded-sm shrink-0"
                        style={{ backgroundColor: SECTOR_COLORS[sec.sector] || '#94A3B8' }}
                      />
                      <span className="text-xs font-bold text-white font-mono">{sec.sector}</span>
                    </div>
                    {sec.is_overweight ? (
                      <span className="px-2 py-0.5 rounded text-[10px] font-bold font-mono bg-[#F59E0B]/15 text-[#F59E0B] border border-[#F59E0B]/40">
                        +{sec.excess_pct}% OVERWEIGHT
                      </span>
                    ) : sec.is_defensive ? (
                      <span className="px-2 py-0.5 rounded text-[10px] font-bold font-mono bg-[#10B981]/15 text-[#10B981] border border-[#10B981]/40">
                        DEFENSIVE
                      </span>
                    ) : (
                      <span className="px-2 py-0.5 rounded text-[10px] font-mono text-[#94A3B8] bg-[#12141C] border border-[#1E2230]">
                        COMPLIANT
                      </span>
                    )}
                  </div>

                  <div className="space-y-1 text-xs font-mono">
                    <div className="flex items-center justify-between text-[#94A3B8]">
                      <span>Allocation Weight:</span>
                      <span className="text-white font-semibold">{sec.weight_pct}%</span>
                    </div>
                    <div className="flex items-center justify-between text-[#94A3B8]">
                      <span>Capital Allocated:</span>
                      <span className="text-white">
                        ₹{sec.value.toLocaleString('en-IN', { maximumFractionDigits: 0 })}
                      </span>
                    </div>
                    <div className="flex items-center justify-between text-[#94A3B8]">
                      <span>Sector Beta:</span>
                      <span className="text-[#06B6D4]">{sec.baseline_beta.toFixed(2)}</span>
                    </div>

                    {sec.is_overweight && (
                      <div className="flex items-center justify-between text-[#F59E0B] pt-1 border-t border-[#1E2230]">
                        <span>Recommended Trim:</span>
                        <span className="font-bold">
                          -₹{sec.excess_value.toLocaleString('en-IN', { maximumFractionDigits: 0 })}
                        </span>
                      </div>
                    )}

                    {/* ASSET PILLS */}
                    <div className="pt-2 flex flex-wrap gap-1">
                      {sec.assets.map((asset) => (
                        <span
                          key={asset}
                          className="px-1.5 py-0.5 rounded text-[10px] bg-[#12141C] text-[#CBD5E1] border border-[#1E2230]"
                        >
                          {asset.replace('.NS', '')}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* ACTIONABLE LEAST-RISK REBALANCING DIRECTIVES */}
          {telemetry.recommendations.length > 0 && (
            <section className="bg-[#12141C] border border-[#1E2230] rounded-xl p-5 space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <ShieldCheck className="h-4 w-4 text-[#10B981]" />
                  <h2 className="text-sm font-bold font-mono uppercase tracking-wider text-white">
                    Actionable Least-Risk Optimization Directives
                  </h2>
                </div>
                <span className="text-xs font-mono text-[#10B981] font-semibold">
                  Projected Beta: {telemetry.projected_rebalanced_beta.toFixed(2)} (-{telemetry.beta_reduction_pct.toFixed(1)}%)
                </span>
              </div>

              <div className="space-y-3">
                {telemetry.recommendations.map((rec, idx) => (
                  <div
                    key={idx}
                    className={`p-4 rounded-lg border text-xs font-mono space-y-2 ${
                      rec.action === 'TRIM'
                        ? 'bg-[#F43F5E]/5 border-[#F43F5E]/40'
                        : rec.action === 'ALLOCATE_DEFENSIVE'
                        ? 'bg-[#10B981]/5 border-[#10B981]/40'
                        : 'bg-[#090A0F] border-[#1E2230]'
                    }`}
                  >
                    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <span
                          className={`px-2 py-0.5 rounded text-[11px] font-bold ${
                            rec.action === 'TRIM'
                              ? 'bg-[#F43F5E]/20 text-[#F43F5E] border border-[#F43F5E]/40'
                              : rec.action === 'ALLOCATE_DEFENSIVE'
                              ? 'bg-[#10B981]/20 text-[#10B981] border border-[#10B981]/40'
                              : 'bg-[#12141C] text-[#94A3B8]'
                          }`}
                        >
                          {rec.action === 'TRIM'
                            ? '1. TRIM OVERWEIGHT RISK'
                            : rec.action === 'ALLOCATE_DEFENSIVE'
                            ? '2. REALLOCATE TO DEFENSIVE'
                            : 'RISK BALANCED'}
                        </span>
                        <span className="text-white font-bold text-sm">{rec.sector}</span>
                      </div>

                      <div className="flex items-center gap-2 text-xs">
                        <span className="text-[#94A3B8]">Allocation Target:</span>
                        <span className="text-white font-semibold">
                          {rec.current_weight_pct.toFixed(1)}%
                        </span>
                        <ArrowRight className="h-3 w-3 text-[#94A3B8]" />
                        <span
                          className={`font-bold ${
                            rec.action === 'TRIM' ? 'text-[#F43F5E]' : 'text-[#10B981]'
                          }`}
                        >
                          {rec.recommended_weight_pct.toFixed(1)}%
                        </span>
                        {rec.amount_currency > 0 && (
                          <span className="text-white font-bold ml-2">
                            (₹{rec.amount_currency.toLocaleString('en-IN', { maximumFractionDigits: 0 })})
                          </span>
                        )}
                      </div>
                    </div>

                    <p className="text-[#CBD5E1] text-[12px] leading-relaxed">
                      {rec.rationale}
                    </p>

                    {rec.proxy_tickers.length > 0 && (
                      <div className="flex items-center gap-2 pt-1 text-[11px]">
                        <span className="text-[#94A3B8]">Target Instruments:</span>
                        <div className="flex flex-wrap gap-1">
                          {rec.proxy_tickers.map((t) => (
                            <span
                              key={t}
                              className="px-2 py-0.5 rounded bg-[#12141C] text-[#06B6D4] border border-[#1E2230] font-semibold"
                            >
                              {t.replace('.NS', '')}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* HOLDINGS TELEMETRY TABLE */}
          <section className="bg-[#12141C] border border-[#1E2230] rounded-xl p-5 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                <Briefcase className="h-4 w-4 text-[#06B6D4]" />
                <h2 className="text-sm font-bold font-mono uppercase tracking-wider text-white">
                  Underlying Equity Holdings & Systematic Betas
                </h2>
              </div>

              {/* FILTER DROPDOWN */}
              <div className="flex items-center gap-2 text-xs font-mono">
                <span className="text-[#94A3B8]">Filter Sector:</span>
                <select
                  value={sectorFilter}
                  onChange={(e) => setSectorFilter(e.target.value)}
                  className="bg-[#090A0F] border border-[#1E2230] text-white rounded px-2.5 py-1 text-xs focus:outline-none focus:border-[#06B6D4]"
                >
                  <option value="ALL">All Sectors ({telemetry.holdings.length})</option>
                  {telemetry.sectors.map((s) => (
                    <option key={s.sector} value={s.sector}>
                      {s.sector} ({s.holdings_count})
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs font-mono">
                <thead className="bg-[#090A0F] text-[#94A3B8] border-b border-[#1E2230]">
                  <tr>
                    <th className="py-2.5 px-3 font-semibold">TICKER / SYMBOL</th>
                    <th className="py-2.5 px-3 font-semibold">SECTOR</th>
                    <th className="py-2.5 px-3 font-semibold text-right">QUANTITY</th>
                    <th className="py-2.5 px-3 font-semibold text-right">AVG PRICE</th>
                    <th className="py-2.5 px-3 font-semibold text-right">LAST PRICE</th>
                    <th className="py-2.5 px-3 font-semibold text-right">CURRENT VALUE</th>
                    <th className="py-2.5 px-3 font-semibold text-right">WEIGHT %</th>
                    <th className="py-2.5 px-3 font-semibold text-right">TOTAL P&L</th>
                    <th className="py-2.5 px-3 font-semibold text-right">BETA</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#1E2230]">
                  {filteredHoldings.map((h) => (
                    <tr key={h.ticker} className="hover:bg-[#1A1E2C] transition-colors">
                      <td className="py-3 px-3">
                        <div className="flex items-center gap-2">
                          <span className="font-bold text-white">{h.tradingsymbol}</span>
                          {h.is_overconcentrated_asset && (
                            <span
                              title="Single asset exceeds 15% concentration threshold"
                              className="px-1.5 py-0.2 rounded text-[9px] font-bold bg-[#F43F5E]/20 text-[#F43F5E] border border-[#F43F5E]/30"
                            >
                              &gt;15% ASSET
                            </span>
                          )}
                        </div>
                        <span className="text-[10px] text-[#64748B] block">{h.ticker}</span>
                      </td>
                      <td className="py-3 px-3">
                        <span
                          className="px-2 py-0.5 rounded text-[10px] font-semibold border"
                          style={{
                            backgroundColor: `${SECTOR_COLORS[h.sector] || '#94A3B8'}15`,
                            borderColor: `${SECTOR_COLORS[h.sector] || '#94A3B8'}40`,
                            color: SECTOR_COLORS[h.sector] || '#94A3B8',
                          }}
                        >
                          {h.sector}
                        </span>
                      </td>
                      <td className="py-3 px-3 text-right text-[#E2E8F0]">{h.quantity}</td>
                      <td className="py-3 px-3 text-right text-[#94A3B8]">
                        ₹{h.average_price.toFixed(2)}
                      </td>
                      <td className="py-3 px-3 text-right font-medium text-white">
                        ₹{h.last_price.toFixed(2)}
                      </td>
                      <td className="py-3 px-3 text-right font-bold text-white">
                        ₹{h.value.toLocaleString('en-IN', { minimumFractionDigits: 2 })}
                      </td>
                      <td className="py-3 px-3 text-right">
                        <span
                          className={`font-semibold ${
                            h.weight_pct > 15 ? 'text-[#F59E0B]' : 'text-white'
                          }`}
                        >
                          {h.weight_pct.toFixed(2)}%
                        </span>
                      </td>
                      <td className="py-3 px-3 text-right">
                        <div
                          className={`font-semibold ${
                            h.pnl >= 0 ? 'text-[#10B981]' : 'text-[#F43F5E]'
                          }`}
                        >
                          {h.pnl >= 0 ? '+' : ''}₹{h.pnl.toLocaleString('en-IN', { maximumFractionDigits: 0 })}
                        </div>
                        <span
                          className={`text-[10px] ${
                            h.pnl_percentage >= 0 ? 'text-[#10B981]' : 'text-[#F43F5E]'
                          }`}
                        >
                          ({h.pnl_percentage >= 0 ? '+' : ''}
                          {h.pnl_percentage.toFixed(2)}%)
                        </span>
                      </td>
                      <td className="py-3 px-3 text-right">
                        <span
                          className={`font-bold ${
                            h.beta > 1.25
                              ? 'text-[#F43F5E]'
                              : h.beta < 0.8
                              ? 'text-[#10B981]'
                              : 'text-[#06B6D4]'
                          }`}
                        >
                          {h.beta.toFixed(2)}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </>
      ) : null}
    </div>
  );
}
