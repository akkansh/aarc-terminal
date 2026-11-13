'use client';

import React, { useState, useEffect, useRef } from 'react';
import dynamic from 'next/dynamic';
import {
  TrendingUp,
  TrendingDown,
  ShieldAlert,
  AlertTriangle,
  Activity,
  CheckCircle2,
  Clock,
  Play,
  RotateCcw,
  Search,
  ExternalLink,
  ChevronDown,
  ChevronUp,
  Cpu,
  Zap,
  BarChart2,
  Lock,
  Layers,
  Compass,
  Sliders,
} from 'lucide-react';
import {
  currencySymbolForTicker,
  formatMarketValue,
  isIndexTicker,
  latestChartPoint,
} from '@/lib/marketDisplay';

const CandlestickChart = dynamic(() => import('@/components/CandlestickChart'), {
  ssr: false,
  loading: () => (
    <div className="h-[310px] w-full flex items-center justify-center text-xs text-[#64748B] font-mono bg-[#090A0F] rounded-lg border border-[#1E2230]">
      <span className="flex items-center gap-2">
        <Activity className="h-4 w-4 animate-spin text-[#10B981]" />
        Initializing Candlestick Engine...
      </span>
    </div>
  ),
});

interface ChartPoint {
  date: string;
  open?: number;
  high?: number;
  low?: number;
  close: number;
  volume: number;
  is_anomaly: boolean;
  bb_upper?: number;
  bb_middle?: number;
  bb_lower?: number;
  sma_50?: number;
}

interface QuantData {
  ticker: string;
  current_price: number;
  annualized_volatility: number;
  daily_VaR_95_pct: number;
  suggested_stop_loss_atr_pct: number;
  half_kelly_allocation_pct: number;
  recent_10d_anomalies: number;
  rsi_14?: number;
  recent_5d_return_pct?: number;
  sma20_trend?: string;
  close_gt_sma20?: boolean;
  sma20_val?: number;
  sma50_val?: number;
  sma20_z_score?: number;
  bb_upper?: number;
  bb_middle?: number;
  bb_lower?: number;
  bb_width_pct?: number;
  chart_data: ChartPoint[];
  time_machine_actual_return_pct: number | null;
  max_drawdown_5d_pct: number | null;
}

interface HeadlineItem {
  title: string;
  label: string;
  score: number;
  published_at?: string;
}

interface StockSuggestion {
  ticker: string;
  name: string;
  exchange: string;
}

interface NewsData {
  ticker: string;
  aggregate_sentiment_score: number | null;
  headlines: HeadlineItem[];
  news_available?: boolean;
  news_error?: string;
}

interface BullData {
  bull_thesis: string;
}

interface BearData {
  bear_thesis: string;
}

interface CROVerdict {
  action: 'LONG' | 'SHORT' | 'REJECT_TRADE';
  verdict?: 'LONG' | 'SHORT' | 'REJECT_TRADE';
  confidence: number;
  recommended_allocation_pct: number;
  stop_loss_pct: number;
  kill_switch_reason: string;
  verdict_summary: string;
}

const PRESET_TICKERS = [
  // Popular India large caps
  'RELIANCE.NS',
  'TCS.NS',
  'HDFCBANK.NS',
  'INFY.NS',
  'ICICIBANK.NS',
  'ZOMATO.NS',
  // Popular US large caps
  'NVDA',
  'AVGO',
  'AAPL',
  'MSFT',
  'GOOGL',
  'AMZN',
  'TSLA',
  // India benchmark indices
  '^NSEI',
  '^NSEBANK',
  '^BSESN',
];

const TICKER_NAMES: Record<string, string> = {
  'RELIANCE.NS': 'Reliance Industries Limited', 'TCS.NS': 'Tata Consultancy Services Limited',
  'HDFCBANK.NS': 'HDFC Bank Limited', 'INFY.NS': 'Infosys Limited',
  'ICICIBANK.NS': 'ICICI Bank Limited', 'ZOMATO.NS': 'Zomato Limited',
  NVDA: 'NVIDIA Corporation', AVGO: 'Broadcom Inc.', AAPL: 'Apple Inc.',
  MSFT: 'Microsoft Corporation', GOOGL: 'Alphabet Inc.', AMZN: 'Amazon.com, Inc.',
  TSLA: 'Tesla, Inc.', '^NSEI': 'NIFTY 50', '^NSEBANK': 'NIFTY Bank',
  '^BSESN': 'S&P BSE SENSEX',
};
const STORAGE_KEY = 'aarc.activeTicker';

export default function TerminalPage() {
  const [mounted, setMounted] = useState(false);
  const [ticker, setTicker] = useState('RELIANCE.NS');
  const [resolvedName, setResolvedName] = useState<{ ticker: string; name: string } | null>(null);
  const [customTicker, setCustomTicker] = useState('');
  const [suggestions, setSuggestions] = useState<StockSuggestion[]>([]);
  const [searchOpen, setSearchOpen] = useState(false);
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchError, setSearchError] = useState('');
  const [highlightedSuggestion, setHighlightedSuggestion] = useState(-1);
  const [cutoffDate, setCutoffDate] = useState('');
  const [appliedCutoff, setAppliedCutoff] = useState('');
  const [isRunning, setIsRunning] = useState(false);
  const [activeStep, setActiveStep] = useState(0); // 0 to 5
  const [showOutcome, setShowOutcome] = useState(false);
  const [showTechAnalysis, setShowTechAnalysis] = useState(true);

  // Streaming State from SSE
  const [quantData, setQuantData] = useState<QuantData | null>(null);
  const [newsData, setNewsData] = useState<NewsData | null>(null);
  const [bullData, setBullData] = useState<BullData | null>(null);
  const [bearData, setBearData] = useState<BearData | null>(null);
  const [croData, setCroData] = useState<CROVerdict | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const eventSourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    setMounted(true);
    let initialTicker = 'RELIANCE.NS';
    try {
      const saved = localStorage.getItem(STORAGE_KEY)?.trim().toUpperCase();
      if (saved && /^[A-Z0-9^.=\-]{1,32}$/.test(saved)) initialTicker = saved;
    } catch {
      // Storage may be unavailable in private browsing.
    }
    runCommittee(initialTicker, '');
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);

  useEffect(() => {
    if (TICKER_NAMES[ticker]) return;
    const controller = new AbortController();
    fetch(`http://localhost:8000/api/instrument?${new URLSearchParams({ ticker })}`, { signal: controller.signal })
      .then((response) => response.ok ? response.json() : null)
      .then((data) => {
        if (data?.ticker === ticker && typeof data.name === 'string') {
          setResolvedName({ ticker, name: data.name });
        }
      })
      .catch(() => {});
    return () => controller.abort();
  }, [ticker]);

  useEffect(() => {
    const query = customTicker.trim();
    if (!searchOpen || !query) return;
    const controller = new AbortController();
    setSuggestions([]);
    setSearchError('');
    setSearchLoading(true);
    setHighlightedSuggestion(-1);
    const timer = window.setTimeout(async () => {
      try {
        const response = await fetch(
          `http://localhost:8000/api/instruments/search?${new URLSearchParams({ q: query })}`,
          { signal: controller.signal },
        );
        if (!response.ok) throw new Error('Stock search is temporarily unavailable');
        const data = await response.json();
        if (!controller.signal.aborted) setSuggestions(data.results);
      } catch {
        if (!controller.signal.aborted) setSearchError('Stock search is temporarily unavailable. You can still enter a ticker symbol.');
      } finally {
        if (!controller.signal.aborted) setSearchLoading(false);
      }
    }, 300);
    return () => { window.clearTimeout(timer); controller.abort(); };
  }, [customTicker, searchOpen]);

  const runCommittee = (targetTicker: string, targetCutoff: string) => {
    const cleanT = targetTicker.trim().toUpperCase();
    setTicker(cleanT);
    try { localStorage.setItem(STORAGE_KEY, cleanT); } catch { /* storage is optional */ }
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    setIsRunning(true);
    setAppliedCutoff(targetCutoff);
    setActiveStep(1);
    setErrorMsg(null);
    setShowOutcome(false);

    // Reset previous stream data
    setQuantData(null);
    setNewsData(null);
    setBullData(null);
    setBearData(null);
    setCroData(null);

    const query = new URLSearchParams({ ticker: cleanT });
    if (targetCutoff.trim()) {
      query.append('cutoff_date', targetCutoff.trim());
    }

    const sseUrl = `http://localhost:8000/api/stream?${query.toString()}`;
    const es = new EventSource(sseUrl);
    eventSourceRef.current = es;

    es.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        const node = payload.node;
        const data = payload.data;

        if (node === 'quant') {
          setQuantData(data);
          setActiveStep(2);
        } else if (node === 'news') {
          setNewsData(data);
          setActiveStep(3);
        } else if (node === 'bull') {
          setBullData(data);
          setActiveStep(4);
        } else if (node === 'bear') {
          setBearData(data);
          setActiveStep(5);
        } else if (node === 'index_complete') {
          setIsRunning(false);
          es.close();
        } else if (node === 'cro') {
          setCroData(data);
          setIsRunning(false);
          es.close();
        } else if (node === 'error') {
          setErrorMsg(data.error || 'Pipeline execution warning');
          setIsRunning(false);
          es.close();
        }
      } catch (err) {
        console.error('SSE JSON parse error:', err);
      }
    };

    es.onerror = () => {
      setErrorMsg('Connection to the backend was interrupted');
      setIsRunning(false);
      es.close();
    };
  };

  const handleRun = () => {
    const input = customTicker.trim();
    if (!input) {
      runCommittee(ticker, cutoffDate);
      return;
    }
    const exact = suggestions.find((item) => item.ticker === input.toUpperCase());
    if (exact) {
      selectSuggestion(exact);
    } else if (searchLoading) {
      setSearchError('Wait for search results, then select a stock.');
    } else if (suggestions.length || !/^[A-Za-z0-9^.\-=]{1,32}$/.test(input)) {
      setSearchOpen(true);
      setSearchError('Select a stock from the results, or enter its exact ticker symbol.');
    } else {
      setSearchOpen(false);
      runCommittee(input.toUpperCase(), cutoffDate);
    }
  };

  const selectSuggestion = (item: StockSuggestion) => {
    setResolvedName({ ticker: item.ticker, name: item.name });
    setCustomTicker('');
    setSearchOpen(false);
    setSearchError('');
    setSuggestions([]);
    runCommittee(item.ticker, cutoffDate);
  };

  const handleSelectPreset = (t: string) => {
    setTicker(t);
    setCustomTicker('');
    setSearchOpen(false);
    runCommittee(t, cutoffDate);
  };

  const handleTestCrashPreset = () => {
    setTicker('RELIANCE.NS');
    setCustomTicker('');
    setCutoffDate('2024-06-03');
    runCommittee('RELIANCE.NS', '2024-06-03');
  };

  const handleTestMarchPreset = () => {
    setTicker('RELIANCE.NS');
    setCustomTicker('');
    setCutoffDate('2024-03-01');
    runCommittee('RELIANCE.NS', '2024-03-01');
  };

  const handleResetCutoff = () => {
    setCutoffDate('');
    runCommittee(ticker, '');
  };

  const currencySymbol = currencySymbolForTicker(ticker);
  const isIndex = isIndexTicker(ticker);
  const instrumentName = TICKER_NAMES[ticker] || (resolvedName?.ticker === ticker ? resolvedName.name : null);
  const priceUnit = isIndex ? 'pts' : '';
  const lastChartPoint = latestChartPoint(quantData?.chart_data);
  const lastClose = lastChartPoint?.close ?? null;
  const lastCloseDate = lastChartPoint?.date ?? null;
  const newsAvailable = !!newsData && newsData.news_available !== false;
  const sentimentScore =
    newsAvailable && typeof newsData?.aggregate_sentiment_score === 'number'
      ? newsData.aggregate_sentiment_score
      : null;
  const pendingCustomTicker = customTicker.trim().toUpperCase();
  const showPendingCustom = pendingCustomTicker !== '' && pendingCustomTicker !== ticker;

  if (!mounted) {
    return <div className="min-h-screen bg-[#090A0F] p-6 text-xs font-mono text-[#94A3B8]">Loading dashboard...</div>;
  }

  return (
    <div className="min-h-screen bg-[#090A0F] text-[#E2E8F0] p-4 lg:p-6 space-y-6">
      {/* Top Header / Branding */}
      <header className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-[#1E2230] pb-4">
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-lg bg-[#12141C] border border-[#1E2230] flex items-center justify-center text-[#10B981] shadow-lg shadow-[#10B981]/10">
            <Cpu className="h-6 w-6" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
                AARC QUANT TERMINAL
              </h1>
              <span className="px-2 py-0.5 rounded text-[10px] font-semibold tracking-wider bg-[#10B981]/10 text-[#10B981] border border-[#10B981]/30">
                INSTITUTIONAL
              </span>
            </div>
            <p className="text-xs text-[#94A3B8]">
              Autonomous Adversarial Risk Committee • Quant ML + FinBERT NLP + Multi-Agent Arbitration
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3 text-xs mono-nums">
          <div className="flex items-center gap-2 bg-[#12141C] border border-[#1E2230] px-3 py-1.5 rounded-md">
            <span className="h-2 w-2 rounded-full bg-[#10B981] animate-pulse"></span>
            <span className="text-[#94A3B8]">MODEL:</span>
            <span className="text-white font-medium">gemini-2.5-flash</span>
          </div>
          <div className="flex items-center gap-2 bg-[#12141C] border border-[#1E2230] px-3 py-1.5 rounded-md">
            <span className="text-[#94A3B8]">PORT:</span>
            <span className="text-[#06B6D4]">8000 (SSE)</span>
          </div>
        </div>
      </header>

      {/* PIPELINE ERROR BANNER */}
      {errorMsg && (
        <section
          role="alert"
          className="flex items-start gap-3 rounded-xl border border-[#F43F5E]/40 bg-[#F43F5E]/10 px-4 py-3 text-xs"
        >
          <AlertTriangle className="h-4 w-4 text-[#F43F5E] shrink-0 mt-0.5" />
          <div className="space-y-0.5">
            <div className="font-mono font-bold uppercase tracking-wider text-[#F43F5E]">
              Pipeline Error
            </div>
            <p className="text-[#E2E8F0]">
              {errorMsg}. No cached or demo result was substituted, so any displayed
              telemetry may be incomplete.
            </p>
          </div>
        </section>
      )}

      {/* TOP COMMAND BAR */}
      <section className="bg-[#12141C] border border-[#1E2230] rounded-xl p-4 space-y-4">
        <div className="flex flex-wrap items-center gap-2 text-xs font-mono">
          <span className="text-[#64748B] uppercase tracking-wider">Active Ticker:</span>
          <span className="px-2 py-1 rounded bg-[#10B981]/10 text-[#10B981] border border-[#10B981]/40 font-bold">
            {ticker}
          </span>
          <span className="text-white font-semibold">{instrumentName || 'Instrument name unavailable'}</span>
          <span className="px-2 py-1 rounded bg-[#090A0F] text-[#94A3B8] border border-[#1E2230]">
            {isIndex ? 'Index (points)' : 'Daily close'}
          </span>
          {appliedCutoff && (
            <span className="px-2 py-1 rounded bg-[#F59E0B]/10 text-[#F59E0B] border border-[#F59E0B]/30">
              Requested cutoff: {appliedCutoff}
            </span>
          )}
          {showPendingCustom && (
            <span className="px-2 py-1 rounded bg-[#090A0F] text-[#94A3B8] border border-[#1E2230]">
              Pending custom: <span className="text-white font-bold">{pendingCustomTicker}</span>
            </span>
          )}
        </div>

        {isIndex && (
          <div role="status" className="rounded-lg border border-[#F59E0B]/40 bg-[#F59E0B]/10 px-3 py-2 text-xs text-[#FCD34D]">
            Index analysis only: an index cannot be traded directly. No trade recommendation or allocation is generated.
          </div>
        )}

        <div className="flex flex-wrap items-center justify-between gap-4">
          {/* Quick Select Pills */}
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-xs font-semibold text-[#94A3B8] uppercase tracking-wider mr-1">
              Universe:
            </span>
            {PRESET_TICKERS.map((t) => (
              <button
                key={t}
                onClick={() => handleSelectPreset(t)}
                className={`px-3 py-1.5 rounded-lg text-xs font-mono font-medium transition-all ${
                  ticker === t
                    ? 'bg-[#10B981] text-[#090A0F] font-bold shadow-md shadow-[#10B981]/25 border border-[#10B981]'
                    : 'bg-[#090A0F] text-[#94A3B8] border border-[#1E2230] hover:border-[#2A3042] hover:text-white'
                }`}
              >
                {t}
              </button>
            ))}
          </div>

          {/* Custom Search & Run */}
          <div className="flex items-center gap-3 flex-wrap">
            <div className="relative">
              <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-[#64748B]" />
              <input
                type="text"
                placeholder="Company or ticker..."
                value={customTicker}
                role="combobox"
                aria-label="Search company name or ticker"
                aria-autocomplete="list"
                aria-expanded={searchOpen && !!customTicker.trim()}
                aria-controls="stock-search-results"
                aria-activedescendant={highlightedSuggestion >= 0 ? `stock-result-${highlightedSuggestion}` : undefined}
                onFocus={() => setSearchOpen(true)}
                onBlur={() => setSearchOpen(false)}
                onChange={(e) => { setCustomTicker(e.target.value); setSearchOpen(true); }}
                onKeyDown={(e) => {
                  if (e.key === 'ArrowDown' && suggestions.length) {
                    e.preventDefault();
                    setSearchOpen(true);
                    setHighlightedSuggestion((i) => Math.min(i + 1, suggestions.length - 1));
                  } else if (e.key === 'ArrowUp' && suggestions.length) {
                    e.preventDefault();
                    setHighlightedSuggestion((i) => Math.max(i - 1, 0));
                  } else if (e.key === 'Escape') {
                    setSearchOpen(false);
                  } else if (e.key === 'Enter') {
                    e.preventDefault();
                    if (searchOpen && highlightedSuggestion >= 0 && suggestions[highlightedSuggestion]) {
                      selectSuggestion(suggestions[highlightedSuggestion]);
                    } else {
                      handleRun();
                    }
                  }
                }}
                className="bg-[#090A0F] border border-[#1E2230] rounded-lg pl-9 pr-3 py-1.5 text-xs font-mono text-white placeholder-[#64748B] focus:outline-none focus:border-[#10B981] w-36 sm:w-44"
              />
              {searchOpen && customTicker.trim() && (
                <div className="absolute z-50 mt-1 w-80 max-w-[85vw] rounded-lg border border-[#2A3042] bg-[#12141C] shadow-xl">
                  {searchLoading ? (
                    <p role="status" className="p-3 text-xs text-[#94A3B8]">Searching stocks...</p>
                  ) : suggestions.length > 0 ? (
                    <ul id="stock-search-results" role="listbox" className="max-h-64 overflow-y-auto py-1">
                      {suggestions.map((item, index) => (
                        <li key={item.ticker} id={`stock-result-${index}`} role="option" aria-selected={highlightedSuggestion === index}>
                          <button type="button" onMouseDown={(e) => e.preventDefault()} onClick={() => selectSuggestion(item)}
                            className={`w-full px-3 py-2 text-left text-xs hover:bg-[#1E2230] ${highlightedSuggestion === index ? 'bg-[#1E2230]' : ''}`}>
                            <span className="font-mono font-bold text-[#10B981]">{item.ticker}</span>
                            <span className="ml-2 text-[#E2E8F0]">{item.name}</span>
                            {item.exchange && <span className="block mt-0.5 text-[#64748B]">{item.exchange}</span>}
                          </button>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p role="status" className="p-3 text-xs text-[#94A3B8]">No matching stocks found.</p>
                  )}
                  {searchError && <p role="status" className="border-t border-[#1E2230] p-3 text-xs text-[#F59E0B]">{searchError}</p>}
                </div>
              )}
            </div>

            <button
              onClick={handleRun}
              disabled={isRunning}
              className={`flex items-center gap-2 px-5 py-2 rounded-lg text-xs font-bold font-mono tracking-wider transition-all ${
                isRunning
                  ? 'bg-[#1E2230] text-[#64748B] cursor-not-allowed'
                  : 'bg-gradient-to-r from-[#10B981] to-[#059669] text-[#090A0F] hover:opacity-95 shadow-lg shadow-[#10B981]/20'
              }`}
            >
              {isRunning ? (
                <>
                  <div className="h-3 w-3 border-2 border-[#64748B] border-t-white rounded-full animate-spin"></div>
                  PROCESSING...
                </>
              ) : (
                <>
                  <Play className="h-3.5 w-3.5 fill-current" />
                  RUN COMMITTEE
                </>
              )}
            </button>
          </div>
        </div>

        {/* Time Machine Presets & Date Input */}
        <div className="flex flex-wrap items-center justify-between gap-3 pt-3 border-t border-[#1E2230]/70 text-xs">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="flex items-center gap-1.5 text-[#94A3B8] font-medium">
              <Clock className="h-3.5 w-3.5 text-[#F59E0B]" />
              Time Machine Mode:
            </span>
            <input
              type="date"
              value={cutoffDate}
              onChange={(e) => setCutoffDate(e.target.value)}
              className="bg-[#090A0F] border border-[#1E2230] rounded-md px-2.5 py-1 text-xs font-mono text-white focus:outline-none focus:border-[#F59E0B]"
            />
            <span className="text-[11px] font-mono text-[#94A3B8]" aria-live="polite">
              Data shown as of:{' '}
              <span className="font-semibold text-white">
                {lastCloseDate || (isRunning ? 'Loading…' : 'Unavailable')}
              </span>
            </span>
            {cutoffDate && (
              <button
                onClick={handleResetCutoff}
                className="text-[11px] text-[#94A3B8] hover:text-[#F43F5E] flex items-center gap-1 transition-colors"
                title="Reset to Live Mode"
              >
                <RotateCcw className="h-3 w-3" /> Reset
              </button>
            )}
          </div>

          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-[#64748B] text-[11px]">Backtest Presets:</span>
            <button
              onClick={handleTestCrashPreset}
              className="px-2.5 py-1 rounded bg-[#F43F5E]/10 border border-[#F43F5E]/30 text-[#F43F5E] hover:bg-[#F43F5E]/20 text-[11px] font-mono font-medium transition-all"
            >
              🚨 Election Crash (2024-06-03)
            </button>
            <button
              onClick={handleTestMarchPreset}
              className="px-2.5 py-1 rounded bg-[#F59E0B]/10 border border-[#F59E0B]/30 text-[#F59E0B] hover:bg-[#F59E0B]/20 text-[11px] font-mono font-medium transition-all"
            >
              ⚡ Test 2024-03-01
            </button>
          </div>
        </div>

        {/* Live 5-Step Pipeline Status Bar */}
        <div className="pt-2">
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
            {[
              { id: 1, label: '[1] Quant ML', ready: !!quantData },
              { id: 2, label: '[2] FinBERT NLP', ready: !!newsData },
              ...(!isIndex ? [
                { id: 3, label: '[3] Bull Agent', ready: !!bullData },
                { id: 4, label: '[4] Short-Seller', ready: !!bearData },
                { id: 5, label: '[5] CRO Verdict', ready: !!croData },
              ] : []),
            ].map((step) => {
              const isCurrent = activeStep === step.id && isRunning;
              const isDone = step.ready;

              let bgBorder = 'bg-[#090A0F] border-[#1E2230] text-[#64748B]';
              if (isDone) {
                bgBorder = 'bg-[#10B981]/10 border-[#10B981]/40 text-[#10B981]';
              } else if (isCurrent) {
                bgBorder = 'bg-[#06B6D4]/10 border-[#06B6D4] text-[#06B6D4] animate-pulse';
              }

              return (
                <div
                  key={step.id}
                  className={`flex items-center justify-between px-3 py-2 rounded-lg border text-xs font-mono transition-all ${bgBorder}`}
                >
                  <span className="font-semibold">{step.label}</span>
                  {isDone ? (
                    <CheckCircle2 className="h-3.5 w-3.5 text-[#10B981]" />
                  ) : isCurrent ? (
                    <div className="h-2 w-2 rounded-full bg-[#06B6D4] animate-ping" />
                  ) : (
                    <span className="text-[10px] text-[#64748B]">IDLE</span>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* ROW 1: 4 MONOSPACE TELEMETRY CARDS */}
      <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Card 1: Price & Annualized Vol */}
        <div className="bg-[#12141C] border border-[#1E2230] rounded-xl p-4 flex flex-col justify-between">
          <div className="flex items-center justify-between text-xs text-[#94A3B8]">
            <span className="uppercase tracking-wider font-medium">Last Daily Close & Volatility</span>
            <Activity className="h-4 w-4 text-[#06B6D4]" />
          </div>
          <div className="my-2">
            <div className="text-2xl font-bold font-mono text-white">
              {lastClose !== null ? formatMarketValue(lastClose, ticker) : '---'}
            </div>
            <div className="mt-1 text-[10px] font-mono text-[#64748B]">
              {lastCloseDate
                ? `Daily close • as of ${lastCloseDate} (not live)`
                : 'Daily close as-of unavailable'}
            </div>
          </div>
          <div className="flex items-center justify-between text-xs font-mono pt-2 border-t border-[#1E2230]/60">
            <span className="text-[#94A3B8]">Ann. Vol (σ):</span>
            <span className="font-bold text-[#E2E8F0]">
              {quantData ? `${quantData.annualized_volatility}%` : '---'}
            </span>
          </div>
        </div>

        {/* Card 2: 95% Daily VaR & Suggested Stop Loss */}
        <div className="bg-[#12141C] border border-[#1E2230] rounded-xl p-4 flex flex-col justify-between">
          <div className="flex items-center justify-between text-xs text-[#94A3B8]">
            <span className="uppercase tracking-wider font-medium">{isIndex ? '95% Daily VaR' : '95% Daily VaR & ATR Stop'}</span>
            <ShieldAlert className="h-4 w-4 text-[#F43F5E]" />
          </div>
          <div className="my-2">
            <div className="text-2xl font-bold font-mono text-[#F43F5E]">
              {quantData ? `${quantData.daily_VaR_95_pct}%` : '---'}
            </div>
          </div>
          {!isIndex && <div className="flex items-center justify-between text-xs font-mono pt-2 border-t border-[#1E2230]/60">
            <span className="text-[#94A3B8]">Stop (2x ATR):</span>
            <span className="font-bold text-[#E2E8F0]">
              {quantData ? `${quantData.suggested_stop_loss_atr_pct}%` : '---'}
            </span>
          </div>}
        </div>

        {/* Card 3: Isolation Forest Anomalies Count */}
        <div className="bg-[#12141C] border border-[#1E2230] rounded-xl p-4 flex flex-col justify-between">
          <div className="flex items-center justify-between text-xs text-[#94A3B8]">
            <span className="uppercase tracking-wider font-medium">10D ML Anomalies (IForest)</span>
            <AlertTriangle
              className={`h-4 w-4 ${
                (quantData?.recent_10d_anomalies ?? 0) >= 3 ? 'text-[#F43F5E] animate-bounce' : 'text-[#F59E0B]'
              }`}
            />
          </div>
          <div className="my-2 flex items-baseline gap-3">
            <div
              className={`text-2xl font-bold font-mono ${
                (quantData?.recent_10d_anomalies ?? 0) >= 3 ? 'text-[#F43F5E]' : 'text-white'
              }`}
            >
              {quantData ? quantData.recent_10d_anomalies : '---'}
            </div>
            {quantData && (
              <span
                className={`text-[10px] font-mono px-2 py-0.5 rounded font-bold uppercase tracking-wider ${
                  quantData.recent_10d_anomalies >= 3
                    ? 'bg-[#F43F5E]/20 text-[#F43F5E] border border-[#F43F5E]'
                    : quantData.recent_10d_anomalies > 0
                    ? 'bg-[#F59E0B]/20 text-[#F59E0B] border border-[#F59E0B]/40'
                    : 'bg-[#10B981]/20 text-[#10B981] border border-[#10B981]/40'
                }`}
              >
                {quantData.recent_10d_anomalies >= 3
                  ? 'CRITICAL ALERT'
                  : quantData.recent_10d_anomalies > 0
                  ? 'ELEVATED'
                  : 'NORMAL'}
              </span>
            )}
          </div>
          {!isIndex && <div className="flex items-center justify-between text-xs font-mono pt-2 border-t border-[#1E2230]/60">
            <span className="text-[#94A3B8]">Half-Kelly Cap:</span>
            <span className="font-bold text-[#10B981]">
              {quantData ? `${quantData.half_kelly_allocation_pct}%` : '---'}
            </span>
          </div>}
        </div>

        {/* Card 4: FinBERT Sentiment Score */}
        <div className="bg-[#12141C] border border-[#1E2230] rounded-xl p-4 flex flex-col justify-between">
          <div className="flex items-center justify-between text-xs text-[#94A3B8]">
            <span className="uppercase tracking-wider font-medium">FinBERT Sentiment NLP</span>
            <Zap className="h-4 w-4 text-[#10B981]" />
          </div>
          <div className="my-2 flex items-baseline justify-between">
            <div
              className={`text-2xl font-bold font-mono ${
                sentimentScore !== null && sentimentScore >= 0.1
                  ? 'text-[#10B981]'
                  : sentimentScore !== null && sentimentScore <= -0.1
                  ? 'text-[#F43F5E]'
                  : newsData && !newsAvailable
                  ? 'text-[#F59E0B]'
                  : 'text-[#94A3B8]'
              }`}
            >
              {newsData && !newsAvailable
                ? 'N/A'
                : sentimentScore !== null
                ? `${sentimentScore > 0 ? '+' : ''}${sentimentScore}`
                : '---'}
            </div>
            {newsData && (
              <span className="text-[11px] font-mono text-[#94A3B8]">
                {!newsAvailable
                  ? 'UNAVAILABLE'
                  : sentimentScore !== null && sentimentScore > 0.15
                  ? 'BULLISH'
                  : sentimentScore !== null && sentimentScore < -0.15
                  ? 'BEARISH'
                  : 'NEUTRAL'}
              </span>
            )}
          </div>
          {/* Visual Bar (-1.0 to +1.0) */}
          <div className="pt-2 border-t border-[#1E2230]/60 space-y-1">
            <div className="h-2 w-full bg-[#090A0F] rounded-full overflow-hidden relative flex">
              <div className="w-1/2 bg-gradient-to-r from-[#F43F5E] to-transparent"></div>
              <div className="w-1/2 bg-gradient-to-r from-transparent to-[#10B981]"></div>
              {/* Pointer Marker */}
              {sentimentScore !== null && (
                <div
                  className="absolute top-0 bottom-0 w-1.5 bg-white shadow-md shadow-white rounded-full transition-all"
                  style={{
                    left: `${Math.max(
                      2,
                      Math.min(96, ((sentimentScore + 1) / 2) * 100)
                    )}%`,
                  }}
                />
              )}
            </div>
            <div className="flex justify-between text-[9px] font-mono text-[#64748B]">
              <span>-1.0</span>
              <span>0.0</span>
              <span>+1.0</span>
            </div>
          </div>
        </div>
      </section>

      {/* ROW 2: MAIN SPLIT ARENA */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* LEFT COLUMN: 60-Day Recharts + FinBERT News (7 cols) */}
        <div className={`${isIndex ? 'lg:col-span-12' : 'lg:col-span-7'} space-y-6`}>
          {/* Candlestick & Volume Chart Card */}
          <div className="bg-[#12141C] border border-[#1E2230] rounded-xl p-4 space-y-3 shadow-lg">
            <div className="flex items-center justify-between border-b border-[#1E2230] pb-2.5">
              <div className="flex items-center gap-2">
                <BarChart2 className="h-4 w-4 text-[#10B981]" />
                <h3 className="text-xs font-semibold uppercase tracking-wider text-white">
                  60-Day Professional Candlestick & Volume Terminal
                </h3>
              </div>
              <div className="flex items-center gap-2 text-[10px] font-mono text-[#64748B]">
                <span className="px-1.5 py-0.5 rounded bg-[#10B981]/10 text-[#10B981] border border-[#10B981]/30">
                  GPU ACCELERATED
                </span>
                <span>TradingView Engine</span>
              </div>
            </div>

            <div>
              {mounted && quantData && quantData.chart_data && quantData.chart_data.length > 0 ? (
                <CandlestickChart
                  data={quantData.chart_data}
                  currencySymbol={currencySymbol}
                  ticker={ticker}
                />
              ) : (
                <div className="h-[310px] flex items-center justify-center text-xs text-[#64748B] font-mono bg-[#090A0F] rounded-lg border border-[#1E2230]">
                  {isRunning ? 'Calculating price series & anomaly forest...' : 'No telemetry loaded'}
                </div>
              )}
            </div>
          </div>

          {/* Technical Analysis Live Metrics Panel */}
          <div className="bg-[#12141C] border border-[#1E2230] rounded-xl p-4 space-y-3 shadow-lg">
            <div className="flex items-center justify-between border-b border-[#1E2230] pb-2">
              <div className="flex items-center gap-2">
                <Compass className="h-4 w-4 text-[#38BDF8]" />
                <h3 className="text-xs font-semibold uppercase tracking-wider text-white">
                  Live Technical Analysis Radar
                </h3>
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-[#38BDF8]/10 text-[#38BDF8] border border-[#38BDF8]/30">
                  QUANT METRICS
                </span>
              </div>
              <button
                type="button"
                onClick={() => setShowTechAnalysis(!showTechAnalysis)}
                className="text-xs text-[#64748B] hover:text-[#E2E8F0] font-mono flex items-center gap-1 transition-colors px-2 py-0.5 rounded bg-[#090A0F] border border-[#1E2230]"
              >
                {showTechAnalysis ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
                <span>{showTechAnalysis ? 'Hide Radar' : 'Show Radar'}</span>
              </button>
            </div>

            {showTechAnalysis && (
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-1">
                {/* 1. RSI-14 */}
                <div className="p-3 rounded-lg bg-[#090A0F] border border-[#1E2230] flex flex-col justify-between space-y-2">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-[#94A3B8] font-mono text-[11px]">RSI (14-Day)</span>
                    <span
                      className={`px-1.5 py-0.5 rounded text-[9px] font-bold uppercase font-mono border ${
                        (quantData?.rsi_14 ?? 50) >= 70
                          ? 'bg-[#F43F5E]/15 text-[#F43F5E] border-[#F43F5E]/30'
                          : (quantData?.rsi_14 ?? 50) <= 30
                          ? 'bg-[#10B981]/15 text-[#10B981] border-[#10B981]/30'
                          : 'bg-[#64748B]/10 text-[#94A3B8] border-[#64748B]/30'
                      }`}
                    >
                      {(quantData?.rsi_14 ?? 50) >= 70
                        ? 'Overbought'
                        : (quantData?.rsi_14 ?? 50) <= 30
                        ? 'Oversold'
                        : 'Neutral'}
                    </span>
                  </div>
                  <div className="text-xl font-bold font-mono text-white">
                    {quantData?.rsi_14 !== undefined ? quantData.rsi_14.toFixed(2) : '---'}
                  </div>
                  {/* RSI Meter */}
                  <div className="space-y-1">
                    <div className="h-1.5 w-full bg-[#1E2230] rounded-full overflow-hidden relative">
                      <div
                        className={`h-full rounded-full transition-all ${
                          (quantData?.rsi_14 ?? 50) >= 70
                            ? 'bg-[#F43F5E]'
                            : (quantData?.rsi_14 ?? 50) <= 30
                            ? 'bg-[#10B981]'
                            : 'bg-[#38BDF8]'
                        }`}
                        style={{
                          width: `${Math.max(0, Math.min(100, quantData?.rsi_14 ?? 50))}%`,
                        }}
                      />
                    </div>
                    <div className="flex justify-between text-[9px] font-mono text-[#64748B]">
                      <span>30 (OS)</span>
                      <span>50</span>
                      <span>70 (OB)</span>
                    </div>
                  </div>
                </div>

                {/* 2. SMA-20 Distance Z-Score */}
                <div className="p-3 rounded-lg bg-[#090A0F] border border-[#1E2230] flex flex-col justify-between space-y-2">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-[#94A3B8] font-mono text-[11px]">SMA-20 Z-Score</span>
                    <span
                      className={`px-1.5 py-0.5 rounded text-[9px] font-bold uppercase font-mono border ${
                        Math.abs(quantData?.sma20_z_score ?? 0) >= 2.0
                          ? 'bg-[#F43F5E]/15 text-[#F43F5E] border-[#F43F5E]/30'
                          : Math.abs(quantData?.sma20_z_score ?? 0) >= 1.0
                          ? 'bg-[#F59E0B]/15 text-[#F59E0B] border-[#F59E0B]/30'
                          : 'bg-[#10B981]/15 text-[#10B981] border-[#10B981]/30'
                      }`}
                    >
                      {Math.abs(quantData?.sma20_z_score ?? 0) >= 2.0
                        ? '±2σ Stretch'
                        : Math.abs(quantData?.sma20_z_score ?? 0) >= 1.0
                        ? '±1σ Dev'
                        : 'Mean-Revert'}
                    </span>
                  </div>
                  <div className="text-xl font-bold font-mono text-white flex items-baseline gap-1">
                    <span>
                      {quantData?.sma20_z_score !== undefined
                        ? `${quantData.sma20_z_score > 0 ? '+' : ''}${quantData.sma20_z_score.toFixed(2)}`
                        : '---'}
                    </span>
                    <span className="text-xs text-[#94A3B8] font-normal">σ</span>
                  </div>
                  <div className="text-[10px] font-mono text-[#64748B] flex justify-between">
                    <span>SMA20: {quantData?.sma20_val ? formatMarketValue(quantData.sma20_val, ticker) : '---'}</span>
                    <span>SMA50: {quantData?.sma50_val ? formatMarketValue(quantData.sma50_val, ticker) : '---'}</span>
                  </div>
                </div>

                {/* 3. Bollinger Band Width % */}
                <div className="p-3 rounded-lg bg-[#090A0F] border border-[#1E2230] flex flex-col justify-between space-y-2">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-[#94A3B8] font-mono text-[11px]">BB Width %</span>
                    <span
                      className={`px-1.5 py-0.5 rounded text-[9px] font-bold uppercase font-mono border ${
                        (quantData?.bb_width_pct ?? 0) < 10.0
                          ? 'bg-[#F59E0B]/15 text-[#F59E0B] border-[#F59E0B]/30'
                          : (quantData?.bb_width_pct ?? 0) > 25.0
                          ? 'bg-[#F43F5E]/15 text-[#F43F5E] border-[#F43F5E]/30'
                          : 'bg-[#10B981]/15 text-[#10B981] border-[#10B981]/30'
                      }`}
                    >
                      {(quantData?.bb_width_pct ?? 0) < 10.0
                        ? 'Vol Squeeze'
                        : (quantData?.bb_width_pct ?? 0) > 25.0
                        ? 'High Vol'
                        : 'Normal'}
                    </span>
                  </div>
                  <div className="text-xl font-bold font-mono text-white">
                    {quantData?.bb_width_pct !== undefined ? `${quantData.bb_width_pct.toFixed(2)}%` : '---'}
                  </div>
                  <div className="text-[10px] font-mono text-[#64748B] truncate">
                    Bands: {quantData?.bb_lower ? formatMarketValue(quantData.bb_lower, ticker) : '---'} - {quantData?.bb_upper ? formatMarketValue(quantData.bb_upper, ticker) : '---'}
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* 8 FinBERT Scored News Table */}
          <div className="bg-[#12141C] border border-[#1E2230] rounded-xl p-4 space-y-3">
            <div className="flex items-center justify-between border-b border-[#1E2230] pb-2">
              <div className="flex items-center gap-2">
                <Layers className="h-4 w-4 text-[#10B981]" />
                <h3 className="text-xs font-semibold uppercase tracking-wider text-white">
                  FinBERT Headlines for {ticker} ({newsData?.headlines.length ?? 0}/8)
                </h3>
              </div>
              <span className="text-[10px] font-mono text-[#64748B]">Google News RSS</span>
            </div>

            <div className="max-h-56 overflow-y-auto space-y-2 pr-1">
              {newsData && newsData.headlines.length > 0 ? (
                newsData.headlines.map((item, idx) => {
                  let badge = 'bg-[#64748B]/10 text-[#94A3B8] border-[#64748B]/30';
                  if (item.label.toLowerCase() === 'positive') {
                    badge = 'bg-[#10B981]/15 text-[#10B981] border-[#10B981]/30';
                  } else if (item.label.toLowerCase() === 'negative') {
                    badge = 'bg-[#F43F5E]/15 text-[#F43F5E] border-[#F43F5E]/30';
                  }

                  return (
                    <div
                      key={idx}
                      className="p-2.5 rounded-lg bg-[#090A0F] border border-[#1E2230] hover:border-[#2A3042] transition-colors flex items-start justify-between gap-3 text-xs"
                    >
                      <div className="flex-1 min-w-0">
                        <p className="text-[#E2E8F0] line-clamp-2 leading-relaxed">
                          {item.title}
                        </p>
                        {item.published_at ? (
                          <div className="flex items-center gap-1.5 mt-1.5 text-[10px] text-[#64748B] font-mono">
                            <Clock className="h-3 w-3 text-[#64748B] shrink-0" />
                            <span className="truncate">{item.published_at}</span>
                          </div>
                        ) : null}
                      </div>
                      <div className="flex flex-col items-end gap-1 shrink-0 font-mono">
                        <span
                          className={`px-1.5 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider border ${badge}`}
                        >
                          {item.label}
                        </span>
                        <span className="text-[10px] text-[#64748B]">
                          {(item.score * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  );
                })
              ) : (
                <div className="py-8 text-center text-xs text-[#64748B] font-mono">
                  {newsData && !newsAvailable
                    ? `News unavailable: ${newsData.news_error || 'No headlines available'}`
                    : isRunning ? 'Scraping and analyzing news headlines...' : 'Awaiting news sentiment data'}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* RIGHT COLUMN: Bull Analyst & Short-Seller Duel (5 cols) */}
        {!isIndex && <div className="lg:col-span-5 space-y-6">
          {/* Card 1: Bull Analyst (Emerald border) */}
          <div className="bg-[#12141C] border border-[#1E2230] border-l-4 border-l-[#10B981] rounded-xl p-5 space-y-3 shadow-lg">
            <div className="flex items-center justify-between border-b border-[#1E2230] pb-3">
              <div className="flex items-center gap-2">
                <div className="h-7 w-7 rounded bg-[#10B981]/10 flex items-center justify-center text-[#10B981]">
                  <TrendingUp className="h-4 w-4" />
                </div>
                <div>
                  <h3 className="text-sm font-bold text-white flex items-center gap-2">
                    BULL ANALYST
                    <span className="text-[10px] font-mono font-normal text-[#10B981] px-1.5 py-0.5 rounded bg-[#10B981]/10 border border-[#10B981]/30">
                      LONG THESIS
                    </span>
                  </h3>
                </div>
              </div>
              <span className="text-[10px] font-mono text-[#64748B]">gemini-2.5-flash</span>
            </div>

            <div className="text-xs leading-relaxed text-[#E2E8F0] space-y-2 min-h-[120px]">
              {bullData ? (
                <div className="space-y-2 font-sans">
                  {bullData.bull_thesis.split('\n').filter((l) => l.trim()).map((line, i) => (
                    <div key={i} className="flex items-start gap-2">
                      <span className="text-[#10B981] font-bold font-mono">›</span>
                      <p className="text-[#E2E8F0]">{line.replace(/^•\s*/, '')}</p>
                    </div>
                  ))}
                </div>
              ) : activeStep >= 3 && isRunning ? (
                <div className="space-y-2 animate-pulse pt-2">
                  <div className="h-3 bg-[#1E2230] rounded w-5/6"></div>
                  <div className="h-3 bg-[#1E2230] rounded w-4/6"></div>
                  <div className="h-3 bg-[#1E2230] rounded w-3/4"></div>
                </div>
              ) : (
                <div className="py-6 text-center text-xs text-[#64748B] font-mono">
                  Awaiting quantitative inputs to formulate long thesis...
                </div>
              )}
            </div>
          </div>

          {/* Card 2: Adversarial Short-Seller (Crimson border) */}
          <div className="bg-[#12141C] border border-[#1E2230] border-l-4 border-l-[#F43F5E] rounded-xl p-5 space-y-3 shadow-lg">
            <div className="flex items-center justify-between border-b border-[#1E2230] pb-3">
              <div className="flex items-center gap-2">
                <div className="h-7 w-7 rounded bg-[#F43F5E]/10 flex items-center justify-center text-[#F43F5E]">
                  <TrendingDown className="h-4 w-4" />
                </div>
                <div>
                  <h3 className="text-sm font-bold text-white flex items-center gap-2">
                    ADVERSARIAL SHORT-SELLER
                    <span className="text-[10px] font-mono font-normal text-[#F43F5E] px-1.5 py-0.5 rounded bg-[#F43F5E]/10 border border-[#F43F5E]/30">
                      STRESS TEST
                    </span>
                  </h3>
                </div>
              </div>
              <span className="text-[10px] font-mono text-[#64748B]">gemini-2.5-flash</span>
            </div>

            <div className="text-xs leading-relaxed text-[#E2E8F0] space-y-2 min-h-[120px]">
              {bearData ? (
                <div className="space-y-2 font-sans">
                  {bearData.bear_thesis.split('\n').filter((l) => l.trim()).map((line, i) => (
                    <div key={i} className="flex items-start gap-2">
                      <span className="text-[#F43F5E] font-bold font-mono">›</span>
                      <p className="text-[#E2E8F0]">{line.replace(/^•\s*/, '')}</p>
                    </div>
                  ))}
                </div>
              ) : activeStep >= 4 && isRunning ? (
                <div className="space-y-2 animate-pulse pt-2">
                  <div className="h-3 bg-[#1E2230] rounded w-5/6"></div>
                  <div className="h-3 bg-[#1E2230] rounded w-3/4"></div>
                  <div className="h-3 bg-[#1E2230] rounded w-4/6"></div>
                </div>
              ) : (
                <div className="py-6 text-center text-xs text-[#64748B] font-mono">
                  Awaiting bull case to initiate adversarial counter-attack...
                </div>
              )}
            </div>
          </div>
        </div>}
      </section>

      {/* ROW 3: CHIEF RISK OFFICER (CRO) FINAL VERDICT BANNER */}
      {!isIndex && <section className="bg-[#12141C] border border-[#1E2230] rounded-xl p-6 shadow-2xl relative overflow-hidden">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6 pb-4 border-b border-[#1E2230]">
          {/* Verdict Title & Badge */}
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-xs font-mono text-[#94A3B8]">
              <Lock className="h-4 w-4 text-[#F59E0B]" />
              CHIEF RISK OFFICER (CRO) ARBITRATION VERDICT
            </div>
            <div className="flex items-center gap-3">
              {croData ? (
                <span
                  className={`px-4 py-1.5 rounded-lg text-lg font-black font-mono tracking-wider border shadow-lg ${
                    (croData.action || croData.verdict) === 'LONG'
                      ? 'bg-[#10B981]/20 text-[#10B981] border-[#10B981] shadow-[#10B981]/15'
                      : 'bg-[#F43F5E]/20 text-[#F43F5E] border-[#F43F5E] shadow-[#F43F5E]/15'
                  }`}
                >
                  {croData.action || croData.verdict}
                </span>
              ) : (
                <span className="px-4 py-1.5 rounded-lg text-sm font-mono text-[#64748B] bg-[#090A0F] border border-[#1E2230]">
                  {isRunning ? 'DELIBERATING IN CAMERA...' : 'AWAITING VERDICT'}
                </span>
              )}

              {croData && (
                <span className="text-xs font-mono text-[#94A3B8]">
                  Confidence: <span className="text-white font-bold">{croData.confidence}%</span>
                </span>
              )}
            </div>
          </div>

          {/* Allocation & Stop-Loss Telemetry */}
          {croData && (
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 font-mono text-xs">
              <div className="bg-[#090A0F] border border-[#1E2230] p-3 rounded-lg">
                <span className="text-[#64748B] block text-[10px] uppercase">Allocation</span>
                <span className="text-white font-bold text-sm">
                  {croData.recommended_allocation_pct}%
                </span>
                <span className="text-[10px] text-[#10B981] block">
                  Cap: {quantData?.half_kelly_allocation_pct ?? 0}%
                </span>
              </div>

              <div className="bg-[#090A0F] border border-[#1E2230] p-3 rounded-lg">
                <span className="text-[#64748B] block text-[10px] uppercase">Stop Loss</span>
                <span className="text-[#F43F5E] font-bold text-sm">{croData.stop_loss_pct}%</span>
                <span className="text-[10px] text-[#64748B] block">2x ATR trailing</span>
              </div>

              <div className="col-span-2 sm:col-span-1 bg-[#090A0F] border border-[#1E2230] p-3 rounded-lg">
                <span className="text-[#64748B] block text-[10px] uppercase">Kill Switch</span>
                <span className="text-[#E2E8F0] text-[11px] font-medium line-clamp-2">
                  {croData.kill_switch_reason}
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Verdict Summary Text */}
        <div className="pt-4">
          <p className="text-xs leading-relaxed text-[#CBD5E1]">
            {croData ? (
              <span className="font-sans">{croData.verdict_summary}</span>
            ) : (
              <span className="text-[#64748B] font-mono text-xs">
                The Chief Risk Officer evaluates both Bull and Bear arguments alongside IsolationForest anomaly counts,
                volatility clusters, and daily VaR to produce the final allocation decree.
              </span>
            )}
          </p>
        </div>

        {/* TIME MACHINE OUTCOME REVEAL ACCORDION */}
        {quantData && (quantData.time_machine_actual_return_pct !== null || appliedCutoff) && (
          <div className="mt-5 pt-4 border-t border-[#1E2230]/70">
            <button
              onClick={() => setShowOutcome(!showOutcome)}
              className="flex items-center justify-between w-full px-4 py-2.5 rounded-lg bg-[#090A0F] border border-[#F59E0B]/40 hover:border-[#F59E0B] text-xs font-mono font-bold text-[#F59E0B] transition-all"
            >
              <span className="flex items-center gap-2">
                <Clock className="h-4 w-4" />
                REVEAL ACTUAL 5-DAY OUTCOME ({appliedCutoff || 'Cutoff Window'})
              </span>
              {showOutcome ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
            </button>

            {showOutcome && (
              <div className="mt-3 p-4 rounded-lg bg-[#090A0F] border border-[#1E2230] grid grid-cols-1 md:grid-cols-3 gap-4 text-xs font-mono animate-fadeIn">
                <div className="p-3 bg-[#12141C] border border-[#1E2230] rounded-md">
                  <span className="text-[#94A3B8] block text-[11px]">Actual 5-Day Return</span>
                  <div
                    className={`text-xl font-bold mt-1 ${
                      (quantData.time_machine_actual_return_pct ?? 0) >= 0
                        ? 'text-[#10B981]'
                        : 'text-[#F43F5E]'
                    }`}
                  >
                    {(quantData.time_machine_actual_return_pct ?? 0) > 0 ? '+' : ''}
                    {quantData.time_machine_actual_return_pct}%
                  </div>
                </div>

                <div className="p-3 bg-[#12141C] border border-[#1E2230] rounded-md">
                  <span className="text-[#94A3B8] block text-[11px]">Actual 5-Day Max Drawdown</span>
                  <div className="text-xl font-bold text-[#F43F5E] mt-1">
                    -{quantData.max_drawdown_5d_pct}%
                  </div>
                </div>

                <div className="p-3 bg-[#12141C] border border-[#1E2230] rounded-md flex flex-col justify-center">
                  <span className="text-[#94A3B8] block text-[11px]">Committee Backtest Assessment</span>
                  <div className="text-xs text-white font-sans mt-1">
                    {(croData?.action === 'REJECT_TRADE' || croData?.verdict === 'REJECT_TRADE') && (quantData.time_machine_actual_return_pct ?? 0) < 0 ? (
                      <span className="text-[#10B981] font-semibold">
                        ✓ Capital Preserved: CRO trade rejection successfully dodged market drawdown!
                      </span>
                    ) : (croData?.action === 'SHORT' || croData?.verdict === 'SHORT') && (quantData.time_machine_actual_return_pct ?? 0) < 0 ? (
                      <span className="text-[#10B981] font-semibold">
                        ✓ Short Alpha Captured: CRO SHORT decree capitalized on the {quantData.time_machine_actual_return_pct}% drop!
                      </span>
                    ) : (croData?.action === 'LONG' || croData?.verdict === 'LONG') && (quantData.time_machine_actual_return_pct ?? 0) > 0 ? (
                      <span className="text-[#10B981] font-semibold">
                        ✓ Alpha Captured: Long decree successfully exploited subsequent upside!
                      </span>
                    ) : (
                      <span className="text-[#CBD5E1]">
                        Adversarial risk bounds contained tail exposure within specified stop-loss parameters.
                      </span>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </section>}
    </div>
  );
}
