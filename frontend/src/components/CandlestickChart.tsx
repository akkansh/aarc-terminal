'use client';

import React, { useEffect, useRef, useState } from 'react';
import {
  createChart,
  CandlestickSeries,
  HistogramSeries,
  LineSeries,
  ColorType,
  LineStyle,
  CrosshairMode,
  createSeriesMarkers,
  IChartApi,
  ISeriesApi,
  ISeriesMarkersPluginApi,
  Time,
} from 'lightweight-charts';
import { AlertTriangle } from 'lucide-react';
import { isIndexTicker, latestChartPoint, sortUniqueChartData } from '@/lib/marketDisplay';

export interface ChartDataPoint {
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

interface CandlestickChartProps {
  data: ChartDataPoint[];
  currencySymbol?: string;
  ticker?: string;
}

type CandleSeriesApi = ISeriesApi<'Candlestick'>;
type HistogramSeriesApi = ISeriesApi<'Histogram'>;
type LineSeriesApi = ISeriesApi<'Line'>;

export default function CandlestickChart({
  data,
  currencySymbol = '₹',
  ticker = '',
}: CandlestickChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<CandleSeriesApi | null>(null);
  const volumeSeriesRef = useRef<HistogramSeriesApi | null>(null);
  const bbUpperSeriesRef = useRef<LineSeriesApi | null>(null);
  const bbMiddleSeriesRef = useRef<LineSeriesApi | null>(null);
  const bbLowerSeriesRef = useRef<LineSeriesApi | null>(null);
  const sma50SeriesRef = useRef<LineSeriesApi | null>(null);
  const markersRef = useRef<ISeriesMarkersPluginApi<Time> | null>(null);
  const sortedDataRef = useRef<ChartDataPoint[]>([]);
  const lastTickerRef = useRef<string | null>(null);

  // Indicator Visibility Toggles
  const [showBollinger, setShowBollinger] = useState(true);
  const [showSma50, setShowSma50] = useState(true);
  const [showVolume, setShowVolume] = useState(true);

  // Hovered item for terminal telemetry
  const [hoveredPoint, setHoveredPoint] = useState<ChartDataPoint | null>(null);

  // Latest point as default (by date, not array order)
  const latestPoint = latestChartPoint(data);
  const activePoint = hoveredPoint || latestPoint;
  const priceUnit = isIndexTicker(ticker) ? ' pts' : '';

  const formatPrice = (value: number) =>
    `${currencySymbol}${value.toFixed(2)}${priceUnit}`;

  // Create the chart and all series once. Indicator toggles only change series
  // visibility, so the viewport (pan/zoom) survives every toggle.
  useEffect(() => {
    if (!containerRef.current) return;

    const container = containerRef.current;
    const width = container.clientWidth || 600;
    const height = 310;

    const chart = createChart(container, {
      width,
      height,
      layout: {
        background: { type: ColorType.Solid, color: '#090A0F' },
        textColor: '#64748B',
        fontFamily: 'monospace',
        fontSize: 10,
      },
      grid: {
        vertLines: { color: '#161923' },
        horzLines: { color: '#161923' },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: {
          color: '#475569',
          width: 1,
          style: LineStyle.Dashed,
          labelBackgroundColor: '#1E2230',
        },
        horzLine: {
          color: '#475569',
          width: 1,
          style: LineStyle.Dashed,
          labelBackgroundColor: '#1E2230',
        },
      },
      rightPriceScale: {
        borderColor: '#1E2230',
        scaleMargins: {
          top: 0.08,
          bottom: showVolume ? 0.22 : 0.08,
        },
      },
      timeScale: {
        borderColor: '#1E2230',
        timeVisible: true,
      },
      handleScroll: true,
      handleScale: true,
    });

    chartRef.current = chart;

    // 1. Candlestick Series
    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#10B981',
      downColor: '#F43F5E',
      borderVisible: true,
      borderUpColor: '#10B981',
      borderDownColor: '#F43F5E',
      wickUpColor: '#10B981',
      wickDownColor: '#F43F5E',
      priceFormat: {
        type: 'price',
        precision: 2,
        minMove: 0.05,
      },
    });
    candleSeriesRef.current = candleSeries;

    // 2. Volume Histogram at bottom
    const volumeSeries = chart.addSeries(HistogramSeries, {
      priceFormat: { type: 'volume' },
      priceScaleId: 'volume',
      visible: showVolume,
    });
    volumeSeriesRef.current = volumeSeries;

    chart.priceScale('volume').applyOptions({
      scaleMargins: {
        top: 0.8,
        bottom: 0,
      },
    });

    // 3. Bollinger Bands (Upper, Middle/SMA20, Lower)
    const bbUpperSeries = chart.addSeries(LineSeries, {
      color: '#818CF8',
      lineWidth: 1,
      lineStyle: LineStyle.Dashed,
      title: 'BB Upper',
      priceLineVisible: false,
      visible: showBollinger,
    });
    const bbMiddleSeries = chart.addSeries(LineSeries, {
      color: '#F59E0B',
      lineWidth: 1,
      lineStyle: LineStyle.Dotted,
      title: 'SMA 20',
      priceLineVisible: false,
      visible: showBollinger,
    });
    const bbLowerSeries = chart.addSeries(LineSeries, {
      color: '#818CF8',
      lineWidth: 1,
      lineStyle: LineStyle.Dashed,
      title: 'BB Lower',
      priceLineVisible: false,
      visible: showBollinger,
    });
    bbUpperSeriesRef.current = bbUpperSeries;
    bbMiddleSeriesRef.current = bbMiddleSeries;
    bbLowerSeriesRef.current = bbLowerSeries;

    // 4. 50-day Moving Average (SMA50)
    const sma50Series = chart.addSeries(LineSeries, {
      color: '#06B6D4',
      lineWidth: 2,
      lineStyle: LineStyle.Solid,
      title: 'SMA 50',
      priceLineVisible: false,
      visible: showSma50,
    });
    sma50SeriesRef.current = sma50Series;

    // Anomaly markers live on the candle series and are refreshed with the data.
    markersRef.current = createSeriesMarkers(candleSeries, []);

    // Crosshair move handler
    chart.subscribeCrosshairMove((param) => {
      if (!param || !param.time) {
        setHoveredPoint(null);
        return;
      }
      const timeStr = String(param.time);
      const match = sortedDataRef.current.find((d) => d.date === timeStr);
      setHoveredPoint(match || null);
    });

    // Resize handling
    const resizeObserver = new ResizeObserver((entries) => {
      if (!entries || entries.length === 0 || !entries[0].contentRect) return;
      const { width: newWidth } = entries[0].contentRect;
      if (newWidth > 0 && chartRef.current) {
        chartRef.current.applyOptions({ width: newWidth });
      }
    });

    resizeObserver.observe(container);

    return () => {
      resizeObserver.disconnect();
      if (markersRef.current) {
        markersRef.current.detach();
        markersRef.current = null;
      }
      if (chartRef.current) {
        chartRef.current.remove();
        chartRef.current = null;
      }
      candleSeriesRef.current = null;
      volumeSeriesRef.current = null;
      bbUpperSeriesRef.current = null;
      bbMiddleSeriesRef.current = null;
      bbLowerSeriesRef.current = null;
      sma50SeriesRef.current = null;
      sortedDataRef.current = [];
      lastTickerRef.current = null;
    };
    // Chart is created once; indicator visibility is applied by the effect below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Refresh series data without recreating the chart. Preserve the visible range
  // unless the user has selected a different ticker.
  useEffect(() => {
    const candleSeries = candleSeriesRef.current;
    if (!candleSeries) return;

    const sortedData = sortUniqueChartData(data || []);
    sortedDataRef.current = sortedData;

    candleSeries.setData(
      sortedData.map((d) => ({
        time: d.date,
        open: typeof d.open === 'number' ? d.open : d.close,
        high: typeof d.high === 'number' ? d.high : d.close,
        low: typeof d.low === 'number' ? d.low : d.close,
        close: d.close,
      })),
    );

    volumeSeriesRef.current?.setData(
      sortedData.map((d) => {
        const open = typeof d.open === 'number' ? d.open : d.close;
        const isUp = d.close >= open;
        return {
          time: d.date,
          value: d.volume,
          color: isUp ? 'rgba(16, 185, 129, 0.35)' : 'rgba(244, 63, 94, 0.35)',
        };
      }),
    );

    bbUpperSeriesRef.current?.setData(
      sortedData
        .filter((d) => typeof d.bb_upper === 'number' && d.bb_upper > 0)
        .map((d) => ({ time: d.date, value: d.bb_upper! })),
    );
    bbMiddleSeriesRef.current?.setData(
      sortedData
        .filter((d) => typeof d.bb_middle === 'number' && d.bb_middle > 0)
        .map((d) => ({ time: d.date, value: d.bb_middle! })),
    );
    bbLowerSeriesRef.current?.setData(
      sortedData
        .filter((d) => typeof d.bb_lower === 'number' && d.bb_lower > 0)
        .map((d) => ({ time: d.date, value: d.bb_lower! })),
    );
    sma50SeriesRef.current?.setData(
      sortedData
        .filter((d) => typeof d.sma_50 === 'number' && d.sma_50 > 0)
        .map((d) => ({ time: d.date, value: d.sma_50! })),
    );

    markersRef.current?.setMarkers(
      sortedData
        .filter((d) => d.is_anomaly)
        .map((d) => ({
          time: d.date,
          position: 'aboveBar' as const,
          color: '#F43F5E',
          shape: 'circle' as const,
          text: 'ANOMALY',
          size: 1.2,
        })),
    );

    setHoveredPoint(null);

    if (lastTickerRef.current !== ticker) {
      lastTickerRef.current = ticker;
      chartRef.current?.timeScale().fitContent();
    }
  }, [data, ticker]);

  // Indicator toggles only apply visibility; the chart and viewport stay intact.
  useEffect(() => {
    bbUpperSeriesRef.current?.applyOptions({ visible: showBollinger });
    bbMiddleSeriesRef.current?.applyOptions({ visible: showBollinger });
    bbLowerSeriesRef.current?.applyOptions({ visible: showBollinger });
    sma50SeriesRef.current?.applyOptions({ visible: showSma50 });
    volumeSeriesRef.current?.applyOptions({ visible: showVolume });
    chartRef.current?.applyOptions({
      rightPriceScale: {
        scaleMargins: {
          top: 0.08,
          bottom: showVolume ? 0.22 : 0.08,
        },
      },
    });
  }, [showBollinger, showSma50, showVolume]);

  return (
    <div className="space-y-2.5">
      {/* Chart Top Bar: Controls & Interactive Legend */}
      <div className="flex flex-wrap items-center justify-between gap-3 text-xs font-mono border-b border-[#1E2230] pb-2.5">
        {/* Left: Indicator Toggles */}
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setShowBollinger(!showBollinger)}
            className={`px-2 py-1 rounded text-[11px] font-semibold flex items-center gap-1.5 transition-colors border ${
              showBollinger
                ? 'bg-[#818CF8]/20 text-[#A5B4FC] border-[#818CF8]/50'
                : 'bg-[#12141C] text-[#64748B] border-[#1E2230] hover:text-[#94A3B8]'
            }`}
          >
            <span className="h-1.5 w-1.5 rounded-full bg-[#818CF8]" />
            BB (20, 2)
          </button>

          <button
            type="button"
            onClick={() => setShowSma50(!showSma50)}
            className={`px-2 py-1 rounded text-[11px] font-semibold flex items-center gap-1.5 transition-colors border ${
              showSma50
                ? 'bg-[#06B6D4]/20 text-[#22D3EE] border-[#06B6D4]/50'
                : 'bg-[#12141C] text-[#64748B] border-[#1E2230] hover:text-[#94A3B8]'
            }`}
          >
            <span className="h-1.5 w-1.5 rounded-full bg-[#06B6D4]" />
            SMA 50
          </button>

          <button
            type="button"
            onClick={() => setShowVolume(!showVolume)}
            className={`px-2 py-1 rounded text-[11px] font-semibold flex items-center gap-1.5 transition-colors border ${
              showVolume
                ? 'bg-[#10B981]/20 text-[#10B981] border-[#10B981]/50'
                : 'bg-[#12141C] text-[#64748B] border-[#1E2230] hover:text-[#94A3B8]'
            }`}
          >
            VOL Bars
          </button>
        </div>

        {/* Right: Anomaly Legend */}
        <div className="flex items-center gap-3 text-[11px]">
          <span className="flex items-center gap-1 text-[#10B981]">
            <span className="inline-block h-2 w-2 bg-[#10B981] rounded-sm" /> Up
          </span>
          <span className="flex items-center gap-1 text-[#F43F5E]">
            <span className="inline-block h-2 w-2 bg-[#F43F5E] rounded-sm" /> Down
          </span>
          <span className="flex items-center gap-1 text-[#F43F5E] font-bold">
            <span className="inline-block h-2 w-2 rounded-full bg-[#F43F5E] ring-2 ring-[#F43F5E]/40" />
            Anomaly
          </span>
        </div>
      </div>

      {/* Live Candlestick Telemetry Ribbon */}
      {activePoint && (
        <div className="flex flex-wrap items-center justify-between bg-[#090A0F] border border-[#1E2230] rounded-lg px-3 py-1.5 text-[11px] font-mono gap-y-1">
          <div className="flex items-center gap-3">
            <span className="text-[#94A3B8] font-bold">{activePoint.date}</span>
            <span className="text-[#64748B]">
              O:{' '}
              <span className="text-white">
                {formatPrice(activePoint.open ?? activePoint.close)}
              </span>
            </span>
            <span className="text-[#64748B]">
              H:{' '}
              <span className="text-white">
                {formatPrice(activePoint.high ?? activePoint.close)}
              </span>
            </span>
            <span className="text-[#64748B]">
              L:{' '}
              <span className="text-white">
                {formatPrice(activePoint.low ?? activePoint.close)}
              </span>
            </span>
            <span className="text-[#64748B]">
              C:{' '}
              <span
                className={`font-bold ${
                  activePoint.close >= (activePoint.open ?? activePoint.close)
                    ? 'text-[#10B981]'
                    : 'text-[#F43F5E]'
                }`}
              >
                {formatPrice(activePoint.close)}
              </span>
            </span>
            <span className="text-[#64748B]">
              V:{' '}
              <span className="text-[#E2E8F0]">
                {activePoint.volume.toLocaleString()}
              </span>
            </span>
          </div>

          <div className="flex items-center gap-3">
            {showBollinger && typeof activePoint.bb_upper === 'number' && (
              <span className="text-[#818CF8] text-[10px]">
                BB: [{activePoint.bb_lower?.toFixed(1)} - {activePoint.bb_upper?.toFixed(1)}]
                {priceUnit}
              </span>
            )}
            {showSma50 && typeof activePoint.sma_50 === 'number' && (
              <span className="text-[#06B6D4] text-[10px]">
                SMA50: {activePoint.sma_50?.toFixed(1)}
                {priceUnit}
              </span>
            )}
            {activePoint.is_anomaly && (
              <span className="px-1.5 py-0.5 rounded bg-[#F43F5E]/20 text-[#F43F5E] border border-[#F43F5E] text-[10px] font-bold flex items-center gap-1 animate-pulse">
                <AlertTriangle className="h-3 w-3" />
                IFOREST ANOMALY
              </span>
            )}
          </div>
        </div>
      )}

      {/* Chart Canvas Container */}
      <div
        ref={containerRef}
        className="w-full h-[310px] relative rounded-lg overflow-hidden border border-[#1E2230]/60 bg-[#090A0F]"
      />
    </div>
  );
}
