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
} from 'lightweight-charts';
import { AlertTriangle, Eye, EyeOff } from 'lucide-react';

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

export default function CandlestickChart({
  data,
  currencySymbol = '₹',
  ticker = '',
}: CandlestickChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  // Indicator Visibility Toggles
  const [showBollinger, setShowBollinger] = useState(true);
  const [showSma50, setShowSma50] = useState(true);
  const [showVolume, setShowVolume] = useState(true);

  // Hovered item for terminal telemetry
  const [hoveredPoint, setHoveredPoint] = useState<ChartDataPoint | null>(null);

  // Latest point as default
  const latestPoint = data && data.length > 0 ? data[data.length - 1] : null;
  const activePoint = hoveredPoint || latestPoint;

  useEffect(() => {
    if (!containerRef.current || !data || data.length === 0) return;

    // Filter and sort data by date ascending, ensure uniqueness
    const sortedData = [...data]
      .sort((a, b) => a.date.localeCompare(b.date))
      .filter((item, index, self) => index === self.findIndex((t) => t.date === item.date));

    if (sortedData.length === 0) return;

    // Clean existing chart if present
    if (chartRef.current) {
      chartRef.current.remove();
      chartRef.current = null;
    }

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

    const candleData = sortedData.map((d) => ({
      time: d.date,
      open: typeof d.open === 'number' ? d.open : d.close,
      high: typeof d.high === 'number' ? d.high : d.close,
      low: typeof d.low === 'number' ? d.low : d.close,
      close: d.close,
    }));

    candleSeries.setData(candleData);

    // 2. Anomaly markers on top of candles
    const anomalyMarkers = sortedData
      .filter((d) => d.is_anomaly)
      .map((d) => ({
        time: d.date,
        position: 'aboveBar' as const,
        color: '#F43F5E',
        shape: 'circle' as const,
        text: 'ANOMALY',
        size: 1.2,
      }));

    if (anomalyMarkers.length > 0) {
      createSeriesMarkers(candleSeries, anomalyMarkers);
    }

    // 3. Volume Histogram at bottom
    if (showVolume) {
      const volumeSeries = chart.addSeries(HistogramSeries, {
        priceFormat: { type: 'volume' },
        priceScaleId: 'volume',
      });

      chart.priceScale('volume').applyOptions({
        scaleMargins: {
          top: 0.8,
          bottom: 0,
        },
      });

      const volumeData = sortedData.map((d) => {
        const o = typeof d.open === 'number' ? d.open : d.close;
        const isUp = d.close >= o;
        return {
          time: d.date,
          value: d.volume,
          color: isUp ? 'rgba(16, 185, 129, 0.35)' : 'rgba(244, 63, 94, 0.35)',
        };
      });

      volumeSeries.setData(volumeData);
    }

    // 4. Bollinger Bands (Upper, Middle/SMA20, Lower)
    if (showBollinger) {
      const bbUpperSeries = chart.addSeries(LineSeries, {
        color: '#818CF8',
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        title: 'BB Upper',
        priceLineVisible: false,
      });
      const bbMiddleSeries = chart.addSeries(LineSeries, {
        color: '#F59E0B',
        lineWidth: 1,
        lineStyle: LineStyle.Dotted,
        title: 'SMA 20',
        priceLineVisible: false,
      });
      const bbLowerSeries = chart.addSeries(LineSeries, {
        color: '#818CF8',
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        title: 'BB Lower',
        priceLineVisible: false,
      });

      const upperData = sortedData
        .filter((d) => typeof d.bb_upper === 'number' && d.bb_upper > 0)
        .map((d) => ({ time: d.date, value: d.bb_upper! }));
      const middleData = sortedData
        .filter((d) => typeof d.bb_middle === 'number' && d.bb_middle > 0)
        .map((d) => ({ time: d.date, value: d.bb_middle! }));
      const lowerData = sortedData
        .filter((d) => typeof d.bb_lower === 'number' && d.bb_lower > 0)
        .map((d) => ({ time: d.date, value: d.bb_lower! }));

      bbUpperSeries.setData(upperData);
      bbMiddleSeries.setData(middleData);
      bbLowerSeries.setData(lowerData);
    }

    // 5. 50-day Moving Average (SMA50)
    if (showSma50) {
      const sma50Series = chart.addSeries(LineSeries, {
        color: '#06B6D4',
        lineWidth: 2,
        lineStyle: LineStyle.Solid,
        title: 'SMA 50',
        priceLineVisible: false,
      });

      const sma50Data = sortedData
        .filter((d) => typeof d.sma_50 === 'number' && d.sma_50 > 0)
        .map((d) => ({ time: d.date, value: d.sma_50! }));

      sma50Series.setData(sma50Data);
    }

    // Crosshair move handler
    chart.subscribeCrosshairMove((param) => {
      if (!param || !param.time) {
        setHoveredPoint(null);
        return;
      }
      const timeStr = String(param.time);
      const match = sortedData.find((d) => d.date === timeStr);
      if (match) {
        setHoveredPoint(match);
      } else {
        setHoveredPoint(null);
      }
    });

    chart.timeScale().fitContent();

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
      if (chartRef.current) {
        chartRef.current.remove();
        chartRef.current = null;
      }
    };
  }, [data, showBollinger, showSma50, showVolume]);

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
                {currencySymbol}
                {(activePoint.open ?? activePoint.close).toFixed(2)}
              </span>
            </span>
            <span className="text-[#64748B]">
              H:{' '}
              <span className="text-white">
                {currencySymbol}
                {(activePoint.high ?? activePoint.close).toFixed(2)}
              </span>
            </span>
            <span className="text-[#64748B]">
              L:{' '}
              <span className="text-white">
                {currencySymbol}
                {(activePoint.low ?? activePoint.close).toFixed(2)}
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
                {currencySymbol}
                {activePoint.close.toFixed(2)}
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
              </span>
            )}
            {showSma50 && typeof activePoint.sma_50 === 'number' && (
              <span className="text-[#06B6D4] text-[10px]">
                SMA50: {activePoint.sma_50?.toFixed(1)}
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
