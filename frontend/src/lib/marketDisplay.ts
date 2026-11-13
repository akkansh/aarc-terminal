/**
 * Display helpers shared by the terminal dashboard and the candlestick chart.
 * These keep index units and daily-close dates consistent across the UI.
 */

const INDIAN_INDEX_TICKERS = new Set<string>([
  '^NSEI',
  '^NSEBANK',
  '^BSESN',
  '^INDIAVIX',
]);

export function isIndexTicker(ticker: string): boolean {
  return ticker.trim().startsWith('^');
}

export function isIndianTicker(ticker: string): boolean {
  const clean = ticker.trim().toUpperCase();
  return (
    clean.endsWith('.NS') ||
    clean.endsWith('.BO') ||
    INDIAN_INDEX_TICKERS.has(clean)
  );
}

export function currencySymbolForTicker(ticker: string): string {
  if (isIndexTicker(ticker)) return '';
  return isIndianTicker(ticker) ? '₹' : '$';
}

export function priceUnitForTicker(ticker: string): string {
  return isIndexTicker(ticker) ? 'pts' : '';
}

export function formatMarketValue(value: number, ticker: string): string {
  if (!Number.isFinite(value)) return '---';
  const formatted = value.toLocaleString(undefined, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
  const unit = priceUnitForTicker(ticker);
  return `${currencySymbolForTicker(ticker)}${formatted}${unit ? ` ${unit}` : ''}`;
}

export function latestChartPoint<T extends { date: string }>(
  points: readonly T[] | null | undefined,
): T | null {
  if (!points || points.length === 0) return null;
  return points.reduce((latest, point) =>
    point.date > latest.date ? point : latest,
  );
}

export function sortUniqueChartData<T extends { date: string }>(
  points: readonly T[],
): T[] {
  const seen = new Set<string>();
  return [...points]
    .sort((a, b) => a.date.localeCompare(b.date))
    .filter((item) => {
      if (seen.has(item.date)) return false;
      seen.add(item.date);
      return true;
    });
}
