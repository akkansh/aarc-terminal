# AARC Quant Terminal

Autonomous Adversarial Risk Committee (AARC) is a dark-mode stock research dashboard. It combines historical market data, technical/risk metrics, financial-news sentiment, and adversarial LONG/SHORT analysis. It is a research and decision-support tool, **not financial advice or a live execution/trading system**.

## Features

- Analyze one stock ticker at a time, using a preset universe or custom ticker/company search.
- Explore popular US and Indian stocks and Indian indices. Indices are analysis-only: the application does not issue a trade recommendation or allocation for them.
- View daily OHLCV candles, volume, technical overlays, and anomaly markers. Toggle chart indicators without resetting the chart viewport.
- See daily-close dates, ticker/company name, risk metrics, and FinBERT sentiment for relevant news headlines.
- Optionally run a historical cutoff/backtest view.
- Stream analysis stages from the FastAPI backend to the dashboard using Server-Sent Events (SSE).

Market prices and headlines are supplied by third parties (Yahoo Finance through `yfinance` and Google News RSS). Data may be delayed, unavailable, adjusted, or incorrect. Confirm values with your broker or an authoritative source before acting.

## Technology and analysis

- **Backend:** Python, FastAPI, Uvicorn, SSE (`sse-starlette`)
- **Frontend:** Next.js App Router, React, TypeScript, Tailwind CSS, Lightweight Charts, Recharts
- **News sentiment:** `ProsusAI/finbert` classifies financial headlines as positive/neutral/negative.
- **Anomaly detection:** scikit-learn Isolation Forest flags unusual returns/volume.
- **Narrative analysis:** Google Gemini generates Bull/Bear theses and a structured CRO verdict when `GOOGLE_API_KEY` is configured; deterministic fallback text/rules are used otherwise.
- **Quant indicators:** RSI, moving averages, Bollinger Bands, volatility, historical VaR, ATR stop distance, and Half-Kelly sizing.

The backend also contains a separate CLI training/evaluation pipeline for 5-trading-day DOWN/FLAT/UP classification (Logistic Regression + LightGBM stacking; optional GRU), regime/volatility analysis, calibration, and evaluation tools. These CLI models are not the model that powers the dashboard's live LONG/SHORT committee stream.

## Requirements

- Python 3.11+ recommended (the project has been exercised with Python 3.14)
- Node.js 20+
- npm
- Network access for market/news requests and first-time FinBERT model download

There is currently no backend requirements lock file. Create the virtual environment and install the runtime packages as follows:

```bash
python3 -m venv backend/.venv
backend/.venv/bin/python -m pip install --upgrade pip
backend/.venv/bin/pip install \
  fastapi 'uvicorn[standard]' sse-starlette python-dotenv pydantic \
  pandas numpy yfinance scikit-learn feedparser torch transformers \
  google-genai scipy lightgbm optuna arch shap pytest
```

Install the JavaScript dependencies (root scripts and the frontend workspace):

```bash
npm ci
```

For Gemini-generated narratives, create a root `.env` file containing:

```dotenv
GOOGLE_API_KEY=your_google_generative_ai_api_key
```

The app can run without this key, but uses its built-in deterministic narrative/rule fallbacks instead. FinBERT runs on CPU by default (or CUDA if available) and may download model weights on first use.

## Run the application

From the repository root, use the npm scripts to run both services together:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The API runs at [http://localhost:8000](http://localhost:8000), with a health endpoint at `/health` and the SSE analysis endpoint at `/api/stream?ticker=NVDA`.

For a production build and run:

```bash
npm run build
npm start
```

`npm run dev` starts FastAPI with reload and the Next.js development server. `npm start` starts both production servers; run `npm run build` first. To run only one service, use `npm run dev:backend` / `npm run dev:frontend`, or `npm run start:backend` / `npm run start:frontend`. The scripts expect the Python environment at `backend/.venv` (created in the setup steps above).

## Custom stock search

Type a company name (for example, `Broadcom`) or a ticker into the dashboard's search field. Select an equity result to analyze it. Search suggestions come from Yahoo Finance and require network access. Direct ticker entry remains supported. The last selected ticker is saved in browser local storage.

The instrument lookup endpoint is `GET /api/instruments/search?q=Broadcom`. It returns up to eight equity results with ticker, name, and exchange; index search results are excluded. `GET /api/instrument?ticker=AVGO` resolves a display name for the exact ticker when Yahoo Finance provides it.

## CLI training, evaluation, and tests

Run commands from the repository root with the backend virtual environment:

```bash
backend/.venv/bin/python -m backend.cli train --ticker RELIANCE.NS --horizon 5
backend/.venv/bin/python -m backend.cli eval --ticker RELIANCE.NS --horizon 5
backend/.venv/bin/python -m backend.cli test
```

The training/evaluation commands fetch historical OHLCV, benchmark/VIX context, fundamentals, and news, then build causal features and use purged walk-forward validation. Training writes model artifacts under `backend/artifacts/`; evaluation writes a Markdown report (default `EVALUATION_REPORT.md`).

Direct test commands:

```bash
backend/.venv/bin/python -m pytest backend/tests -q
cd frontend && npm run build
```

## Project layout

```text
backend/
  main.py             FastAPI API and SSE endpoint
  graph.py            Sequential committee stream
  tools/              Quant metrics and news sentiment
  data/               Market/news ingestion and disk cache
  features/           Feature engineering and labels for CLI models
  models/              Classifiers, ensemble, and regime models
  tests/               Backend tests
frontend/
  src/app/             Dashboard page
  src/components/      Charts and UI components
package.json          Root npm scripts for the API and frontend
```

## Data and limitations

- Price charts and technical values are based on daily historical data, not streaming quotes. The UI marks the latest available daily close and its date.
- News sentiment depends on Google News RSS results and FinBERT availability. If data or the model is unavailable, news is shown as unavailable rather than as a fabricated neutral score.
- Historical news cutoff mode excludes headlines after the cutoff (and headlines without a verifiable publication date).
- The demo cache stores successful committee runs in `backend/demo_cache.json`; it is a generated runtime artifact, not a dependable market-data source.
- An index (ticker starting with `^`) is for benchmark analysis only; the committee does not issue trade theses or allocations for it.
- Validate ticker symbols and all outputs independently. No backtest or model result guarantees future performance.
