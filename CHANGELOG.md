# CHANGELOG: AARC Quantitative Decision-Support System Upgrade

All notable changes and architectural upgrades to the Autonomous Adversarial Risk Committee (AARC) trading decision system are documented herein.

---

## [Phase 1] Architecture, Configuration & Data Ingestion
- **Config Management (`config.yaml`)**: Centralized horizon (5d), triple-barrier volatility factor (0.5σ), technical windows (RSI-14, SMA-20, SMA-50, Bollinger 20/2.0), model toggles, and seeds.
- **Thread-Safe Disk Cache (`backend/data/cache.py`)**: Persistent SHA-256 keyed cache with expiration TTL, atomic disk writes, and cross-thread safety.
- **Market Data Fetcher (`backend/data/fetcher.py`)**:
  - Resilient ingestion with auto-fallback between `yf.download` and `yf.Ticker.history`.
  - Ingestion of market context: Indian benchmark (`^NSEI`), US benchmark (`^GSPC`), Indian VIX (`^INDIAVIX`), US VIX (`^VIX`).
  - Strict cutoff date handling for time-machine backtesting with zero future leakage.
  - Google News RSS scraper extracting exact `published_at` timestamps, article URLs, and title deduplication.

---

## [Phase 2] Stationary Feature Engineering & Anti-Leakage Gates
- **Stationary Technical Features (`backend/features/technical.py`)**:
  - Replaced raw non-stationary prices with 1d, 5d, and 20d log returns.
  - Normalized RSI bounded in $[-1.0, 1.0]$.
  - Price-normalized MACD histogram, Bollinger %B and normalized bandwidth.
  - Volume Z-score, High-Low Parkinson spread ratio, moving average ratios (`Close/SMA20`, `Close/SMA50`, `SMA20/SMA50`).
- **Rolling Isolation Forest (`backend/features/anomaly.py`)**:
  - Eliminates look-ahead bias by fitting strictly on trailing causal historical bars.
  - Periodic refitting every 5 bars to eliminate execution latency while maintaining causal guarantees.
- **FinBERT NLP Sentiment Engine (`backend/features/sentiment.py`)**:
  - Singleton loader with text-hash caching.
  - Extracts full 3-class probability distributions: $P(\text{positive}), P(\text{neutral}), P(\text{negative})$.
  - Computes rolling 3d, 7d, and 14d sentiment momentum and cross-source headline dispersion.
- **Strict Invariance Unit Tests (`backend/tests/test_leakage.py`, `backend/tests/test_features.py`)**:
  - Test verifying feature invariance under future price perturbations.
  - Test verifying news items published after target timestamps are purged.
  - Test verifying triple-barrier forward return labels require future data.

---

## [Phase 3] Walk-Forward Models & Regime Awareness
- **Vectorized 2-State Gaussian HMM (`backend/models/regime.py`)**:
  - Pure NumPy implementation of Baum-Welch EM and Forward-Backward algorithm for volatility regime detection.
  - Analytical GARCH(1,1) volatility forecasting.
- **Purged Walk-Forward Cross Validation (`backend/eval/walk_forward.py`)**:
  - Strict temporal ordering with an embargo gap equal to prediction horizon (5 days).
- **Optuna-Tuned LightGBM (`backend/models/tree.py`)**:
  - Hyperparameter optimization across walk-forward folds targeting multi-class log loss.
- **Stacking Meta-Ensemble (`backend/models/ensemble.py`)**:
  - Optimal convex combination of out-of-fold probability distributions from L2 Logistic Baseline and LightGBM.
  - Model bundle serialization and versioning with git commit hash.

---

## [Phase 4] Probability Calibration & Conformal Prediction
- **Temperature Scaling (`backend/calibration/calibrator.py`)**:
  - Temperature Scaling (Platt scaling extension) that preserves monotonic class ranking while softening overconfidence.
  - Evaluated on holdout data with ECE and multiclass Brier score audits.
- **Split Conformal Prediction (`backend/calibration/conformal.py`)**:
  - Generates adaptive prediction sets (e.g. `["LONG"]`, `["LONG", "FLAT"]`) with 90% finite-sample coverage guarantees.
- **Transparent Calibrated Confidence (`backend/calibration/conformal.py`)**:
  - Applies transparent, auditable penalties for ensemble disagreement, microstructural anomalies, and high-volatility regimes.
  - System abstains (`HOLD`) when conformal prediction sets are ambiguous or confidence falls below threshold.

---

## [Phase 5] Grounded Explainability & Evaluation Report
- **TreeSHAP Explanations (`backend/explain/shap_explainer.py`)**:
  - Computes top 5 positive and negative feature drivers.
- **Grounded Gemini Synthesis (`backend/explain/gemini_synthesis.py`)**:
  - Strict Pydantic JSON schema and deterministic numerical fallback narrative.
- **Comprehensive Evaluation Runner (`backend/eval/evaluate.py`)**:
  - Automates reliability diagrams, confidence bucket analysis, feature ablation, and realistic backtests with 10 bps transaction costs and 5 bps slippage.
  - Generates `EVALUATION_REPORT.md`.

---

## [Phase 6] SSE Streaming Engine & CLI Deliverables
- **Unified Pipeline (`backend/graph.py`, `backend/main.py`)**:
  - SSE endpoint (`/api/stream`) streams 5 stages (`quant` $\rightarrow$ `news` $\rightarrow$ `bull` $\rightarrow$ `bear` $\rightarrow$ `cro`).
  - Seamlessly emits calibrated probabilities, conformal sets, and SHAP drivers while maintaining 100% backward compatibility with Next.js frontend.
- **Unified Makefile & CLI (`backend/cli.py`, `Makefile`)**:
  - `make test`: Executes pytest anti-leakage suite.
  - `make train`: Trains walk-forward ensemble and persists model bundle.
  - `make eval`: Executes comprehensive evaluation and updates `EVALUATION_REPORT.md`.
  - `make run`: Boots FastAPI backend (8000) and Next.js frontend (3000).
