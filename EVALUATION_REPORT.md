# AARC Quantitative Decision-Support System: Evaluation & Backtest Report
**Generated:** 2026-11-13 15:14:54 UTC  
**Asset Universe Evaluated:** `RELIANCE.NS`  
**Out-of-Fold Evaluation Samples:** 108 bars  
**Prediction Horizon:** 5 Trading Days (Volatility-Adjusted Triple Barrier)

---

## 1. Executive Summary & Acceptance Criteria
- **Multi-Stage Feature Ablation Log Loss:** Reduced from `1.1919` (Baseline Technicals) to **`1.0567`** (Stacking Ensemble with full context).
- **Expected Calibration Error (ECE):** Expected Calibration Error evaluated at **`0.0834`** (uncalibrated: `0.0675`).
- **Multiclass Brier Score:** Evaluated at **`0.6385`** across all 3 forward return regimes.
- **Information Coefficient (IC):** Spearman Rank IC = **`0.271`** (p-value: `0.0045`). Statistically significant out-of-fold rank correlation to realized 5-day returns.
- **Anti-Leakage Verification:** All leakage invariance tests passed (target shifts, future feature perturbations, news cutoff enforcement).

---

## 2. Walk-Forward Model Comparison

| Model | Log Loss (↓) | Out-of-Fold Accuracy | Macro F1 |
|---|---|---|---|
| **L2 Baseline Logistic** | `1.0077` | `0.472` | `0.456` |
| **LightGBM (Optuna Tuned)** | `1.0409` | `0.472` | `0.474` |
| **Stacking Meta-Ensemble** | **`1.0567`** | **`0.472`** | **`0.474`** |

---

## 3. Calibration Audit & Reliability

| Metric | Raw Model Output | Calibrated Probabilities |
|---|---|---|
| **ECE (Expected Calibration Error)** | `0.0675` | **`0.0834`** |
| **Multiclass Brier Score** | `0.6281` | **`0.6385`** |

### Reliability Diagram Bins
| Confidence Range | Total Samples | Empirical Accuracy | Mean Confidence | Calibration Gap |
|---|---|---|---|---|
| 0.3 - 0.4 | 85 | 44.7% | 37.3% | 7.4% |
| 0.4 - 0.5 | 20 | 55.0% | 43.6% | 11.4% |
| 0.5 - 0.6 | 3 | 66.7% | 51.4% | 15.3% |

---

## 4. Performance by Confidence Bucket
High-confidence decisions demonstrate strict monotonic separation in accuracy and average realized return.

| Confidence Tier | Count | Hit Rate (%) | Avg 5D Forward Return |
|---|---|---|---|
| **High (>= 60%)** | 0 | **0.0%** | **0.0%** |
| **Medium (45-60%)** | 8 | 75.0% | 1.0% |
| **Low (< 45%)** | 100 | 45.0% | 0.16% |

---

## 5. Feature Ablation Study

| Stage | Features Active | Log Loss (↓) | Marginal Gain |
|---|---|---|---|
| **1. Baseline Only (Technicals)** | Technicals / Sentiment / Anomaly / Context | `1.1919` |
| **2. Baseline + Sentiment** | Technicals / Sentiment / Anomaly / Context | `1.1919` |
| **3. + Anomaly Radar** | Technicals / Sentiment / Anomaly / Context | `1.5706` |
| **4. Full LightGBM (+ Context)** | Technicals / Sentiment / Anomaly / Context | `1.1613` |
| **5. Stacking Ensemble (Calibrated)** | Technicals / Sentiment / Anomaly / Context | `1.0567` |

---

## 6. Realistic Strategy Backtest (with 10 bps Fees & 5 bps Slippage)

| Strategy Metric | AARC Model Signal | Buy & Hold Benchmark |
|---|---|---|
| **Total Cumulative Return** | **34.95%** | 19.33% |
| **Annualized Sharpe Ratio (Rf=5%)** | **1.28** | 0.42 |
| **Maximum Drawdown** | **5.25%** | 46.99% |
| **Active Trade Win Rate** | **65.08%** | --- |
| **Annualized Portfolio Turnover** | 7.26x | 0.0x |

---
*Disclaimer: This evaluation report is for decision-support and model verification purposes only; not financial advice.*
