# AARC SSE Stream API Specification (`/api/stream`)

The Autonomous Adversarial Risk Committee (AARC) backend exposes a Server-Sent Events (SSE) streaming endpoint:

```http
GET /api/stream?ticker={TICKER}&cutoff_date={YYYY-MM-DD}
```

Each stage of the 5-node risk committee emits a distinct JSON event as `data: <json>\n\n`. All existing frontend contract keys are preserved for 100% backward compatibility, with upgraded quantitative analytics, calibrated probabilities, and explainability drivers.

---

## 1. Node 1: Quant ML Telemetry (`node: "quant"`)

Emitted upon completion of stationary causal feature calculation, GARCH volatility modeling, and Isolation Forest anomaly radar.

```json
{
  "node": "quant",
  "data": {
    "ticker": "RELIANCE.NS",
    "current_price": 2984.50,
    "annualized_volatility": 18.42,
    "daily_VaR_95_pct": -2.15,
    "suggested_stop_loss_atr_pct": 2.85,
    "half_kelly_allocation_pct": 11.45,
    "recent_10d_anomalies": 0,
    "rsi_14": 58.4,
    "recent_5d_return_pct": 1.82,
    "sma20_trend": "BULLISH",
    "close_gt_sma20": true,
    "regime": {
      "state": 0,
      "label": "LOW_VOLATILITY",
      "prob_high_vol": 0.12
    },
    "garch_volatility_forecast_pct": 18.79,
    "chart_data": [
      {
        "time": "2024-05-31",
        "open": 2960.00,
        "high": 2990.00,
        "low": 2950.00,
        "close": 2984.50,
        "volume": 6842100,
        "is_anomaly": false,
        "sma20": 2920.10,
        "sma50": 2880.40,
        "upper_band": 3010.50,
        "lower_band": 2830.00
      }
    ]
  }
}
```

---

## 2. Node 2: FinBERT NLP Sentiment (`node: "news"`)

Emitted upon scoring top news items via ProsusAI/FinBERT with exact publication timestamps and text-hash deduplication.

```json
{
  "node": "news",
  "data": {
    "ticker": "RELIANCE.NS",
    "aggregate_sentiment_score": 0.42,
    "headlines": [
      {
        "title": "Reliance Industries expands green hydrogen capacity with new gigafactory milestone",
        "label": "positive",
        "score": 0.942,
        "published_at": "Fri, 31 May 2024 08:30:00 GMT"
      }
    ]
  }
}
```

---

## 3. Node 3: Bull Analyst Node (`node: "bull"`)

Emitted with 3 institutional quantitative bullets justifying the long position and top positive feature drivers.

```json
{
  "node": "bull",
  "data": {
    "bull_thesis": "• Trend regime is BULLISH with 5-day momentum at +1.82%, supporting an active Half-Kelly capacity allocation of 11.45%.\n• RSI-14 sits at 58.4 with contained 95% daily VaR of -2.15%, providing an asymmetrical entry window against 18.42% annualized vol.\n• Aggregate news sentiment reads +0.42 with 0 price/volume anomalies reflecting institutional accumulation."
  }
}
```

---

## 4. Node 4: Short-Seller Adversary Node (`node: "bear"`)

Emitted with 3 adversarial counterarguments weaponizing VaR, volatility, and distribution anomalies.

```json
{
  "node": "bear",
  "data": {
    "bear_thesis": "• Tail risk is severe: 95% daily VaR of -2.15% with 18.42% annualized vol exposes any 11.45% allocation to fat-tail liquidation.\n• 14-day RSI sits at 58.4; any momentum stalling will trigger abrupt mean-reversion and trap retail longs chasing ₹2984.50.\n• 0 price/volume anomalies in 10 sessions combined with headline sentiment (+0.42) flag institutional distribution risk."
  }
}
```

---

## 5. Node 5: CRO Arbitration Verdict (`node: "cro"`)

Emitted with final arbitration action (`LONG`, `SHORT`, or `REJECT_TRADE`), calibrated probabilities, split conformal prediction set, and transparent risk penalties.

```json
{
  "node": "cro",
  "data": {
    "action": "LONG",
    "verdict": "LONG",
    "confidence": 84,
    "recommended_allocation_pct": 11.45,
    "stop_loss_pct": 2.85,
    "kill_switch_reason": "Breakdown below 20-day SMA or ATR trailing stop (2.85%).",
    "verdict_summary": "Institutional LONG execution approved: Trend structure is constructive (SMA20: BULLISH, Close: 2984.5, RSI: 58.4) with positive news sentiment (+0.42). Half-Kelly capacity allocated at 11.45% with stop loss at 2.85%.",
    "calibrated_probabilities": {
      "DOWN": 0.12,
      "FLAT": 0.22,
      "UP": 0.66
    },
    "conformal_prediction_set": [
      "LONG"
    ],
    "confidence_breakdown": {
      "raw_model_confidence": 84,
      "agreement_penalty": 0.0,
      "anomaly_penalty": 0.0,
      "regime_penalty": 0.0,
      "calibrated_confidence": 84
    },
    "top_drivers": {
      "positive": [
        {"feature": "close_gt_sma20", "value": true, "impact": "+28.4%"},
        {"feature": "sentiment_score", "value": 0.42, "impact": "+22.1%"},
        {"feature": "momentum_5d", "value": 1.82, "impact": "+14.2%"}
      ],
      "negative": [
        {"feature": "daily_VaR_95", "value": -2.15, "impact": "-31.2%"},
        {"feature": "volatility", "value": 18.42, "impact": "-24.5%"},
        {"feature": "isolation_forest_anomalies", "value": 0, "impact": "-20.8%"}
      ]
    }
  }
}
```
