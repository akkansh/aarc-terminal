import sys
import json
from backend.tools.quant_math import analyze_ticker_quant
from backend.tools.sentiment_nlp import analyze_ticker_sentiment

def main():
    print("=== Testing quant_math.py with RELIANCE.NS ===")
    quant_res = analyze_ticker_quant("RELIANCE.NS")
    print(f"Ticker: {quant_res['ticker']}")
    print(f"Current Price: {quant_res['current_price']}")
    print(f"Annualized Volatility: {quant_res['annualized_volatility']}%")
    print(f"Daily VaR (95%): {quant_res['daily_VaR_95_pct']}%")
    print(f"Suggested Stop Loss (2x ATR): {quant_res['suggested_stop_loss_atr_pct']}%")
    print(f"Half-Kelly Allocation: {quant_res['half_kelly_allocation_pct']}%")
    print(f"Recent 10d Anomalies: {quant_res['recent_10d_anomalies']}")
    print(f"Chart Data Points: {len(quant_res['chart_data'])}")
    if quant_res['chart_data']:
        print(f"Sample Chart Point: {quant_res['chart_data'][-1]}")

    print("\n=== Testing quant_math.py with cutoff_date ===")
    # Test cutoff date e.g. 2024-03-01
    quant_cutoff = analyze_ticker_quant("RELIANCE.NS", cutoff_date="2024-03-01")
    print(f"Cutoff 2024-03-01 Actual Return: {quant_cutoff['time_machine_actual_return_pct']}%")
    print(f"Cutoff 2024-03-01 Max Drawdown: {quant_cutoff['max_drawdown_5d_pct']}%")

    print("\n=== Testing sentiment_nlp.py with RELIANCE.NS ===")
    sent_res = analyze_ticker_sentiment("RELIANCE.NS")
    print(f"Aggregate Sentiment Score: {sent_res['aggregate_sentiment_score']}")
    print(f"Headlines Scraped & Scored: {len(sent_res['headlines'])}")
    for i, h in enumerate(sent_res['headlines'][:3]):
        print(f"  [{i+1}] {h['label'].upper()} ({h['score']}): {h['title']}")

    # Verify JSON serialization without NaN
    json_str = json.dumps({"quant": quant_res, "sentiment": sent_res}, allow_nan=False)
    print(f"\nJSON serialization successful! Total payload bytes: {len(json_str)}")
    print("\nALL TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
