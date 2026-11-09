import asyncio
import json
from backend.graph import stream_committee

async def test_stream():
    print("=== Testing stream_committee('RELIANCE.NS') ===")
    node_count = 0
    async for event in stream_committee("RELIANCE.NS"):
        node = event.get("node")
        node_count += 1
        print(f"[{node_count}] Received node event: {node}")
        if node == "quant":
            print(f"  Price: {event['data']['current_price']}, Vol: {event['data']['annualized_volatility']}%")
        elif node == "news":
            print(f"  Sentiment: {event['data']['aggregate_sentiment_score']}, Headlines: {len(event['data']['headlines'])}")
        elif node == "bull":
            print(f"  Bull Thesis (preview): {event['data']['bull_thesis'][:100]}...")
        elif node == "bear":
            print(f"  Bear Thesis (preview): {event['data']['bear_thesis'][:100]}...")
        elif node == "cro":
            cro = event['data']
            print(f"  CRO Verdict: Action={cro.get('action')}, Confidence={cro.get('confidence')}%, Alloc={cro.get('recommended_allocation_pct')}%")
            print(f"  Stop Loss: {cro.get('stop_loss_pct')}%, Kill Switch: {cro.get('kill_switch_reason')}")
        elif node == "error":
            print(f"  ERROR: {event['data']}")

    print(f"\nCompleted {node_count} node events successfully!")

if __name__ == "__main__":
    asyncio.run(test_stream())
