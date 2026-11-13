"""Command Line Interface for AARC quantitative training, evaluation, and backtesting."""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="AARC Quantitative Decision-Support System CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train command
    train_parser = subparsers.add_parser("train", help="Train walk-forward models and persist artifacts")
    train_parser.add_argument("--ticker", type=str, default="RELIANCE.NS", help="Asset ticker")
    train_parser.add_argument("--splits", type=int, default=3, help="Number of walk-forward splits")
    train_parser.add_argument("--horizon", type=int, default=5, help="Prediction horizon in trading days")

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Run comprehensive evaluation and generate markdown report")
    eval_parser.add_argument("--ticker", type=str, default="RELIANCE.NS", help="Asset ticker")
    eval_parser.add_argument("--splits", type=int, default=3, help="Number of walk-forward splits")
    eval_parser.add_argument("--horizon", type=int, default=5, help="Prediction horizon in trading days")
    eval_parser.add_argument("--output", type=str, default="EVALUATION_REPORT.md", help="Output report markdown path")

    # Leakage test command
    test_parser = subparsers.add_parser("test", help="Run pytest anti-leakage and feature tests")

    args = parser.parse_args()

    if args.command == "train":
        from backend.data.fetcher import MarketDataFetcher
        from backend.features.pipeline import FeaturePipeline
        from backend.eval.walk_forward import PurgedWalkForwardCV
        from backend.models.ensemble import StackingEnsemble

        print(f"[CLI TRAIN] Ingesting data for {args.ticker}...")
        fetcher = MarketDataFetcher()
        stock_df = fetcher.fetch_ohlcv(args.ticker, period="2y")
        context = fetcher.fetch_market_context(args.ticker, period="2y")
        fundamentals = fetcher.fetch_fundamentals(args.ticker)
        news = fetcher.fetch_news_rss(args.ticker, limit=15)

        print("[CLI TRAIN] Building stationary causal feature matrix...")
        pipe = FeaturePipeline(horizon_days=args.horizon, volatility_factor=0.5)
        X, y_df = pipe.build_feature_table(
            stock_df=stock_df,
            benchmark_df=context["benchmark_df"],
            vix_df=context["vix_df"],
            scored_headlines=news,
            fundamentals=fundamentals,
        )

        valid_mask = ~y_df["target_class"].isna()
        X_clean = X.loc[valid_mask].copy()
        y_clean = y_df.loc[valid_mask, "target_class"].astype(int)

        cv = PurgedWalkForwardCV(n_splits=args.splits, horizon_days=args.horizon)
        splits = cv.split(X_clean)

        print(f"[CLI TRAIN] Fitting StackingEnsemble across {args.splits} purged walk-forward splits...")
        ensemble = StackingEnsemble().fit_walk_forward(X_clean, y_clean, splits)
        artifact_path = ensemble.save_model_bundle(f"ensemble_{args.ticker.replace('.', '_')}")
        print(f"[CLI TRAIN] Successfully saved model bundle to {artifact_path}")
        print(f"[CLI TRAIN] Model metadata: {ensemble.metadata['metrics']}")

    elif args.command == "eval":
        from backend.eval.evaluate import run_comprehensive_evaluation
        print(f"[CLI EVAL] Running comprehensive evaluation for {args.ticker}...")
        res = run_comprehensive_evaluation(
            ticker=args.ticker,
            n_splits=args.splits,
            horizon_days=args.horizon,
            output_report_path=args.output,
        )
        print(f"[CLI EVAL] Finished! Report generated at: {args.output}")
        print(f"[CLI EVAL] Key metrics: {res['metrics']}")

    elif args.command == "test":
        import pytest
        print("[CLI TEST] Running strict anti-leakage and stationarity test suite...")
        exit_code = pytest.main(["backend/tests", "-v", "--durations=5"])
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
