# Project Rules: Autonomous Adversarial Risk Committee (AARC)
- Monorepo structure: `/backend` (FastAPI + LangGraph) and `/frontend` (Next.js App Router, TypeScript, Tailwind CSS, Lucide Icons, Recharts).
- Data rules: Flatten `yfinance` MultiIndex columns immediately (`if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)`). Replace all NaN/Inf values with 0.0 before JSON serialization.
- LLM: Use `langchain-google-genai` with model `gemini-2.5-flash`.
- UI Theme: Institutional Dark Mode (`#090A0F` background, `#12141C` cards, `1px solid #1E2230` borders, Emerald `#10B981` for Bull, Crimson `#F43F5E` for Bear/Anomalies, Monospace font for all numbers).
