import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        institutional: {
          bg: "#090A0F",
          card: "#12141C",
          "card-hover": "#171A24",
          border: "#1E2230",
          "border-light": "#2A3042",
          emerald: "#10B981",
          "emerald-glow": "rgba(16, 185, 129, 0.15)",
          crimson: "#F43F5E",
          "crimson-glow": "rgba(244, 63, 94, 0.15)",
          gold: "#F59E0B",
          cyan: "#06B6D4",
          text: "#E2E8F0",
          muted: "#94A3B8",
          dim: "#64748B",
        },
      },
      fontFamily: {
        mono: [
          "JetBrains Mono",
          "SFMono-Regular",
          "Menlo",
          "Monaco",
          "Consolas",
          "Liberation Mono",
          "Courier New",
          "monospace",
        ],
      },
    },
  },
  plugins: [],
};

export default config;
