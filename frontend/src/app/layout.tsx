import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AARC | Autonomous Adversarial Risk Committee Terminal",
  description: "Institutional Quantitative Trading & Adversarial Multi-Agent Risk Terminal",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="bg-[#090A0F] text-[#E2E8F0] min-h-screen antialiased selection:bg-[#10B981]/20 selection:text-[#10B981]">
        {children}
      </body>
    </html>
  );
}
