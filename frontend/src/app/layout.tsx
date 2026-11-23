import type { Metadata } from "next";
import { Inter, Literata } from "next/font/google";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin", "cyrillic", "cyrillic-ext"],
  display: "swap",
});

const literata = Literata({
  variable: "--font-literata",
  subsets: ["latin", "cyrillic", "cyrillic-ext"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "TMS — Therapist Machine Support",
  description:
    "AI-powered diagnostic assistant. Describe your symptoms and get preliminary diagnosis with doctor recommendations.",
  icons: { icon: "/favicon.ico" },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ru">
      <body
        className={`${inter.variable} ${literata.variable} font-sans antialiased bg-background text-foreground`}
      >
        {children}
      </body>
    </html>
  );
}
