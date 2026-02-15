// src/layout.tsx
import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title:       "Vaxel",
  description: "Vaxel protects images against deepfake manipulation",
  authors:     [{ name: "Vaxel Team" }],
};

export const viewport: Viewport = {
  width:        "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" data-theme="dark">
      <body className="antialiased font-sans">{children}</body>
    </html>
  );
}
