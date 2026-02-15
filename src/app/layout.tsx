// src/layout.tsx
import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title:       "DeepFake Defense",
  description: "Vaccinate your images against deepfake manipulation",
  authors:     [{ name: "DeepFake Defense Team" }],
};

export const viewport: Viewport = {
  width:        "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="antialiased font-sans">{children}</body>
    </html>
  );
}
