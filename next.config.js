/** @type {import('next').NextConfig} */
const nextConfig = {
  // MANDATORY for Tauri: output static files that Tauri can serve from disk
  output: "export",

  // MANDATORY: Next.js Image Optimization requires a server, disable it
  images: {
    unoptimized: true,
  },

  // No trailing slashes (Tauri serves index.html directly)
  trailingSlash: false,

  // Disable server-side features
  reactStrictMode: true,

};

module.exports = nextConfig;
