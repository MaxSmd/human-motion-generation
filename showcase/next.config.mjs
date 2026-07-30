/** @type {import('next').NextConfig} */
const nextConfig = {
  // Fully static site: `next build` emits ./out with no Node server at runtime.
  output: "export",
  images: { unoptimized: true },
  trailingSlash: true,
};

export default nextConfig;
