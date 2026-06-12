/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Standalone output keeps the production Docker image small (copies only the
  // traced runtime + node_modules needed to serve).
  output: "standalone",
};

export default nextConfig;
