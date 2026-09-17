import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  allowedDevOrigins: ["172.20.10.2", "localhost:3000", "*.local"],
};

export default nextConfig;
