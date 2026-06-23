import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: Number(process.env.PORT) || 5173,
    proxy: {
      "/analyze": "http://localhost:8000",
      "/explain": "http://localhost:8000",
      "/bias": "http://localhost:8000",
      "/evidence": "http://localhost:8000",
    },
  },
});
