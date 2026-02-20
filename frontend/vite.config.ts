import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    proxy: {
      // /ui/* → FastAPI (localhost:8000)
      "/ui": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
