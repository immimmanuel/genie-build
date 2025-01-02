import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    preserveSymlinks: true,
  },
  build: {
    outDir: "./dist",
    emptyOutDir: true,
    sourcemap: true,
    target: "esnext",
    rollupOptions: {
      output: {
        manualChunks: id => {
          if (id.includes("@fluentui/react-icons")) {
            return "fluentui-icons";
          } else if (id.includes("@fluentui/react")) {
            return "fluentui-react";
          } else if (id.includes("node_modules")) {
            return "vendor";
          }
        },
      },
    },
  },
  server: {
    proxy: {
      "/chat": "http://localhost:8000",
    },
    host: "127.0.0.1",
  },
  base: "./",
});
