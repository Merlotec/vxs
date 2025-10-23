import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import wasm from 'vite-plugin-wasm'
import topLevelAwait from 'vite-plugin-top-level-await'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    react(),
    wasm(),
    topLevelAwait()
  ],
  resolve: {
    alias: {
      '@': '/src'
    }
  },
  server: {
    port: 3000,
    proxy: {
      '/ws': {
        target: 'ws://localhost:9080',
        ws: true
      }
    }
  },
  build: {
    target: 'esnext',
    assetsInlineLimit: 0 // Don't inline WASM files
  },
  optimizeDeps: {
    exclude: ['voxelsim-renderer-wasm']
  }
})
