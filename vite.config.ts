import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// base './' makes the build work from any GitHub Pages subpath.
export default defineConfig({
  plugins: [react()],
  base: './',
  build: {
    chunkSizeWarningLimit: 1500,
  },
})
