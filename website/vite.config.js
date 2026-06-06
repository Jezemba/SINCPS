import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Project page at https://jezemba.github.io/SINCPS
// If you move to a separate repo or a custom domain, change `base` (and the
// 404.html pathSegmentsToKeep) accordingly.
export default defineConfig({
  base: '/SINCPS/',
  plugins: [react(), tailwindcss()],
})
