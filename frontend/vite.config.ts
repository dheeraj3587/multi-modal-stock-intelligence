import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
    plugins: [react()],
    server: {
        proxy: {
            '/upstox': {
                target: 'https://api.upstox.com',
                changeOrigin: true,
                rewrite: (path) => path.replace(/^\/upstox/, ''),
            },
        },
    },
})
