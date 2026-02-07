import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
    const env = loadEnv(mode, process.cwd(), '')
    // In local dev, you can point VITE_API_URL at a remote backend.
    // Otherwise we proxy /api -> backend during `vite dev`.
    const apiTarget = env.VITE_API_URL || 'http://localhost:8000'

    return {
        plugins: [react()],
        resolve: {
            alias: {
                '@': path.resolve(__dirname, './src'),
            },
        },
        server: {
            proxy: {
                '/api': {
                    target: apiTarget,
                    changeOrigin: true,
                    rewrite: (path) => path.replace(/^\/api/, ''),
                },
                '/upstox': {
                    target: 'https://api.upstox.com',
                    changeOrigin: true,
                    rewrite: (path) => path.replace(/^\/upstox/, ''),
                },
            },
        },
    }
})
