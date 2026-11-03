import { defineConfig } from 'astro/config';

// Démo locale. Le frontend parle à l'API FastAPI (http://localhost:8000).
// En dev, on proxifie /api -> backend pour éviter les soucis de CORS.
export default defineConfig({
  server: { port: 4321 },
  vite: {
    server: {
      proxy: {
        '/api': {
          target: 'http://localhost:8000',
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api/, ''),
        },
      },
    },
  },
});
