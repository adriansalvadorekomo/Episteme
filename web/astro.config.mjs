import { defineConfig } from 'astro/config';

// Démo locale. Le frontend parle à l'API FastAPI (http://localhost:8000).
// En dev, on proxifie /api -> backend pour éviter les soucis de CORS.
export default defineConfig({
  server: { port: 4321, host: true },
  vite: {
    // Mermaid est importé à la demande (premier diagramme). Sans pré-bundling, Vite le
    // découvre en pleine session, le ré-optimise et invalide l'ancien hash (504 "Outdated
    // Optimize Dep") : on le pré-bundle au démarrage du serveur de dev.
    optimizeDeps: { include: ['mermaid', 'gsap', 'gsap/Flip', 'gsap/MotionPathPlugin', 'marked', 'dompurify'] },
    server: {
      // Allow Cloudflare Quick Tunnel domains (*.trycloudflare.com) and any
      // other external host — safe for local dev/demo, not for production.
      allowedHosts: 'all',
      host: '0.0.0.0',
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
