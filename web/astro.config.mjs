import { defineConfig } from 'astro/config';

// Démo locale. Le frontend parle à l'API FastAPI (http://localhost:8000).
// En dev, on proxifie /api -> backend pour éviter les soucis de CORS.
export default defineConfig({
  server: { port: 4321, host: true, allowedHosts: true },
  vite: {
    // Mermaid est importé à la demande (premier diagramme). Sans pré-bundling, Vite le
    // découvre en pleine session, le ré-optimise et invalide l'ancien hash (504 "Outdated
    // Optimize Dep") : on le pré-bundle au démarrage du serveur de dev.
    optimizeDeps: { include: ['mermaid', 'gsap', 'gsap/Flip', 'gsap/MotionPathPlugin', 'marked', 'dompurify'] },
    server: {
      // Allow any external host (Cloudflare tunnel, etc.) — Vite 6 requires
      // boolean true, not the string 'all'. Safe for local dev/demo only.
      allowedHosts: true,
      host: true,
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
