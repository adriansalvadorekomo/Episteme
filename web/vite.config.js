import { defineConfig } from 'vite';

// Standalone Vite config — used by Vite 6 directly alongside astro.config.mjs.
// allowedHosts: true disables the host-header check that blocks Cloudflare tunnels.
export default defineConfig({
  server: {
    allowedHosts: true,
    host: true,
  },
});
