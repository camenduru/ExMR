import { defineConfig } from 'vite';

export default defineConfig({
  base: '/',
  build: {
    outDir: '../dist',
    emptyOutDir: true,
    target: 'es2020',
  },
  server: {
    port: 5174,
    proxy: {
      '/api': 'http://127.0.0.1:8010',
      '/videos': 'http://127.0.0.1:8010',
      '/vrm': 'http://127.0.0.1:8010',
      '/exports': 'http://127.0.0.1:8010',
    },
  },
});
