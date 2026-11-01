import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
export default defineConfig({plugins:[react()],server:{port:5174,proxy:{'/api/objects':'http://127.0.0.1:8001','/api':'http://127.0.0.1:8000'}},build:{chunkSizeWarningLimit:1200,copyPublicDir:false}});
