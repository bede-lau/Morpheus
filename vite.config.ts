import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { cp, access } from 'node:fs/promises';
// Include generated policy evidence while keeping unlicensed scan binaries out
// of production bundles. Absent artifacts are an explicit UI empty state.
export default defineConfig({plugins:[react(),{
  name:'generated-policy-artifacts',
  async closeBundle(){
    try { await access('public/policies'); } catch { return; }
    await cp('public/policies','dist/policies',{recursive:true});
  },
}],server:{port:5174,proxy:{'/api':'http://127.0.0.1:8000'}},build:{chunkSizeWarningLimit:1200,copyPublicDir:false}});
