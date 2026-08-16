import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const root = resolve(import.meta.dirname, '..');
const [html, config, vercel] = await Promise.all([
  readFile(resolve(root, 'index.html'), 'utf8'),
  readFile(resolve(root, 'config.js'), 'utf8'),
  readFile(resolve(root, 'vercel.json'), 'utf8'),
]);

JSON.parse(vercel);
if (!html.includes('config.js') || !html.includes('chat/stream')) {
  throw new Error('Static app is missing its runtime API configuration or SSE integration.');
}
if (!config.includes('__API_BASE_URL__')) {
  throw new Error('config.js must define window.__API_BASE_URL__.');
}
console.log('Static frontend validation passed.');
