import React from 'react';
import { createRoot } from 'react-dom/client';

// A random browser-local bearer ID keeps hosted saves separate without an account.
const visitorKey = 'morpheus-visitor-v1';
const randomVisitor = () => Array.from(crypto.getRandomValues(new Uint8Array(32)), n => n.toString(16).padStart(2, '0')).join('');
let visitor: string;
try {
  const stored = localStorage.getItem(visitorKey);
  visitor = stored && /^[0-9a-f]{64}$/.test(stored) ? stored : randomVisitor();
  localStorage.setItem(visitorKey, visitor);
} catch {
  // Storage may be blocked in private browsing; this session remains isolated.
  visitor = randomVisitor();
}
const nativeFetch = globalThis.fetch.bind(globalThis);
globalThis.fetch = (input, init) => {
  const url = new URL(input instanceof Request ? input.url : String(input), location.href);
  if (url.origin !== location.origin || !url.pathname.startsWith('/api/')) return nativeFetch(input, init);
  const headers = new Headers(input instanceof Request ? input.headers : undefined);
  new Headers(init?.headers).forEach((value, key) => headers.set(key, value));
  headers.set('X-Morpheus-Visitor', visitor);
  return nativeFetch(input, { ...init, headers });
};

const view=new URLSearchParams(location.search).get('view');
// `/` (and the `?view=demo` alias) is the showpiece demo; the full workspace lives at `?view=workspace`.
if (view === 'objects') {
  void import('./object-workspace').then(({ default: Objects }) => {
    createRoot(document.getElementById('root')!).render(<Objects/>);
  });
} else if (view === 'builder') {
  void import('./builder');
} else if (view === 'workspace') {
  void import('./workspace').then(({ default: Workspace }) => {
    createRoot(document.getElementById('root')!).render(<Workspace/>);
  });
} else {
  void import('./demo').then(({ default: Demo }) => {
    createRoot(document.getElementById('root')!).render(<Demo/>);
  });
}
