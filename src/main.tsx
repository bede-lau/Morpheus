import React from 'react';
import { createRoot } from 'react-dom/client';

// A random browser-local bearer ID keeps hosted saves separate without an account.
const visitorKey = 'morpheus-visitor-v1';
const randomVisitor = () => Array.from(crypto.getRandomValues(new Uint8Array(32)), n => n.toString(16).padStart(2, '0')).join('');
const localVisitor = () => {
  try {
    const stored = localStorage.getItem(visitorKey);
    const visitor = stored && /^[0-9a-f]{64}$/.test(stored) ? stored : randomVisitor();
    localStorage.setItem(visitorKey, visitor);
    return visitor;
  } catch { return randomVisitor(); }
};
const siteOrigin = 'https://morpheus-robot-simulation.bedelau59.chatgpt.site';
const visitor = window.parent === window ? Promise.resolve(localVisitor()) : new Promise<string>(resolve => {
  let settled = false;
  const finish = (value: string) => {
    if (settled) return;
    settled = true;
    window.removeEventListener('message', receive);
    resolve(value);
  };
  const receive = (event: MessageEvent) => {
    if (event.origin === siteOrigin && event.source === window.parent && event.data?.type === 'morpheus-visitor' && /^[0-9a-f]{64}$/.test(event.data?.visitor)) finish(event.data.visitor);
  };
  window.addEventListener('message', receive);
  window.parent.postMessage({type: 'morpheus-visitor-request'}, siteOrigin);
  setTimeout(() => finish(localVisitor()), 1500);
});
const nativeFetch = globalThis.fetch.bind(globalThis);
globalThis.fetch = async (input, init) => {
  const url = new URL(input instanceof Request ? input.url : String(input), location.href);
  if (url.origin !== location.origin || !url.pathname.startsWith('/api/')) return nativeFetch(input, init);
  const headers = new Headers(input instanceof Request ? input.headers : undefined);
  new Headers(init?.headers).forEach((value, key) => headers.set(key, value));
  headers.set('X-Morpheus-Visitor', await visitor);
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
