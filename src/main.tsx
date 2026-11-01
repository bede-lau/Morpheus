import React from 'react';
import { createRoot } from 'react-dom/client';

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
