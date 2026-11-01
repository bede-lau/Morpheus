import React from 'react';
import { createRoot } from 'react-dom/client';

// Keep the existing scan preparation workflow available to the backend team.
if (new URLSearchParams(location.search).get('view') === 'objects') {
  void import('./object-workspace').then(({ default: Objects }) => {
    createRoot(document.getElementById('root')!).render(<Objects/>);
  });
} else if (new URLSearchParams(location.search).get('view') === 'builder') {
  void import('./builder');
} else {
  void import('./workspace').then(({ default: Workspace }) => {
    createRoot(document.getElementById('root')!).render(<Workspace/>);
  });
}
