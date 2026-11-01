import React from 'react';
import logo from './assets/morpheus-gradient-logo.png';

/** One approved asset shared by every application header. */
export function Brand({ large = false }: { large?: boolean }) {
  return <img src={logo} alt="Morpheus" width={large ? 280 : 164} height={large ? 56 : 40} draggable={false} style={{ display: 'block', objectFit: 'cover', flexShrink: 0, borderRadius: 3 }} />;
}
