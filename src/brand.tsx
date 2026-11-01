import React from 'react';
import logo from './assets/morpheus-gradient-logo.png';
import cutout from './assets/morpheus-gradient-logo-transparent.png';

/** Preserve the approved artwork's colours while using the cutout's alpha. */
export function Brand({ large = false, inverse = false }: { large?: boolean; inverse?: boolean }) {
  const width = large ? 280 : 164, height = large ? 56 : 40;
  const style: React.CSSProperties = {
    display: 'block', objectFit: 'cover', flexShrink: 0,
    maskImage: `url("${cutout}")`, maskSize: 'cover', maskPosition: 'center', maskRepeat: 'no-repeat',
    WebkitMaskImage: `url("${cutout}")`, WebkitMaskSize: 'cover', WebkitMaskPosition: 'center', WebkitMaskRepeat: 'no-repeat',
  };
  const image = <img src={logo} alt="Morpheus" width={width} height={height} draggable={false} style={style} />;
  if (!inverse) return image;
  return <span style={{ position: 'relative', display: 'block', width, height, flexShrink: 0 }}>
    {image}
    <img src={logo} alt="" aria-hidden="true" width={width} height={height} draggable={false} style={{ ...style, position: 'absolute', inset: 0, filter: 'brightness(0) invert(1)', clipPath: 'inset(0 0 0 36%)' }} />
  </span>;
}
