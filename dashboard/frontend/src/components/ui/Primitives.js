import React, { useEffect, useRef, useState } from 'react';

/**
 * Counts from the previous value to the next one so a changed prediction reads
 * as a movement rather than a swap. Honours prefers-reduced-motion.
 */
export const AnimatedNumber = ({ value, duration = 700, className = '' }) => {
  const [display, setDisplay] = useState(value);
  const fromRef = useRef(value);
  const frameRef = useRef();

  useEffect(() => {
    const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    const from = fromRef.current;
    const to = value;
    fromRef.current = value;

    if (reduce || from === to) {
      setDisplay(to);
      return undefined;
    }

    const start = performance.now();
    const tick = (now) => {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      setDisplay(from + (to - from) * eased);
      if (t < 1) frameRef.current = requestAnimationFrame(tick);
    };
    frameRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frameRef.current);
  }, [value, duration]);

  return <span className={`stat-num ${className}`}>{Math.round(display)}</span>;
};

/** A labelled figure with an optional reference frame underneath. */
export const StatTile = ({ label, value, hint, tone = 'default', className = '' }) => {
  const tones = {
    default: 'text-white',
    accent: 'text-accent',
    oppo: 'text-oppo',
    gold: 'text-cricket-gold',
    muted: 'text-dark-muted',
  };
  return (
    <div className={`surface-inset px-3 py-2 ${className}`}>
      <div className="eyebrow mb-0.5">{label}</div>
      <div className={`stat-num text-lg font-semibold ${tones[tone] || tones.default}`}>{value}</div>
      {hint && <div className="mt-0.5 text-[11px] text-dark-muted">{hint}</div>}
    </div>
  );
};

/** Signed delta badge — green for gain, red for loss, neutral at zero. */
export const DeltaBadge = ({ value, unit = 'runs', size = 'md' }) => {
  const rounded = Math.round(value);
  const tone =
    rounded > 0
      ? 'bg-accent/12 text-accent border-accent/30'
      : rounded < 0
        ? 'bg-cricket-red/12 text-cricket-red border-cricket-red/30'
        : 'bg-ink-800 text-dark-muted border-dark-border';
  const sizing = size === 'sm' ? 'px-1.5 py-0.5 text-[11px]' : 'px-2.5 py-1 text-sm';
  return (
    <span className={`stat-num inline-flex items-center rounded-md border font-semibold ${tone} ${sizing}`}>
      {rounded > 0 ? '+' : ''}{rounded}{unit ? ` ${unit}` : ''}
    </span>
  );
};

/** Horizontal 0–10 meter with a labelled scale. */
export const Meter = ({ value, max = 10, tone = 'accent' }) => {
  const pct = Math.max(0, Math.min(100, (value / max) * 100));
  const bar = tone === 'oppo' ? 'bg-oppo' : tone === 'gold' ? 'bg-cricket-gold' : 'bg-accent';
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-ink-800">
      <div
        className={`h-full rounded-full ${bar} transition-[width] duration-500`}
        style={{ width: `${pct}%` }}
      />
    </div>
  );
};

export const ROLE_STYLES = {
  Batsman: { badge: 'bg-accent/12 text-accent border-accent/25', dot: 'bg-accent', short: 'BAT' },
  Bowler: { badge: 'bg-oppo/12 text-oppo border-oppo/25', dot: 'bg-oppo', short: 'BWL' },
  'All-rounder': { badge: 'bg-cricket-gold/12 text-cricket-gold border-cricket-gold/25', dot: 'bg-cricket-gold', short: 'AR' },
};

export const roleStyle = (role) =>
  ROLE_STYLES[role] || { badge: 'bg-ink-800 text-dark-muted border-dark-border', dot: 'bg-ink-500', short: '—' };

const Primitives = { AnimatedNumber, StatTile, DeltaBadge, Meter, roleStyle };

export default Primitives;
