import React, { useMemo, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ArrowRightLeft, RotateCcw } from 'lucide-react';
import { sortByRelevance, isEstablished, squadComposition } from '../utils/squad';

/**
 * Swap a player in the batting XI and compare squad strength before/after.
 * Runs entirely on the loaded player data — instant, no round-trip.
 */

const METRICS = [
  { key: 'avgBatting', label: 'Team batting average', max: 45, fmt: (v) => v.toFixed(1) },
  { key: 'elite', label: 'Elite batsmen (avg 40+)', max: 11, fmt: (v) => String(v) },
  { key: 'depth', label: 'Batting depth (avg 30+)', max: 11, fmt: (v) => String(v) },
];

const ImpactLab = ({ teamA, players }) => {
  const [outPlayer, setOutPlayer] = useState('');
  const [inPlayer, setInPlayer] = useState('');

  const lookup = useMemo(() => new Map(players.map((p) => [p.player_id, p])), [players]);

  const candidates = useMemo(() => {
    const inSquad = new Set(teamA.players.map((p) => p.id));
    return sortByRelevance(
      players.filter((p) => !inSquad.has(p.player_id) && isEstablished(p))
    ).slice(0, 150);
  }, [players, teamA.players]);

  const base = useMemo(() => squadComposition(teamA.players, players), [teamA.players, players]);

  const swapped = useMemo(() => {
    if (!outPlayer || !inPlayer) return null;
    const meta = lookup.get(inPlayer);
    if (!meta) return null;
    const next = teamA.players.map((p) =>
      p.id === outPlayer ? { id: meta.player_id, name: meta.player_name, country: meta.country } : p
    );
    return squadComposition(next, players);
  }, [outPlayer, inPlayer, teamA.players, players, lookup]);

  const ready = teamA.players.length > 0;
  const outName = teamA.players.find((p) => p.id === outPlayer)?.name;
  const inName = lookup.get(inPlayer)?.player_name;

  const reset = () => { setOutPlayer(''); setInPlayer(''); };

  return (
    <section className="surface p-4">
      <div className="mb-1 flex items-center gap-2">
        <ArrowRightLeft className="h-4 w-4 text-accent" />
        <h2 className="section-title">Player swap explorer</h2>
      </div>
      <p className="mb-4 text-[13px] text-dark-muted">
        Swap a player into the batting XI to see how the squad profile changes.
      </p>

      <div className="grid gap-4 md:grid-cols-[1fr_auto_1fr]">
        <div>
          <label className="label" htmlFor="impact-out">Take out</label>
          <select
            id="impact-out"
            className="cricket-select"
            value={outPlayer}
            onChange={(e) => setOutPlayer(e.target.value)}
            disabled={!ready}
          >
            <option value="">Select a player…</option>
            {teamA.players.map((p) => {
              const m = lookup.get(p.id);
              return (
                <option key={p.id} value={p.id}>
                  {p.name}{m?.batting_avg > 0 ? ` · ${m.batting_avg.toFixed(1)} avg` : ''}
                </option>
              );
            })}
          </select>
        </div>

        <div className="hidden items-end pb-3 text-dark-muted md:flex" aria-hidden>
          <ArrowRightLeft className="h-4 w-4" />
        </div>

        <div>
          <label className="label" htmlFor="impact-in">Bring in</label>
          <select
            id="impact-in"
            className="cricket-select"
            value={inPlayer}
            onChange={(e) => setInPlayer(e.target.value)}
            disabled={!ready}
          >
            <option value="">Select a player…</option>
            {candidates.map((p) => (
              <option key={p.player_id} value={p.player_id}>
                {p.player_name} · {p.country}
                {p.batting_avg > 0 ? ` · ${p.batting_avg.toFixed(1)} avg` : ''}
              </option>
            ))}
          </select>
        </div>
      </div>

      {!ready && (
        <p className="mt-4 text-xs text-dark-muted">Pick a batting XI to compare squads.</p>
      )}

      <AnimatePresence>
        {swapped && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="mt-4 surface-inset p-4"
          >
            <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
              <div className="flex items-center gap-3 text-sm">
                <span className="text-dark-muted line-through">{outName}</span>
                <ArrowRightLeft className="h-3.5 w-3.5 text-dark-muted" />
                <span className="font-semibold text-white">{inName}</span>
              </div>
              <button onClick={reset} className="btn-ghost !py-1.5 !px-3 text-xs">
                <RotateCcw className="h-3 w-3" />
                Reset
              </button>
            </div>

            <div className="space-y-4">
              {METRICS.map((m) => {
                const before = base[m.key];
                const after = swapped[m.key];
                const delta = after - before;
                const pct = (v) => Math.max(2, Math.min(100, (v / m.max) * 100));
                const up = delta > 0.05;
                const down = delta < -0.05;
                return (
                  <div key={m.key}>
                    <div className="mb-2 flex items-baseline justify-between text-xs">
                      <span className="text-dark-muted">{m.label}</span>
                      <span className="flex items-baseline gap-2">
                        <span className="stat-num text-dark-muted">{m.fmt(before)}</span>
                        <span className="text-dark-muted">→</span>
                        <span className="stat-num font-semibold text-white">{m.fmt(after)}</span>
                        {(up || down) && (
                          <span
                            className={`stat-num text-[11px] font-semibold ${
                              up ? 'text-accent' : 'text-cricket-red'
                            }`}
                          >
                            {up ? '▲' : '▼'} {Math.abs(delta).toFixed(m.key === 'avgBatting' ? 1 : 0)}
                          </span>
                        )}
                      </span>
                    </div>

                    {/* Baseline track with the new value overlaid. */}
                    <div className="relative h-2 overflow-hidden rounded-full bg-ink-800">
                      <div
                        className="absolute inset-y-0 left-0 rounded-full bg-ink-600"
                        style={{ width: `${pct(before)}%` }}
                      />
                      <motion.div
                        className={`absolute inset-y-0 left-0 rounded-full ${
                          down ? 'bg-cricket-red' : 'bg-accent'
                        }`}
                        initial={{ width: `${pct(before)}%` }}
                        animate={{ width: `${pct(after)}%` }}
                        transition={{ duration: 0.55, ease: [0.16, 1, 0.3, 1] }}
                        style={{ opacity: 0.9 }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>

            <p className="mt-4 text-[13px] leading-relaxed text-dark-muted">
              {swapped.avgBatting - base.avgBatting > 0.05 ? (
                <>
                  <span className="text-dark-text">{inName}</span> strengthens the top order, lifting
                  the team average to{' '}
                  <span className="stat-num text-dark-text">{swapped.avgBatting.toFixed(1)}</span>.
                </>
              ) : swapped.avgBatting - base.avgBatting < -0.05 ? (
                <>
                  This swap trades batting for balance. Team average drops to{' '}
                  <span className="stat-num text-dark-text">{swapped.avgBatting.toFixed(1)}</span>.
                </>
              ) : (
                <>These two are an even trade on batting strength.</>
              )}
            </p>
          </motion.div>
        )}
      </AnimatePresence>
    </section>
  );
};

export default ImpactLab;
