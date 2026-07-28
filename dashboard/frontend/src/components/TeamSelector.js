import React, { useMemo, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Plus, X, Search, Wand2 } from 'lucide-react';
import { roleStyle } from './ui/Primitives';
import { sortByRelevance, isEstablished, buildBalancedXI, squadComposition } from '../utils/squad';

const ROLES = ['All', 'Batsman', 'All-rounder', 'Bowler'];

const TeamSelector = ({
  teamType,
  team,
  teams,
  players,
  whatIfAllPlayers = false,
  onTeamSelect,
  onPlayerSelect,
  onRemovePlayer,
  onAutoFill,
}) => {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [role, setRole] = useState('All');
  const [showFringe, setShowFringe] = useState(false);

  const isBatting = teamType === 'A';
  const accentText = isBatting ? 'text-accent' : 'text-oppo';
  const accentBg = isBatting ? 'bg-accent' : 'bg-oppo';
  const accentSoft = isBatting ? 'bg-accent/10 border-accent/25' : 'bg-oppo/10 border-oppo/25';

  const comp = useMemo(() => squadComposition(team.players, players), [team.players, players]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    const chosen = new Set(team.players.map((p) => p.id));

    let pool = players.filter((p) => {
      if (chosen.has(p.player_id)) return false;
      if (role !== 'All' && (p.player_role || 'All-rounder') !== role) return false;
      if (team.team_name && !whatIfAllPlayers) {
        if ((p.country || '').toLowerCase() !== team.team_name.toLowerCase()) return false;
      }
      if (!q) return true;
      return (
        (p.player_name || '').toLowerCase().includes(q) ||
        (p.country || '').toLowerCase().includes(q)
      );
    });

    // Hide one-cap players by default — they're what made the list look broken.
    if (!showFringe && !q) {
      const established = pool.filter(isEstablished);
      if (established.length >= 12) pool = established;
    }

    return sortByRelevance(pool);
  }, [players, team.players, team.team_name, query, role, whatIfAllPlayers, showFringe]);

  const handleTeamChange = (e) => {
    const id = parseInt(e.target.value, 10);
    const t = teams.find((x) => x.team_id === id);
    if (t) onTeamSelect(teamType, id, t.team_name);
  };

  const autoFill = () => {
    const xi = buildBalancedXI(players, team.team_name);
    if (xi.length) onAutoFill(teamType, xi);
  };

  const full = team.players.length === 11;

  return (
    <motion.section
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="surface flex flex-col p-4"
    >
      {/* Header */}
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <span className={`h-2 w-2 rounded-full ${accentBg}`} />
          <h2 className="section-title">{isBatting ? 'Batting XI' : 'Bowling XI'}</h2>
        </div>
        <span className={`stat-num text-xs ${full ? accentText : 'text-dark-muted'}`}>
          {team.players.length}/11
        </span>
      </div>

      {/* Country */}
      <select
        value={team.team_id || ''}
        onChange={handleTeamChange}
        className="cricket-select mb-2.5"
        aria-label={`${isBatting ? 'Batting' : 'Bowling'} team country`}
      >
        <option value="">Choose a country…</option>
        {teams.map((t) => (
          <option key={t.team_id} value={t.team_id}>{t.team_name}</option>
        ))}
      </select>

      {!team.team_name && (
        <p className="px-1 py-6 text-center text-xs leading-relaxed text-dark-muted">
          Choose a country to pick this {isBatting ? 'batting' : 'bowling'} XI,
          <br />
          or turn on Dream-XI mode to select from anyone.
        </p>
      )}

      {team.team_name && (
        <>
          {/* Composition summary — replaces the old "Bowlers: 0" warning card. */}
          <div className={`mb-4 flex flex-wrap items-center gap-x-4 gap-y-1.5 rounded-lg border px-3 py-2 ${accentSoft}`}>
            <span className="text-sm font-semibold text-white">{team.team_name}</span>
            <div className="flex items-center gap-3 text-[11px] text-dark-muted">
              <span><b className="stat-num text-dark-text">{comp.batsmen}</b> bat</span>
              <span><b className="stat-num text-dark-text">{comp.allRounders}</b> all</span>
              <span><b className="stat-num text-dark-text">{comp.bowlers}</b> bowl</span>
            </div>
            {team.players.length > 0 && (
              <span className="ml-auto text-[11px] text-dark-muted">
                {isBatting
                  ? <>avg <b className="stat-num text-dark-text">{comp.avgBatting.toFixed(1)}</b></>
                  : <>econ <b className="stat-num text-dark-text">{comp.avgEconomy.toFixed(2)}</b></>}
              </span>
            )}
          </div>

          {/* Actions */}
          <div className="mb-2.5 flex items-center gap-2">
            <button onClick={autoFill} className="btn-ghost !py-1.5 !px-3 text-xs">
              <Wand2 className="h-3.5 w-3.5" />
              Auto-pick XI
            </button>
            {!full && (
              <button
                onClick={() => setOpen((v) => !v)}
                className="btn-ghost !py-1.5 !px-3 text-xs"
                aria-expanded={open}
              >
                <Plus className="h-3.5 w-3.5" />
                Add player
              </button>
            )}
          </div>

          {/* Picker */}
          <AnimatePresence initial={false}>
            {open && !full && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="mb-4 overflow-hidden"
              >
                <div className="surface-inset p-3">
                  <div className="relative mb-2.5">
                    <Search className="pointer-events-none absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-dark-muted" />
                    <input
                      type="text"
                      placeholder="Search players…"
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      className="cricket-input !pl-9"
                    />
                  </div>

                  <div className="mb-2.5 flex flex-wrap gap-1.5">
                    {ROLES.map((r) => (
                      <button
                        key={r}
                        onClick={() => setRole(r)}
                        className={`rounded-md px-2.5 py-1 text-[11px] font-medium transition-colors ${
                          role === r
                            ? `${accentBg} text-ink-950`
                            : 'bg-ink-800 text-dark-muted hover:text-dark-text'
                        }`}
                      >
                        {r}
                      </button>
                    ))}
                  </div>

                  <div className="max-h-72 overflow-y-auto rounded-lg border border-dark-border">
                    {filtered.slice(0, 60).map((p) => {
                      const rs = roleStyle(p.player_role);
                      return (
                        <button
                          key={p.player_id}
                          onClick={() => {
                            onPlayerSelect(teamType, p.player_id, p.player_name, p.country);
                            setQuery('');
                          }}
                          className="flex w-full items-center gap-3 border-b border-dark-border/70 px-3 py-2.5 text-left transition-colors last:border-0 hover:bg-ink-800"
                        >
                          <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${rs.dot}`} />
                          <span className="min-w-0 flex-1">
                            <span className="block truncate text-sm font-medium text-dark-text">
                              {p.player_name}
                            </span>
                            <span className="mt-0.5 block truncate text-[11px] text-dark-muted">
                              {p.country}
                              {p.batting_avg > 0 && <> · <span className="stat-num">{p.batting_avg.toFixed(1)}</span> avg</>}
                              {p.bowling_economy > 0 && <> · <span className="stat-num">{p.bowling_economy.toFixed(1)}</span> econ</>}
                              {p.total_matches > 0 && <> · <span className="stat-num">{p.total_matches}</span> mat</>}
                            </span>
                          </span>
                          <span className={`shrink-0 rounded border px-1.5 py-0.5 text-[10px] font-semibold ${rs.badge}`}>
                            {rs.short}
                          </span>
                        </button>
                      );
                    })}
                    {filtered.length === 0 && (
                      <p className="px-3 py-6 text-center text-xs text-dark-muted">
                        No players match. Try a different filter.
                      </p>
                    )}
                  </div>

                  <label className="mt-2.5 flex cursor-pointer items-center gap-2 text-[11px] text-dark-muted">
                    <input
                      type="checkbox"
                      checked={showFringe}
                      onChange={(e) => setShowFringe(e.target.checked)}
                      className="h-3 w-3 rounded border-ink-500 bg-ink-950"
                    />
                    Include players with fewer than 20 caps
                  </label>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Selected XI */}
          <ol className="space-y-1">
            <AnimatePresence initial={false}>
              {team.players.map((p, i) => {
                const meta = players.find((x) => x.player_id === p.id);
                const rs = roleStyle(meta?.player_role);
                return (
                  <motion.li
                    key={p.id}
                    layout
                    initial={{ opacity: 0, x: -8 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 8 }}
                    transition={{ duration: 0.18 }}
                    className="group flex items-center gap-2.5 rounded-md px-2 py-1 hover:bg-ink-850"
                  >
                    <span className="stat-num w-4 shrink-0 text-right text-[11px] text-dark-muted">
                      {i + 1}
                    </span>
                    <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${rs.dot}`} />
                    <span className="min-w-0 flex-1 truncate text-sm text-dark-text">{p.name}</span>
                    {meta?.batting_avg > 0 && (
                      <span className="stat-num shrink-0 text-[11px] text-dark-muted">
                        {meta.batting_avg.toFixed(1)}
                      </span>
                    )}
                    <button
                      onClick={() => onRemovePlayer(teamType, p.id)}
                      aria-label={`Remove ${p.name}`}
                      className="shrink-0 text-dark-muted opacity-0 transition-opacity hover:text-cricket-red focus:opacity-100 group-hover:opacity-100"
                    >
                      <X className="h-3.5 w-3.5" />
                    </button>
                  </motion.li>
                );
              })}
            </AnimatePresence>
          </ol>

          {team.players.length === 0 && (
            <p className="py-6 text-center text-xs text-dark-muted">
              No players yet. Use <b className="text-dark-text">Auto-pick XI</b> to load a realistic squad.
            </p>
          )}
        </>
      )}
    </motion.section>
  );
};

export default TeamSelector;
