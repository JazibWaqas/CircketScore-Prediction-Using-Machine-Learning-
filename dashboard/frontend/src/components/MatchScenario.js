import React, { useMemo } from 'react';
import { motion } from 'framer-motion';
import { Loader2 } from 'lucide-react';
import { curateVenues } from '../utils/squad';

/**
 * Full-width match-situation bar. Sits above the two team columns so the
 * workbench stays visually balanced whether or not the XIs are populated.
 */
const MatchScenario = ({
  scenario,
  onChange,
  venues,
  battingPlayers,
  onPredict,
  predicting,
  ready,
  teamA,
  teamB,
  whatIfAllPlayers,
  onToggleWhatIf,
}) => {
  const set = (field, value) => onChange({ ...scenario, [field]: value });

  const setNumber = (field, raw, { min, max }) => {
    if (raw === '') return set(field, '');
    const n = parseInt(raw, 10);
    if (Number.isNaN(n)) return undefined;
    return set(field, Math.max(min, Math.min(max, n)));
  };

  const handleVenue = (name) => {
    const v = venues.find((x) => x.venue_name === name);
    onChange({
      ...scenario,
      venue: name,
      venue_avg_score: v && v.avg_score ? v.avg_score : 250,
    });
  };

  const { major, rest } = useMemo(() => curateVenues(venues), [venues]);

  const overs = Number(scenario.overs) || 0;
  const current = Number(scenario.current_score) || 0;
  const rr = overs > 0 ? (current / overs).toFixed(2) : '—';
  const oversLeft = Math.max(0, 50 - overs);

  const missing = [
    teamA.players.length < 11 && `batting XI (${teamA.players.length}/11)`,
    teamB.players.length < 11 && `bowling XI (${teamB.players.length}/11)`,
    !scenario.venue && 'venue',
  ].filter(Boolean);

  return (
    <motion.section
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="surface p-4"
    >
      <div className="mb-3 flex flex-wrap items-baseline gap-x-4 gap-y-1">
        <h2 className="section-title">Match situation</h2>
        <span className="text-[11px] text-dark-muted">
          {overs > 0 && (
            <>
              Run rate <b className="stat-num text-dark-text">{rr}</b>
              <span className="mx-1.5 opacity-40">·</span>
            </>
          )}
          <b className="stat-num text-dark-text">{oversLeft}</b> overs left
        </span>
        <label className="ml-auto flex cursor-pointer items-center gap-2 text-[11px] text-dark-muted">
          <input
            type="checkbox"
            checked={whatIfAllPlayers}
            onChange={(e) => onToggleWhatIf(e.target.checked)}
            className="h-3.5 w-3.5 rounded border-ink-500 bg-ink-950"
          />
          <b className="text-dark-text">Dream-XI mode</b>
          <span className="hidden sm:inline">picks from any country</span>
        </label>
      </div>

      {/* Inputs flow horizontally; the CTA anchors the right edge. */}
      <div className="grid gap-2.5 md:grid-cols-2 lg:grid-cols-[minmax(0,2fr)_repeat(4,minmax(0,1fr))_auto]">
        <div className="lg:col-span-1">
          <label className="label" htmlFor="venue">Venue</label>
          <select
            id="venue"
            value={scenario.venue}
            onChange={(e) => handleVenue(e.target.value)}
            className="cricket-select"
          >
            <option value="">Select venue…</option>
            <optgroup label="Major international grounds">
              {major.map((v) => (
                <option key={v.venue_name} value={v.venue_name}>
                  {v.venue_name}{v.avg_score ? ` · par ${v.avg_score.toFixed(0)}` : ''}
                </option>
              ))}
            </optgroup>
            <optgroup label="All other venues">
              {rest.map((v) => (
                <option key={v.venue_name} value={v.venue_name}>
                  {v.venue_name}{v.avg_score ? ` · par ${v.avg_score.toFixed(0)}` : ''}
                </option>
              ))}
            </optgroup>
          </select>
        </div>

        <div>
          <label className="label" htmlFor="score">Score</label>
          <input
            id="score" type="number" inputMode="numeric"
            value={scenario.current_score}
            onChange={(e) => setNumber('current_score', e.target.value, { min: 0, max: 500 })}
            className="cricket-input"
          />
        </div>
        <div>
          <label className="label" htmlFor="wickets">Wickets</label>
          <input
            id="wickets" type="number" inputMode="numeric"
            value={scenario.wickets_fallen}
            onChange={(e) => setNumber('wickets_fallen', e.target.value, { min: 0, max: 10 })}
            className="cricket-input"
          />
        </div>
        <div>
          <label className="label" htmlFor="overs">Overs</label>
          <input
            id="overs" type="number" inputMode="numeric"
            value={scenario.overs}
            onChange={(e) => setNumber('overs', e.target.value, { min: 0, max: 50 })}
            className="cricket-input"
          />
        </div>
        <div>
          <label className="label" htmlFor="last10">Last 10 ov</label>
          <input
            id="last10" type="number" inputMode="numeric"
            value={scenario.runs_last_10}
            onChange={(e) => setNumber('runs_last_10', e.target.value, { min: 0, max: 200 })}
            className="cricket-input"
          />
        </div>

        <div className="flex items-end">
          <button
            onClick={onPredict}
            disabled={!ready || predicting}
            className="btn-primary h-[34px] w-full whitespace-nowrap lg:w-auto"
          >
            {predicting && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
            {predicting ? 'Predicting…' : 'Predict score'}
          </button>
        </div>
      </div>

      {/* Optional batsmen tucked away — they rarely change the answer much. */}
      <details className="group mt-3">
        <summary className="inline-flex cursor-pointer list-none items-center gap-1.5 text-[11px] text-dark-muted transition-colors hover:text-dark-text">
          <span className="transition-transform group-open:rotate-90">›</span>
          Set batsmen at the crease (optional)
        </summary>
        <div className="mt-2.5 grid gap-2.5 sm:grid-cols-2 lg:max-w-md">
          <div>
            <label className="label" htmlFor="bat1">At the crease</label>
            <select
              id="bat1" value={scenario.batsman_1}
              onChange={(e) => set('batsman_1', e.target.value)}
              className="cricket-select"
            >
              <option value="">None</option>
              {battingPlayers.map((p) => (
                <option key={p.id} value={p.name}>{p.name}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="label" htmlFor="bat2">Partner</label>
            <select
              id="bat2" value={scenario.batsman_2}
              onChange={(e) => set('batsman_2', e.target.value)}
              className="cricket-select"
            >
              <option value="">None</option>
              {battingPlayers.map((p) => (
                <option key={p.id} value={p.name}>{p.name}</option>
              ))}
            </select>
          </div>
        </div>
      </details>

      {missing.length > 0 && (
        <p className="mt-2.5 text-[11px] text-cricket-gold/90">
          Add {missing.join(', ')} to run a prediction.
        </p>
      )}
    </motion.section>
  );
};

export default MatchScenario;
