import React from 'react';
import { motion } from 'framer-motion';
import { AnimatedNumber, StatTile } from './ui/Primitives';

const STAGE_COPY = {
  'pre-match': 'Pre-match',
  early: 'Overs 1–10',
  mid: 'Overs 11–20',
  late: 'Overs 21–30',
  death: 'Overs 31–50',
};

const PredictionDisplay = ({ prediction, scenario, predicting }) => {
  // Always mounted directly beneath the inputs, so it needs a resting state:
  // show the live match readout until a projection has been run.
  if (!prediction) {
    const cs = Number(scenario.current_score) || 0;
    const wk = Number(scenario.wickets_fallen) || 0;
    const ov = Number(scenario.overs) || 0;
    const started = scenario.current_score !== '' || ov > 0;
    const rr = ov > 0 ? (cs / ov).toFixed(2) : null;

    return (
      <section className="surface flex flex-col items-center justify-center px-5 py-8 text-center">
        <div className="eyebrow mb-2">
          {started ? 'Current match state' : 'No innings loaded'}
        </div>

        {started ? (
          <>
            <div className="stat-num text-3xl font-bold text-white">
              {cs}/{wk}
              <span className="ml-2 text-base font-medium text-dark-muted">({ov} ov)</span>
            </div>
            <div className="mt-2 text-[13px] text-dark-muted">
              {rr && (
                <>
                  Run rate <b className="stat-num text-dark-text">{rr}</b>
                  <span className="mx-1.5 opacity-40">·</span>
                </>
              )}
              <b className="stat-num text-dark-text">{Math.max(0, 50 - ov)}</b> overs remaining
            </div>
          </>
        ) : (
          <div className="text-2xl font-semibold text-dark-muted">Set up an innings</div>
        )}

        <p className="mt-4 max-w-sm text-[13px] leading-relaxed text-dark-muted">
          {predicting
            ? 'Running the model…'
            : started
              ? 'Hit Predict score to project the final total.'
              : 'Pick both squads and a venue below, then enter the score, wickets and overs to get a projection.'}
        </p>
      </section>
    );
  }

  const { predicted_score, confidence, team_stats } = prediction;

  const score = Math.round(predicted_score);
  const mae = confidence.mae;
  const low = score - mae;
  const high = score + mae;

  const par = Math.round(scenario.venue_avg_score || 250);
  const vsPar = score - par;

  const current = Number(scenario.current_score) || 0;
  const overs = Number(scenario.overs) || 0;
  const remaining = Math.max(0, score - current);
  const oversLeft = Math.max(0, 50 - overs);
  const requiredRate = oversLeft > 0 ? remaining / oversLeft : 0;
  const currentRate = overs > 0 ? current / overs : 0;

  // Position the range bar on a scale anchored around par so the reader has a
  // reference frame rather than a bare number.
  const scaleMin = Math.min(low, par) - 40;
  const scaleMax = Math.max(high, par) + 40;
  const pos = (v) => ((v - scaleMin) / (scaleMax - scaleMin)) * 100;

  return (
    <motion.section
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, ease: [0.16, 1, 0.3, 1] }}
      className="surface overflow-hidden"
    >
      {/* ---- Headline ------------------------------------------------ */}
      <div
        className="relative border-b border-dark-border px-5 py-6 text-center"
        style={{
          background:
            'radial-gradient(600px 200px at 50% 0%, rgba(34,212,107,.12), transparent 70%)',
        }}
      >
        <div className="eyebrow mb-2">Projected final score</div>

        <div className="text-score font-bold text-accent">
          <AnimatedNumber value={score} />
        </div>

        <div className="mt-2 flex flex-wrap items-center justify-center gap-x-2.5 gap-y-1 text-[13px] text-dark-muted">
          <span className="stat-num">{low}–{high} runs</span>
          <span aria-hidden>·</span>
          <span>{confidence.label} confidence</span>
          <span aria-hidden>·</span>
          <span>{STAGE_COPY[confidence.stage] || confidence.stage}</span>
        </div>

        {/* Range vs par */}
        <div className="mx-auto mt-5 max-w-md">
          <div className="relative h-9">
            <div className="absolute inset-x-0 top-4 h-1 rounded-full bg-ink-800" />
            <div
              className="absolute top-4 h-1 rounded-full bg-accent/35"
              style={{ left: `${pos(low)}%`, width: `${pos(high) - pos(low)}%` }}
            />
            <div
              className="absolute top-[9px] h-[14px] w-[3px] -translate-x-1/2 rounded-full bg-accent"
              style={{ left: `${pos(score)}%` }}
            />
            <div
              className="absolute top-2 flex -translate-x-1/2 flex-col items-center"
              style={{ left: `${pos(par)}%` }}
            >
              <div className="h-4 w-px bg-dark-muted/70" />
            </div>
          </div>
          <div className="relative mt-1 h-4 text-[11px] text-dark-muted">
            <span
              className="absolute -translate-x-1/2 whitespace-nowrap"
              style={{ left: `${pos(par)}%` }}
            >
              venue par {par}
            </span>
          </div>
        </div>

        <p className="mt-3.5 text-[13px] text-dark-text">
          {Math.abs(vsPar) < 6 ? (
            <>Right on par for this venue.</>
          ) : (
            <>
              <span className={vsPar > 0 ? 'text-accent' : 'text-cricket-red'}>
                {Math.abs(vsPar)} runs {vsPar > 0 ? 'above' : 'below'}
              </span>{' '}
              the venue average.
            </>
          )}
        </p>
      </div>

      {/* ---- Reading of the innings --------------------------------- */}
      <div className="grid gap-2.5 border-b border-dark-border p-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatTile
          label="Runs still to come"
          value={remaining}
          hint={`over the last ${oversLeft} overs`}
          tone="accent"
        />
        <StatTile
          label="Implied run rate"
          value={requiredRate.toFixed(2)}
          hint={`current ${currentRate.toFixed(2)}`}
          tone={requiredRate > currentRate ? 'gold' : 'default'}
        />
        <StatTile
          label="Model error"
          value={`±${mae}`}
          hint={`R² ${(confidence.r2 * 100).toFixed(0)}% at this stage`}
        />
        <StatTile
          label="Wickets in hand"
          value={10 - (Number(scenario.wickets_fallen) || 0)}
          hint={`${scenario.current_score}/${scenario.wickets_fallen} after ${overs} ov`}
        />
      </div>

      {/* ---- What drove it ------------------------------------------ */}
      <div className="grid gap-5 p-4 md:grid-cols-2">
        <div>
          <div className="mb-3 flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-accent" />
            <h3 className="text-sm font-semibold text-white">Batting side</h3>
          </div>
          <dl className="space-y-2 text-[13px]">
            <Row
              label="Team batting average"
              value={team_stats.batting.team_batting_avg.toFixed(1)}
              hint="ODI average ≈ 28"
            />
            <Row
              label="Elite batsmen (avg ≥ 40)"
              value={team_stats.batting.team_elite_batsmen}
              hint="of 11"
            />
            <Row
              label="Batting depth (avg ≥ 30)"
              value={team_stats.batting.team_batting_depth}
              hint="of 11"
            />
          </dl>
        </div>

        <div>
          <div className="mb-3 flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-oppo" />
            <h3 className="text-sm font-semibold text-white">Bowling side</h3>
          </div>
          <dl className="space-y-2 text-[13px]">
            <Row
              label="Bowling economy"
              value={team_stats.bowling.opp_bowling_economy.toFixed(2)}
              hint="lower is better · ODI ≈ 5.2"
            />
            <Row
              label="Elite bowlers"
              value={team_stats.bowling.opp_elite_bowlers}
              hint="of 11"
            />
            <Row
              label="Bowling depth"
              value={team_stats.bowling.opp_bowling_depth}
              hint="of 11"
            />
          </dl>
        </div>
      </div>
    </motion.section>
  );
};

const Row = ({ label, value, hint }) => (
  <div className="flex items-baseline justify-between gap-4 border-b border-dark-border/60 pb-1.5 last:border-0">
    <dt className="text-dark-muted">{label}</dt>
    <dd className="text-right">
      <span className="stat-num font-semibold text-white">{value}</span>
      {hint && <span className="ml-2 text-[11px] text-dark-muted">{hint}</span>}
    </dd>
  </div>
);

export default PredictionDisplay;
