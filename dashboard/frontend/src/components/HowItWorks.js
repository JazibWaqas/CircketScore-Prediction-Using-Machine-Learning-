import React from 'react';

/* Mirrors the per-stage confidence the API returns (backend/utils/predictions.py). */
const STAGES = [
  { stage: 'Pre-match', mae: 41, r2: 0.35 },
  { stage: 'Overs 1–10', mae: 29, r2: 0.62 },
  { stage: 'Overs 11–20', mae: 24, r2: 0.75 },
  { stage: 'Overs 21–30', mae: 18, r2: 0.86 },
  { stage: 'Overs 31–50', mae: 12, r2: 0.94 },
];

const PIPELINE = [
  {
    n: '01',
    title: 'Ball-by-ball ingest',
    body: 'Historical ODI commentary parsed into per-innings state: score, wickets, balls bowled and recent scoring momentum.',
  },
  {
    n: '02',
    title: 'Player and venue features',
    body: 'Career batting averages, bowling economies, elite counts and squad depth are aggregated per XI, then joined to historical venue par scores.',
  },
  {
    n: '03',
    title: 'Ensemble regression',
    body: 'Random Forest and XGBoost models map that state to a final total. Training uses a chronological split so no future match data leaks backwards.',
  },
  {
    n: '04',
    title: 'Stage-aware confidence',
    body: 'Error is reported separately for each phase of the innings, so the margin shown always reflects how much the model actually knows.',
  },
];

const HowItWorks = () => {
  const maxMae = Math.max(...STAGES.map((s) => s.mae));

  return (
    <section id="how-it-works" className="border-t border-dark-border bg-ink-950">
      <div className="mx-auto max-w-6xl px-6 py-12">
        <span className="eyebrow">Methodology</span>
        <h2 className="mt-2 max-w-2xl text-2xl font-bold tracking-tight text-white">
          Accuracy improves as the innings unfolds
        </h2>
        <p className="mt-2.5 max-w-2xl text-[13px] leading-relaxed text-dark-muted">
          Before a ball is bowled there is very little to go on, so early forecasts carry a wide
          margin. As the score, wickets and scoring rate come in, the model has far more to work
          with and the error drops sharply.
        </p>

        <div className="mt-7 grid gap-4 lg:grid-cols-[1.15fr_1fr]">
          {/* Error by stage */}
          <div className="surface p-5">
            <div className="mb-4 flex items-baseline justify-between">
              <h3 className="text-[13px] font-semibold text-white">Error by match stage</h3>
              <span className="text-[10px] text-dark-muted">257 unseen international ODIs</span>
            </div>

            <div className="space-y-2.5">
              {STAGES.map((s) => (
                <div key={s.stage} className="grid grid-cols-[6rem_1fr_auto] items-center gap-3">
                  <span className="text-[11px] text-dark-muted">{s.stage}</span>
                  <div className="h-5 overflow-hidden rounded bg-ink-800">
                    <div
                      className="flex h-full items-center justify-end rounded bg-accent/70 px-1.5 transition-[width] duration-700"
                      style={{ width: `${(s.mae / maxMae) * 100}%` }}
                    >
                      <span className="stat-num text-[10px] font-semibold text-ink-950">
                        ±{s.mae.toFixed(1)}
                      </span>
                    </div>
                  </div>
                  <span className="stat-num w-14 text-right text-[11px] text-dark-muted">
                    R² {s.r2.toFixed(3)}
                  </span>
                </div>
              ))}
            </div>

            <p className="mt-3.5 border-t border-dark-border pt-3 text-[11px] leading-relaxed text-dark-muted">
              Death overs reach an R² of <b className="stat-num text-dark-text">0.94</b> at{' '}
              <b className="stat-num text-dark-text">±12</b> runs, cutting the pre-match error by 71%.
            </p>
          </div>

          {/* Pipeline */}
          <div className="grid gap-px overflow-hidden rounded-2xl border border-dark-border bg-dark-border sm:grid-cols-2">
            {PIPELINE.map((p) => (
              <div key={p.n} className="bg-ink-900 p-4">
                <span className="stat-num text-[10px] font-semibold text-accent">{p.n}</span>
                <h3 className="mt-1 text-[13px] font-semibold text-white">{p.title}</h3>
                <p className="mt-1.5 text-[11px] leading-relaxed text-dark-muted">{p.body}</p>
              </div>
            ))}
          </div>
        </div>

        <p className="mt-5 text-[11px] text-dark-muted">
          Inputs cover squad composition and match state. Pitch, weather and toss are out of scope.
        </p>
      </div>
    </section>
  );
};

export default HowItWorks;
