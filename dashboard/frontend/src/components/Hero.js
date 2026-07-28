import React from 'react';
import { motion } from 'framer-motion';
import { ArrowRight, Zap } from 'lucide-react';

const CREDENTIALS = [
  { value: '0.94', unit: '', label: 'Death-overs R²' },
  { value: '±12', unit: 'runs', label: 'Death-overs error' },
  { value: '257', unit: '', label: 'Unseen ODIs tested' },
  { value: '18', unit: '', label: 'Model features' },
];

const Hero = ({ onTryScenario }) => (
  <section className="relative overflow-hidden border-b border-dark-border">
    <div
      aria-hidden
      className="pointer-events-none absolute inset-0 opacity-[0.5]"
      style={{
        background:
          'radial-gradient(700px 280px at 12% -20%, rgba(34,212,107,.16), transparent 65%), radial-gradient(560px 240px at 85% 0%, rgba(91,157,255,.10), transparent 60%)',
      }}
    />

    <div className="relative mx-auto max-w-6xl px-6 py-12 md:py-16">
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.45, ease: [0.16, 1, 0.3, 1] }}
        className="flex flex-col gap-8 lg:flex-row lg:items-end lg:justify-between"
      >
        <div className="max-w-2xl">
          <span className="chip mb-4 !py-0.5">
            <span className="h-1.5 w-1.5 rounded-full bg-accent" />
            Machine learning · ODI cricket
          </span>

          <h1 className="text-display font-bold text-white">
            What will they
            <br />
            <span className="text-accent">finish on</span>?
          </h1>

          <p className="mt-4 max-w-xl text-sm leading-relaxed text-dark-muted">
            Enter the score, wickets, overs, venue and both squads from any ODI innings. A model
            trained on ball-by-ball history projects the final total, tightening to within 12 runs
            as the innings closes.
          </p>

          <div className="mt-6 flex flex-wrap items-center gap-2.5">
            <button onClick={onTryScenario} className="btn-primary !py-2.5">
              <Zap className="h-4 w-4" />
              Load a live scenario
            </button>
            <a href="#how-it-works" className="btn-ghost !py-2.5">
              How it works
              <ArrowRight className="h-4 w-4" />
            </a>
          </div>
        </div>

        {/* Credentials sit beside the copy on desktop instead of adding another band. */}
        <dl className="grid shrink-0 grid-cols-2 gap-x-8 gap-y-4 border-t border-dark-border pt-5 lg:mb-1 lg:border-l lg:border-t-0 lg:pl-8 lg:pt-0">
          {CREDENTIALS.map((c) => (
            <div key={c.label}>
              <dd className="stat-num text-xl font-semibold text-white">
                {c.value}
                {c.unit && <span className="ml-1 text-xs text-dark-muted">{c.unit}</span>}
              </dd>
              <dt className="mt-0.5 text-[11px] text-dark-muted">{c.label}</dt>
            </div>
          ))}
        </dl>
      </motion.div>
    </div>
  </section>
);

export default Hero;
