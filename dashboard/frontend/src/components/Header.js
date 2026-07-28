import React from 'react';
import { Github } from 'lucide-react';

const Header = () => (
  <header className="sticky top-0 z-40 border-b border-dark-border bg-ink-950/80 backdrop-blur-md">
    <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-3.5">
      <a href="#top" className="flex items-center gap-2.5">
        <span className="flex h-7 w-7 items-center justify-center rounded-md bg-accent text-sm">
          🏏
        </span>
        <span className="text-sm font-semibold tracking-tight text-white">
          ODI Score Predictor
        </span>
      </a>

      <nav className="flex items-center gap-5 text-xs text-dark-muted">
        <a href="#predictor" className="transition-colors hover:text-dark-text">Predictor</a>
        <a href="#how-it-works" className="hidden transition-colors hover:text-dark-text sm:inline">
          How it works
        </a>
        <a
          href="https://github.com/JazibWaqas/CircketScore-Prediction-Using-Machine-Learning-"
          target="_blank"
          rel="noreferrer noopener"
          className="flex items-center gap-1.5 transition-colors hover:text-dark-text"
        >
          <Github className="h-3.5 w-3.5" />
          <span className="hidden sm:inline">Source</span>
        </a>
      </nav>
    </div>
  </header>
);

export default Header;
