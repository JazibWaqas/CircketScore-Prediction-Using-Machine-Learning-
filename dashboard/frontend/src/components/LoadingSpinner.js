import React from 'react';

/** Full-page boot state. The API cold-starts on Render, so this can be visible for a while. */
const LoadingSpinner = ({ message = 'Loading match data…' }) => (
  <div className="mx-auto w-full max-w-6xl px-6 py-24">
    <div className="mb-10 space-y-3">
      <div className="skeleton h-10 w-2/3 max-w-lg" />
      <div className="skeleton h-4 w-1/2 max-w-md" />
    </div>
    <div className="grid gap-5 lg:grid-cols-3">
      <div className="skeleton h-72" />
      <div className="skeleton h-72" />
      <div className="skeleton h-72" />
    </div>
    <p className="mt-8 text-center text-xs text-dark-muted">
      {message} <span className="text-dark-muted/70">The API may take a moment to wake up.</span>
    </p>
  </div>
);

export default LoadingSpinner;
