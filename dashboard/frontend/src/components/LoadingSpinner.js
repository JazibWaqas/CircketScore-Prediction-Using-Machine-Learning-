import React from 'react';

const LoadingSpinner = () => {
  return (
    <div className="flex flex-col items-center justify-center space-y-4">
      <div className="loading-spinner"></div>
      <p className="text-dark-muted">Loading...</p>
    </div>
  );
};

export default LoadingSpinner;

