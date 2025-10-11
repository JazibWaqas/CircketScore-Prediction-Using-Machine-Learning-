import React from 'react';
import { motion } from 'framer-motion';

const Header = () => {
  return (
    <motion.header 
      initial={{ y: -20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      className="bg-gradient-to-r from-dark-card to-dark-border border-b-2 border-cricket-green/30 py-6 px-8 shadow-xl"
    >
      <div className="max-w-7xl mx-auto">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="w-12 h-12 bg-cricket-green rounded-lg flex items-center justify-center">
              <span className="text-2xl">🏏</span>
            </div>
            <div>
              <h1 className="text-3xl font-bold text-white">ODI Progressive Predictor</h1>
              <p className="text-dark-muted text-sm">Fantasy Team Builder & Progressive Score Prediction</p>
            </div>
          </div>
          <div className="text-right">
            <div className="text-cricket-green font-semibold">Progressive Accuracy</div>
            <div className="text-sm text-dark-muted">Pre-match: 35% → Death: 94%</div>
          </div>
        </div>
      </div>
    </motion.header>
  );
};

export default Header;

