import React from 'react';
import { motion } from 'framer-motion';
import { Target } from 'lucide-react';

const Header = () => {
  return (
    <motion.header
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="bg-dark-card border-b border-dark-border"
    >
      <div className="container mx-auto px-4 py-6">
        <div className="flex items-center justify-center">
          <motion.div
            animate={{ rotate: [0, 10, -10, 0] }}
            transition={{ duration: 2, repeat: Infinity, repeatDelay: 3 }}
            className="mr-4"
          >
            <Target className="h-12 w-12 text-cricket-green" />
          </motion.div>
          
          <div className="text-center">
            <h1 className="text-4xl font-bold text-cricket-green mb-2">
              Cricket Score Predictor
            </h1>
            <p className="text-dark-muted text-lg">
              AI-Powered T20 Match Predictions
            </p>
          </div>
        </div>
      </div>
    </motion.header>
  );
};

export default Header;
