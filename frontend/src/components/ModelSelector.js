import React from 'react';
import { motion } from 'framer-motion';
import { Brain, Zap, TrendingUp } from 'lucide-react';

const ModelSelector = ({ selectedModel, onModelChange }) => {
  const models = [
    {
      id: 'xgboost',
      name: 'XGBoost',
      icon: <Zap className="h-5 w-5" />,
      accuracy: '86.2% R²',
      description: 'BEST: 86.2% accuracy - Production ready',
      badge: '🏆 Best'
    },
    {
      id: 'random_forest',
      name: 'Random Forest',
      icon: <Brain className="h-5 w-5" />,
      accuracy: '82.5% R²',
      description: 'GOOD: 82.5% accuracy - Reliable predictions'
    },
    {
      id: 'linear_regression',
      name: 'Linear Regression',
      icon: <TrendingUp className="h-5 w-5" />,
      accuracy: '68.0% R²',
      description: 'BASELINE: 68.0% accuracy - Fast predictions'
    }
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="cricket-card mb-8"
    >
      <h3 className="text-xl font-semibold text-cricket-green mb-4 flex items-center">
        <Brain className="h-6 w-6 mr-2" />
        Select ML Model
      </h3>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {models.map((model) => (
          <motion.button
            key={model.id}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => onModelChange(model.id)}
            className={`p-4 rounded-lg border-2 transition-all duration-300 ${
              selectedModel === model.id
                ? 'border-cricket-green bg-cricket-green/10 text-cricket-green'
                : 'border-dark-border bg-dark-card hover:border-cricket-green/50'
            }`}
          >
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center">
                {model.icon}
                <span className="ml-2 font-semibold">{model.name}</span>
              </div>
              {model.badge && (
                <span className="text-xs bg-cricket-green text-white px-2 py-1 rounded-full">
                  {model.badge}
                </span>
              )}
            </div>
            <div className="text-sm text-dark-muted">
              <div className="font-medium text-cricket-gold">{model.accuracy}</div>
              <div>{model.description}</div>
            </div>
          </motion.button>
        ))}
      </div>
    </motion.div>
  );
};

export default ModelSelector;
