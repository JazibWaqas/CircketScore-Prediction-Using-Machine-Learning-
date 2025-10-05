import React from 'react';
import { motion } from 'framer-motion';
import { Brain, Zap, TrendingUp } from 'lucide-react';

const ModelSelector = ({ selectedModel, onModelChange }) => {
  const models = [
    {
      id: 'random_forest',
      name: 'Random Forest',
      icon: <Brain className="h-5 w-5" />,
      accuracy: '75% R²',
      description: 'Best overall performance'
    },
    {
      id: 'xgboost',
      name: 'XGBoost',
      icon: <Zap className="h-5 w-5" />,
      accuracy: '72% R²',
      description: 'Fast and accurate'
    },
    {
      id: 'linear_regression',
      name: 'Linear Regression',
      icon: <TrendingUp className="h-5 w-5" />,
      accuracy: '65% R²',
      description: 'Simple and interpretable'
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
            <div className="flex items-center mb-2">
              {model.icon}
              <span className="ml-2 font-semibold">{model.name}</span>
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
