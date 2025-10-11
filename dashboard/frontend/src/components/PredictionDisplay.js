import React from 'react';
import { motion } from 'framer-motion';

const PredictionDisplay = ({ prediction, scenario }) => {
  const { predicted_score, confidence, team_stats } = prediction;
  
  const getConfidenceColor = (label) => {
    switch (label) {
      case 'Very High': return 'text-green-400';
      case 'High': return 'text-cricket-green';
      case 'Medium': return 'text-yellow-400';
      case 'Low': return 'text-orange-400';
      default: return 'text-dark-muted';
    }
  };
  
  const getStageDescription = (stage) => {
    const descriptions = {
      'pre-match': 'Pre-Match Prediction (Ball 0-10)',
      'early': 'Early Match (Overs 1-10)',
      'mid': 'Mid Match (Overs 11-20)',
      'late': 'Late Match (Overs 21-40)',
      'death': 'Death Overs (Overs 41-50)'
    };
    return descriptions[stage] || stage;
  };
  
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5 }}
      className="cricket-card bg-gradient-to-br from-dark-card to-dark-border"
    >
      <h2 className="text-2xl font-bold text-cricket-green mb-6">📊 Prediction Results</h2>
      
      {/* Main Prediction */}
      <div className="text-center py-8 mb-8 bg-dark-bg rounded-xl border border-cricket-green/30">
        <div className="text-dark-muted mb-2">Predicted Final Score</div>
        <motion.div
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ delay: 0.2, type: 'spring' }}
          className="score-display"
        >
          {Math.round(predicted_score)} runs
        </motion.div>
        <div className="mt-4 text-dark-muted">
          ± {confidence.mae} runs ({confidence.label} Confidence)
        </div>
        <div className="mt-2 text-sm text-dark-muted">
          {getStageDescription(confidence.stage)}
        </div>
      </div>
      
      {/* Confidence Details */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        <div className="bg-dark-bg p-4 rounded-lg">
          <div className="text-dark-muted text-sm mb-1">Confidence Level</div>
          <div className={`text-2xl font-bold ${getConfidenceColor(confidence.label)}`}>
            {confidence.label}
          </div>
        </div>
        
        <div className="bg-dark-bg p-4 rounded-lg">
          <div className="text-dark-muted text-sm mb-1">R² Score</div>
          <div className="text-2xl font-bold text-cricket-gold">
            {(confidence.r2 * 100).toFixed(1)}%
          </div>
        </div>
        
        <div className="bg-dark-bg p-4 rounded-lg">
          <div className="text-dark-muted text-sm mb-1">Expected Error</div>
          <div className="text-2xl font-bold text-white">
            ± {confidence.mae} runs
          </div>
        </div>
      </div>
      
      {/* Team Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-dark-bg p-6 rounded-lg border border-dark-border">
          <h3 className="text-xl font-bold text-cricket-green mb-4">Batting Team Stats</h3>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-dark-muted">Team Batting Average:</span>
              <span className="text-white font-semibold">
                {team_stats.batting.team_batting_avg.toFixed(2)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-dark-muted">Elite Batsmen:</span>
              <span className="text-white font-semibold">
                {team_stats.batting.team_elite_batsmen}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-dark-muted">Batting Depth:</span>
              <span className="text-white font-semibold">
                {team_stats.batting.team_batting_depth}
              </span>
            </div>
          </div>
        </div>
        
        <div className="bg-dark-bg p-6 rounded-lg border border-dark-border">
          <h3 className="text-xl font-bold text-cricket-green mb-4">Opposition Stats</h3>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-dark-muted">Bowling Economy:</span>
              <span className="text-white font-semibold">
                {team_stats.bowling.opp_bowling_economy.toFixed(2)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-dark-muted">Elite Bowlers:</span>
              <span className="text-white font-semibold">
                {team_stats.bowling.opp_elite_bowlers}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-dark-muted">Bowling Depth:</span>
              <span className="text-white font-semibold">
                {team_stats.bowling.opp_bowling_depth}
              </span>
            </div>
          </div>
        </div>
      </div>
      
      {/* Match Context */}
      <div className="mt-6 pt-6 border-t border-dark-border">
        <h3 className="text-lg font-semibold text-white mb-4">Match Context</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div>
            <span className="text-dark-muted">Current:</span>
            <span className="ml-2 text-white font-medium">
              {scenario.current_score}/{scenario.wickets_fallen}
            </span>
          </div>
          <div>
            <span className="text-dark-muted">Overs:</span>
            <span className="ml-2 text-white font-medium">
              {scenario.overs}/50
            </span>
          </div>
          <div>
            <span className="text-dark-muted">Venue:</span>
            <span className="ml-2 text-white font-medium truncate">
              {scenario.venue.split(',')[0]}
            </span>
          </div>
          <div>
            <span className="text-dark-muted">Venue Avg:</span>
            <span className="ml-2 text-white font-medium">
              {Math.round(scenario.venue_avg_score)}
            </span>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default PredictionDisplay;

