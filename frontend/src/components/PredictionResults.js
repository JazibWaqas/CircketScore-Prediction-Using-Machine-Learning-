import React from 'react';
import { motion } from 'framer-motion';
import { Trophy, Target, Brain, TrendingUp, Zap, Award } from 'lucide-react';

const PredictionResults = ({ prediction }) => {
  const isTeamAWinner = prediction.predicted_score_a > prediction.predicted_score_b;
  const scoreDifference = Math.abs(prediction.predicted_score_a - prediction.predicted_score_b);

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5, type: "spring", stiffness: 100 }}
      className="prediction-card"
    >
      {/* Header */}
      <div className="text-center mb-8">
        <motion.div
          animate={{ rotate: [0, 5, -5, 0] }}
          transition={{ duration: 2, repeat: Infinity, repeatDelay: 3 }}
          className="inline-block mb-4"
        >
          <Trophy className="h-16 w-16 text-cricket-gold mx-auto" />
        </motion.div>
        
        <h2 className="text-3xl font-bold text-cricket-green mb-2">
          Match Prediction
        </h2>
        <p className="text-dark-muted">
          {prediction.team_a} vs {prediction.team_b}
        </p>
        <p className="text-sm text-dark-muted">
          at {prediction.venue}
        </p>
      </div>

      {/* Scores */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
        {/* Team A Score */}
        <motion.div
          initial={{ opacity: 0, x: -30 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
          className={`text-center p-6 rounded-xl border-2 ${
            isTeamAWinner 
              ? 'border-cricket-green bg-cricket-green/10' 
              : 'border-dark-border bg-dark-card'
          }`}
        >
          <div className="flex items-center justify-center mb-4">
            {isTeamAWinner && <Award className="h-6 w-6 text-cricket-gold mr-2" />}
            <h3 className="text-xl font-semibold text-cricket-green">
              {prediction.team_a}
            </h3>
          </div>
          <div className="score-display text-5xl font-bold text-cricket-green">
            {Math.round(prediction.predicted_score_a)}
          </div>
          <div className="text-sm text-dark-muted mt-2">predicted runs</div>
        </motion.div>

        {/* Team B Score */}
        <motion.div
          initial={{ opacity: 0, x: 30 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.4 }}
          className={`text-center p-6 rounded-xl border-2 ${
            !isTeamAWinner 
              ? 'border-cricket-green bg-cricket-green/10' 
              : 'border-dark-border bg-dark-card'
          }`}
        >
          <div className="flex items-center justify-center mb-4">
            {!isTeamAWinner && <Award className="h-6 w-6 text-cricket-gold mr-2" />}
            <h3 className="text-xl font-semibold text-cricket-green">
              {prediction.team_b}
            </h3>
          </div>
          <div className="score-display text-5xl font-bold text-cricket-green">
            {Math.round(prediction.predicted_score_b)}
          </div>
          <div className="text-sm text-dark-muted mt-2">predicted runs</div>
        </motion.div>
      </div>

      {/* Winner Prediction */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
        className="text-center mb-8"
      >
        <div className="inline-flex items-center bg-cricket-gold/20 text-cricket-gold px-6 py-3 rounded-full border border-cricket-gold/30">
          <Trophy className="h-5 w-5 mr-2" />
          <span className="font-semibold text-lg">
            Predicted Winner: {prediction.predicted_winner}
          </span>
        </div>
        <div className="text-sm text-dark-muted mt-2">
          Margin: {scoreDifference.toFixed(1)} runs
        </div>
      </motion.div>

      {/* Model Information */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        {/* Confidence */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8 }}
          className="text-center p-4 bg-dark-card border border-dark-border rounded-lg"
        >
          <Target className="h-8 w-8 text-cricket-green mx-auto mb-2" />
          <div className="text-2xl font-bold text-cricket-green">
            {Math.round(prediction.confidence * 100)}%
          </div>
          <div className="text-sm text-dark-muted">Confidence</div>
        </motion.div>

        {/* Model Used */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.0 }}
          className="text-center p-4 bg-dark-card border border-dark-border rounded-lg"
        >
          <Brain className="h-8 w-8 text-cricket-green mx-auto mb-2" />
          <div className="text-lg font-semibold text-cricket-green capitalize">
            {prediction.model_used.replace('_', ' ')}
          </div>
          <div className="text-sm text-dark-muted">ML Model</div>
        </motion.div>

        {/* Prediction Quality */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.2 }}
          className="text-center p-4 bg-dark-card border border-dark-border rounded-lg"
        >
          <TrendingUp className="h-8 w-8 text-cricket-green mx-auto mb-2" />
          <div className="text-lg font-semibold text-cricket-green">
            {scoreDifference > 20 ? 'High' : scoreDifference > 10 ? 'Medium' : 'Close'}
          </div>
          <div className="text-sm text-dark-muted">Match Intensity</div>
        </motion.div>
      </div>

      {/* Additional Context */}
      {prediction.match_context && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.4 }}
          className="bg-dark-card border border-dark-border rounded-lg p-4"
        >
          <h4 className="text-sm font-medium text-cricket-green mb-3 flex items-center">
            <Zap className="h-4 w-4 mr-1" />
            Match Context
          </h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            {prediction.match_context.isFinal && (
              <div className="flex items-center text-cricket-gold">
                <Trophy className="h-4 w-4 mr-1" />
                Final Match
              </div>
            )}
            {prediction.match_context.isIPL && (
              <div className="flex items-center text-cricket-green">
                <Zap className="h-4 w-4 mr-1" />
                IPL Match
              </div>
            )}
            {prediction.match_context.isT20WorldCup && (
              <div className="flex items-center text-cricket-gold">
                <Trophy className="h-4 w-4 mr-1" />
                T20 World Cup
              </div>
            )}
            {prediction.match_context.isHomeTeam && (
              <div className="flex items-center text-cricket-green">
                <span className="mr-1">🏠</span>
                Home Advantage
              </div>
            )}
          </div>
        </motion.div>
      )}

      {/* Disclaimer */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.6 }}
        className="mt-6 text-center"
      >
        <p className="text-xs text-dark-muted">
          * Predictions are based on historical data and machine learning models. 
          Actual results may vary. For entertainment purposes only.
        </p>
      </motion.div>
    </motion.div>
  );
};

export default PredictionResults;
