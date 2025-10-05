import React from 'react';
import { motion } from 'framer-motion';
import { MapPin, Calendar, Settings, Trophy, Home, Zap } from 'lucide-react';
import DatePicker from 'react-datepicker';
import 'react-datepicker/dist/react-datepicker.css';

const MatchContext = ({ venues, teams, context, onContextChange }) => {
  const handleChange = (field, value) => {
    onContextChange(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const availableTeams = teams.filter(team => team.id);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="cricket-card"
    >
      <div className="flex items-center mb-6">
        <Settings className="h-6 w-6 text-cricket-green mr-2" />
        <h3 className="text-xl font-semibold text-cricket-green">
          Match Context
        </h3>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {/* Venue Selection */}
        <div>
          <label className="block text-sm font-medium text-dark-muted mb-2">
            <MapPin className="h-4 w-4 inline mr-1" />
            Venue
          </label>
          <select
            value={context.venue?.id || ''}
            onChange={(e) => {
              const venueId = parseInt(e.target.value);
              const selectedVenue = venues.find(v => v.venue_id === venueId);
              handleChange('venue', selectedVenue);
            }}
            className="cricket-select w-full"
          >
            <option value="">Select venue...</option>
            {venues.map(venue => (
              <option key={venue.venue_id} value={venue.venue_id}>
                {venue.venue_name} ({venue.city}, {venue.country})
              </option>
            ))}
          </select>
          {context.venue && (
            <div className="mt-2 text-sm text-dark-muted">
              Avg runs: {context.venue.avg_runs_scored?.toFixed(1) || 'N/A'} | 
              Matches: {context.venue.total_matches || 'N/A'}
            </div>
          )}
        </div>

        {/* Date Selection */}
        <div>
          <label className="block text-sm font-medium text-dark-muted mb-2">
            <Calendar className="h-4 w-4 inline mr-1" />
            Match Date
          </label>
          <DatePicker
            selected={context.date}
            onChange={(date) => handleChange('date', date)}
            dateFormat="dd/MM/yyyy"
            className="cricket-input w-full"
            placeholderText="Select date..."
          />
        </div>

        {/* Toss Winner */}
        <div>
          <label className="block text-sm font-medium text-dark-muted mb-2">
            <Trophy className="h-4 w-4 inline mr-1" />
            Toss Winner
          </label>
          <select
            value={context.tossWinner?.id || ''}
            onChange={(e) => {
              const teamId = parseInt(e.target.value);
              const selectedTeam = availableTeams.find(t => t.id === teamId);
              handleChange('tossWinner', selectedTeam);
            }}
            className="cricket-select w-full"
          >
            <option value="">Select toss winner...</option>
            {availableTeams.map(team => (
              <option key={team.id} value={team.id}>
                {team.name}
              </option>
            ))}
          </select>
        </div>

        {/* Toss Decision */}
        <div>
          <label className="block text-sm font-medium text-dark-muted mb-2">
            Toss Decision
          </label>
          <select
            value={context.tossDecision || ''}
            onChange={(e) => handleChange('tossDecision', e.target.value)}
            className="cricket-select w-full"
          >
            <option value="">Select decision...</option>
            <option value="bat">Bat First</option>
            <option value="field">Field First</option>
          </select>
        </div>

        {/* Batting First */}
        <div>
          <label className="block text-sm font-medium text-dark-muted mb-2">
            Batting First
          </label>
          <select
            value={context.battingFirst?.id || ''}
            onChange={(e) => {
              const teamId = parseInt(e.target.value);
              const selectedTeam = availableTeams.find(t => t.id === teamId);
              handleChange('battingFirst', selectedTeam);
            }}
            className="cricket-select w-full"
          >
            <option value="">Select batting first...</option>
            {availableTeams.map(team => (
              <option key={team.id} value={team.id}>
                {team.name}
              </option>
            ))}
          </select>
        </div>

        {/* Season Year */}
        <div>
          <label className="block text-sm font-medium text-dark-muted mb-2">
            Season Year
          </label>
          <input
            type="number"
            value={context.seasonYear || 2024}
            onChange={(e) => handleChange('seasonYear', parseInt(e.target.value))}
            className="cricket-input w-full"
            min="2005"
            max="2030"
          />
        </div>
      </div>

      {/* Toggle Options */}
      <div className="mt-6">
        <h4 className="text-sm font-medium text-dark-muted mb-4">Match Context</h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {/* Home Team Advantage */}
          <label className="flex items-center space-x-3 cursor-pointer">
            <input
              type="checkbox"
              checked={context.isHomeTeam}
              onChange={(e) => handleChange('isHomeTeam', e.target.checked)}
              className="w-4 h-4 text-cricket-green bg-dark-card border-dark-border rounded focus:ring-cricket-green"
            />
            <div className="flex items-center">
              <Home className="h-4 w-4 mr-1 text-dark-muted" />
              <span className="text-sm text-dark-text">Home Advantage</span>
            </div>
          </label>

          {/* Final Match */}
          <label className="flex items-center space-x-3 cursor-pointer">
            <input
              type="checkbox"
              checked={context.isFinal}
              onChange={(e) => handleChange('isFinal', e.target.checked)}
              className="w-4 h-4 text-cricket-green bg-dark-card border-dark-border rounded focus:ring-cricket-green"
            />
            <div className="flex items-center">
              <Trophy className="h-4 w-4 mr-1 text-dark-muted" />
              <span className="text-sm text-dark-text">Final Match</span>
            </div>
          </label>

          {/* IPL Match */}
          <label className="flex items-center space-x-3 cursor-pointer">
            <input
              type="checkbox"
              checked={context.isIPL}
              onChange={(e) => handleChange('isIPL', e.target.checked)}
              className="w-4 h-4 text-cricket-green bg-dark-card border-dark-border rounded focus:ring-cricket-green"
            />
            <div className="flex items-center">
              <Zap className="h-4 w-4 mr-1 text-dark-muted" />
              <span className="text-sm text-dark-text">IPL Match</span>
            </div>
          </label>

          {/* T20 World Cup */}
          <label className="flex items-center space-x-3 cursor-pointer">
            <input
              type="checkbox"
              checked={context.isT20WorldCup}
              onChange={(e) => handleChange('isT20WorldCup', e.target.checked)}
              className="w-4 h-4 text-cricket-green bg-dark-card border-dark-border rounded focus:ring-cricket-green"
            />
            <div className="flex items-center">
              <Trophy className="h-4 w-4 mr-1 text-dark-muted" />
              <span className="text-sm text-dark-text">T20 World Cup</span>
            </div>
          </label>
        </div>
      </div>

      {/* Context Summary */}
      {(context.venue || context.tossWinner || context.battingFirst) && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-6 p-4 bg-cricket-green/10 border border-cricket-green/30 rounded-lg"
        >
          <h4 className="text-sm font-medium text-cricket-green mb-2">Match Summary</h4>
          <div className="text-sm text-dark-muted space-y-1">
            {context.venue && <div>📍 Venue: {context.venue.venue_name}</div>}
            {context.tossWinner && <div>🏆 Toss Winner: {context.tossWinner.name}</div>}
            {context.tossDecision && <div>⚡ Decision: {context.tossDecision}</div>}
            {context.battingFirst && <div>🏏 Batting First: {context.battingFirst.name}</div>}
            {context.isFinal && <div>🏆 Final Match</div>}
            {context.isIPL && <div>⚡ IPL Match</div>}
            {context.isT20WorldCup && <div>🌍 T20 World Cup</div>}
          </div>
        </motion.div>
      )}
    </motion.div>
  );
};

export default MatchContext;
