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

  const availableTeams = teams.filter(team => team.team_id);

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
            value={context.venue?.venue_id || ''}
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
            value={context.tossWinner?.team_id || ''}
            onChange={(e) => {
              const teamId = parseInt(e.target.value);
              const selectedTeam = availableTeams.find(t => t.team_id === teamId);
              handleChange('tossWinner', selectedTeam);
            }}
            className="cricket-select w-full"
          >
            <option value="">Select toss winner...</option>
            {availableTeams.map(team => (
              <option key={team.team_id} value={team.team_id}>
                {team.team_name}
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
            value={context.battingFirst?.team_id || ''}
            onChange={(e) => {
              const teamId = parseInt(e.target.value);
              const selectedTeam = availableTeams.find(t => t.team_id === teamId);
              handleChange('battingFirst', selectedTeam);
            }}
            className="cricket-select w-full"
          >
            <option value="">Select batting first...</option>
            {availableTeams.map(team => (
              <option key={team.team_id} value={team.team_id}>
                {team.team_name}
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

      {/* Tournament Selection */}
      <div className="mt-6">
        <label className="block text-sm font-medium text-dark-muted mb-2">
          <Trophy className="h-4 w-4 inline mr-1" />
          Tournament Type
        </label>
        <select
          value={context.tournamentType || ''}
          onChange={(e) => handleChange('tournamentType', e.target.value)}
          className="cricket-select w-full"
        >
          <option value="">Select tournament...</option>
          <option value="bilateral">Bilateral Series</option>
          <option value="t20_world_cup">T20 World Cup</option>
          <option value="vitality_blast">Vitality Blast (England)</option>
          <option value="natwest_t20">NatWest T20 Blast</option>
          <option value="psl">Pakistan Super League</option>
          <option value="csa_t20">CSA T20 Challenge (South Africa)</option>
          <option value="ram_slam">Ram Slam T20 Challenge</option>
          <option value="t20_qualifier">T20 World Cup Qualifier</option>
          <option value="international_league">International League T20</option>
        </select>
      </div>

          {/* Season Information */}
          <div className="mt-6">
            <h4 className="text-sm font-medium text-dark-muted mb-4">Season Information</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
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
              <div>
                <label className="block text-sm font-medium text-dark-muted mb-2">
                  Season Month
                </label>
                <select
                  value={context.seasonMonth || 6}
                  onChange={(e) => handleChange('seasonMonth', parseInt(e.target.value))}
                  className="cricket-select w-full"
                >
                  <option value={1}>January</option>
                  <option value={2}>February</option>
                  <option value={3}>March</option>
                  <option value={4}>April</option>
                  <option value={5}>May</option>
                  <option value={6}>June</option>
                  <option value={7}>July</option>
                  <option value={8}>August</option>
                  <option value={9}>September</option>
                  <option value={10}>October</option>
                  <option value={11}>November</option>
                  <option value={12}>December</option>
                </select>
              </div>
            </div>
          </div>

          {/* Weather Conditions */}
          <div className="mt-6">
            <h4 className="text-sm font-medium text-dark-muted mb-4">Weather Conditions</h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <label className="flex items-center space-x-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={context.isWinter || false}
                  onChange={(e) => handleChange('isWinter', e.target.checked)}
                  className="w-4 h-4 text-cricket-green bg-dark-card border-dark-border rounded focus:ring-cricket-green"
                />
                <span className="text-sm text-dark-text">Winter Season</span>
              </label>
              <label className="flex items-center space-x-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={context.isSummer || false}
                  onChange={(e) => handleChange('isSummer', e.target.checked)}
                  className="w-4 h-4 text-cricket-green bg-dark-card border-dark-border rounded focus:ring-cricket-green"
                />
                <span className="text-sm text-dark-text">Summer Season</span>
              </label>
              <label className="flex items-center space-x-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={context.isMonsoon || false}
                  onChange={(e) => handleChange('isMonsoon', e.target.checked)}
                  className="w-4 h-4 text-cricket-green bg-dark-card border-dark-border rounded focus:ring-cricket-green"
                />
                <span className="text-sm text-dark-text">Monsoon Season</span>
              </label>
            </div>
          </div>

          {/* Match Context Options */}
          <div className="mt-6">
            <h4 className="text-sm font-medium text-dark-muted mb-4">Match Context</h4>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
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

              {/* Semi Final */}
              <label className="flex items-center space-x-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={context.isSemiFinal}
                  onChange={(e) => handleChange('isSemiFinal', e.target.checked)}
                  className="w-4 h-4 text-cricket-green bg-dark-card border-dark-border rounded focus:ring-cricket-green"
                />
                <div className="flex items-center">
                  <Trophy className="h-4 w-4 mr-1 text-dark-muted" />
                  <span className="text-sm text-dark-text">Semi Final</span>
                </div>
              </label>

              {/* Playoff */}
              <label className="flex items-center space-x-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={context.isPlayoff}
                  onChange={(e) => handleChange('isPlayoff', e.target.checked)}
                  className="w-4 h-4 text-cricket-green bg-dark-card border-dark-border rounded focus:ring-cricket-green"
                />
                <div className="flex items-center">
                  <Trophy className="h-4 w-4 mr-1 text-dark-muted" />
                  <span className="text-sm text-dark-text">Playoff</span>
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

              {/* IPL */}
              <label className="flex items-center space-x-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={context.isIPL}
                  onChange={(e) => handleChange('isIPL', e.target.checked)}
                  className="w-4 h-4 text-cricket-green bg-dark-card border-dark-border rounded focus:ring-cricket-green"
                />
                <div className="flex items-center">
                  <Trophy className="h-4 w-4 mr-1 text-dark-muted" />
                  <span className="text-sm text-dark-text">IPL</span>
                </div>
              </label>

              {/* Bilateral Series */}
              <label className="flex items-center space-x-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={context.isBilateral}
                  onChange={(e) => handleChange('isBilateral', e.target.checked)}
                  className="w-4 h-4 text-cricket-green bg-dark-card border-dark-border rounded focus:ring-cricket-green"
                />
                <div className="flex items-center">
                  <Home className="h-4 w-4 mr-1 text-dark-muted" />
                  <span className="text-sm text-dark-text">Bilateral Series</span>
                </div>
              </label>

              {/* Important Match */}
              <label className="flex items-center space-x-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={context.isImportantMatch}
                  onChange={(e) => handleChange('isImportantMatch', e.target.checked)}
                  className="w-4 h-4 text-cricket-green bg-dark-card border-dark-border rounded focus:ring-cricket-green"
                />
                <div className="flex items-center">
                  <Zap className="h-4 w-4 mr-1 text-dark-muted" />
                  <span className="text-sm text-dark-text">Important Match</span>
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
            {context.tournamentType && <div>🏆 Tournament: {context.tournamentType.replace('_', ' ').toUpperCase()}</div>}
            {context.tossWinner && <div>🏆 Toss Winner: {context.tossWinner.team_name}</div>}
            {context.tossDecision && <div>⚡ Decision: {context.tossDecision}</div>}
            {context.battingFirst && <div>🏏 Batting First: {context.battingFirst.team_name}</div>}
            <div>📅 Season: {context.seasonYear} - {['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][context.seasonMonth-1]}</div>
            {context.isFinal && <div>🏆 Final Match</div>}
            {context.isSemiFinal && <div>🏆 Semi Final</div>}
            {context.isPlayoff && <div>🏆 Playoff</div>}
            {context.isT20WorldCup && <div>🌍 T20 World Cup</div>}
            {context.isIPL && <div>🏏 IPL</div>}
            {context.isBilateral && <div>🤝 Bilateral Series</div>}
            {context.isImportantMatch && <div>⭐ Important Match</div>}
            {context.isHomeTeam && <div>🏠 Home Advantage</div>}
            {context.isWinter && <div>❄️ Winter Season</div>}
            {context.isSummer && <div>☀️ Summer Season</div>}
            {context.isMonsoon && <div>🌧️ Monsoon Season</div>}
          </div>
        </motion.div>
      )}
    </motion.div>
  );
};

export default MatchContext;
