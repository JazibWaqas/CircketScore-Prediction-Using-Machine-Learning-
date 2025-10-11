import React from 'react';
import { motion } from 'framer-motion';

const MatchScenario = ({ scenario, onChange, venues, battingPlayers }) => {
  const handleChange = (field, value) => {
    onChange({ ...scenario, [field]: value });
  };
  
  const handleVenueChange = (venueName) => {
    const selectedVenue = venues.find(v => v.venue_name === venueName);
    onChange({
      ...scenario,
      venue: venueName,
      venue_avg_score: selectedVenue && selectedVenue.avg_score ? selectedVenue.avg_score : 250
    });
  };
  
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="cricket-card"
    >
      <h2 className="text-2xl font-bold text-cricket-green mb-6">Match Scenario</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {/* Venue */}
        <div>
          <label className="block text-dark-muted mb-2">Venue</label>
          <select
            value={scenario.venue}
            onChange={(e) => handleVenueChange(e.target.value)}
            className="cricket-select w-full"
          >
            <option value="">Select venue...</option>
            {venues.map(venue => (
              <option key={venue.venue_name} value={venue.venue_name}>
                {venue.venue_name} {venue.avg_score ? `(Avg: ${venue.avg_score.toFixed(0)})` : ''}
              </option>
            ))}
          </select>
        </div>
        
        {/* Current Score */}
        <div>
          <label className="block text-dark-muted mb-2">Current Score</label>
          <input
            type="number"
            value={scenario.current_score}
            onChange={(e) => handleChange('current_score', parseInt(e.target.value) || 0)}
            className="cricket-input w-full"
            min="0"
            max="500"
          />
        </div>
        
        {/* Wickets */}
        <div>
          <label className="block text-dark-muted mb-2">Wickets Fallen</label>
          <input
            type="number"
            value={scenario.wickets_fallen}
            onChange={(e) => handleChange('wickets_fallen', parseInt(e.target.value) || 0)}
            className="cricket-input w-full"
            min="0"
            max="10"
          />
        </div>
        
        {/* Overs */}
        <div>
          <label className="block text-dark-muted mb-2">Overs Completed</label>
          <input
            type="number"
            value={scenario.overs}
            onChange={(e) => handleChange('overs', parseInt(e.target.value) || 0)}
            className="cricket-input w-full"
            min="0"
            max="50"
            step="1"
          />
        </div>
        
        {/* Runs in Last 10 Overs */}
        <div>
          <label className="block text-dark-muted mb-2">Runs in Last 10 Overs</label>
          <input
            type="number"
            value={scenario.runs_last_10}
            onChange={(e) => handleChange('runs_last_10', parseInt(e.target.value) || 0)}
            className="cricket-input w-full"
            min="0"
            max="200"
          />
        </div>
        
        {/* Current Batsman 1 */}
        <div>
          <label className="block text-dark-muted mb-2">Current Batsman 1 (optional)</label>
          <select
            value={scenario.batsman_1}
            onChange={(e) => handleChange('batsman_1', e.target.value)}
            className="cricket-select w-full"
          >
            <option value="">None</option>
            {battingPlayers.length > 0 && battingPlayers.map(player => (
              <option key={player.id} value={player.name}>
                {player.name}
              </option>
            ))}
          </select>
        </div>
        
        {/* Current Batsman 2 */}
        <div>
          <label className="block text-dark-muted mb-2">Current Batsman 2 (optional)</label>
          <select
            value={scenario.batsman_2}
            onChange={(e) => handleChange('batsman_2', e.target.value)}
            className="cricket-select w-full"
          >
            <option value="">None</option>
            {battingPlayers.length > 0 && battingPlayers.map(player => (
              <option key={player.id} value={player.name}>
                {player.name}
              </option>
            ))}
          </select>
        </div>
      </div>
      
      {/* Quick Scenarios */}
      <div className="mt-6 pt-6 border-t border-dark-border">
        <p className="text-dark-muted mb-3">Quick Scenarios:</p>
        <div className="flex flex-wrap gap-3">
          <button
            onClick={() => onChange({
              ...scenario,
              current_score: 0,
              wickets_fallen: 0,
              overs: 0,
              runs_last_10: 0
            })}
            className="px-4 py-2 bg-dark-card border border-dark-border rounded-lg hover:border-cricket-green transition-colors text-sm"
          >
            Pre-Match
          </button>
          <button
            onClick={() => onChange({
              ...scenario,
              current_score: 55,
              wickets_fallen: 1,
              overs: 10,
              runs_last_10: 55
            })}
            className="px-4 py-2 bg-dark-card border border-dark-border rounded-lg hover:border-cricket-green transition-colors text-sm"
          >
            After 10 Overs
          </button>
          <button
            onClick={() => onChange({
              ...scenario,
              current_score: 115,
              wickets_fallen: 2,
              overs: 20,
              runs_last_10: 60
            })}
            className="px-4 py-2 bg-dark-card border border-dark-border rounded-lg hover:border-cricket-green transition-colors text-sm"
          >
            After 20 Overs
          </button>
          <button
            onClick={() => onChange({
              ...scenario,
              current_score: 180,
              wickets_fallen: 3,
              overs: 30,
              runs_last_10: 65
            })}
            className="px-4 py-2 bg-dark-card border border-dark-border rounded-lg hover:border-cricket-green transition-colors text-sm"
          >
            After 30 Overs
          </button>
          <button
            onClick={() => onChange({
              ...scenario,
              current_score: 250,
              wickets_fallen: 5,
              overs: 40,
              runs_last_10: 70
            })}
            className="px-4 py-2 bg-dark-card border border-dark-border rounded-lg hover:border-cricket-green transition-colors text-sm"
          >
            After 40 Overs
          </button>
        </div>
      </div>
    </motion.div>
  );
};

export default MatchScenario;

