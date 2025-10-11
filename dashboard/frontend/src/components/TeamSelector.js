import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Users, Plus, X, Search } from 'lucide-react';

const TeamSelector = ({ 
  teamType, 
  team, 
  teams, 
  players, 
  onTeamSelect, 
  onPlayerSelect, 
  onRemovePlayer 
}) => {
  const [showPlayerDropdown, setShowPlayerDropdown] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  const handleTeamChange = (e) => {
    const teamId = parseInt(e.target.value);
    const selectedTeam = teams.find(t => t.team_id === teamId);
    if (selectedTeam) {
      onTeamSelect(teamType, teamId, selectedTeam.team_name);
    }
  };

  const handlePlayerAdd = (player) => {
    if (team.players.length < 11) {
      onPlayerSelect(teamType, player.player_id, player.player_name, player.country);
      setSearchQuery('');
    }
  };

  const filteredPlayers = players.filter(player => {
    const query = searchQuery.toLowerCase();
    const name = (player.player_name || '').toLowerCase();
    const country = (player.country || '').toLowerCase();
    
    // Not already selected
    const notSelected = !team.players.some(p => p.id === player.player_id);
    
    return (name.includes(query) || country.includes(query)) && notSelected;
  });

  return (
    <motion.div
      initial={{ opacity: 0, x: teamType === 'A' ? -20 : 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.5 }}
      className="team-card"
    >
      <div className="flex items-center mb-4">
        <Users className="h-6 w-6 text-cricket-green mr-2" />
        <h3 className="text-xl font-semibold text-cricket-green">
          Team {teamType} {teamType === 'A' ? '(Batting)' : '(Opposition)'}
        </h3>
      </div>

      {/* Team Selection */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-dark-muted mb-2">
          Select Team/Country
        </label>
        <select
          value={team.team_id || ''}
          onChange={handleTeamChange}
          className="cricket-select w-full"
        >
          <option value="">Choose a team...</option>
          {teams.map(teamOption => (
            <option key={teamOption.team_id} value={teamOption.team_id}>
              {teamOption.team_name}
            </option>
          ))}
        </select>
      </div>

      {/* Team Name Display */}
      {team.team_name && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-4 p-3 bg-cricket-green/10 border border-cricket-green/30 rounded-lg"
        >
          <div className="text-cricket-green font-semibold">{team.team_name}</div>
          <div className="text-sm text-dark-muted">
            {team.players.length}/11 players selected
          </div>
        </motion.div>
      )}

      {/* Player Selection */}
      {team.team_name && (
        <div className="mb-4">
          <div className="flex items-center justify-between mb-3">
            <label className="text-sm font-medium text-dark-muted">
              Players ({team.players.length}/11)
            </label>
            {team.players.length < 11 && (
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setShowPlayerDropdown(!showPlayerDropdown)}
                className="flex items-center gap-2 text-cricket-green hover:text-green-400 transition-colors"
              >
                <Plus className="h-4 w-4" />
                Add Player
              </motion.button>
            )}
          </div>

          {/* Player Dropdown */}
          <AnimatePresence>
            {showPlayerDropdown && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="mb-4"
              >
                <div className="relative">
                  <div className="flex items-center gap-2 mb-2">
                    <Search className="h-4 w-4 text-dark-muted" />
                    <input
                      type="text"
                      placeholder="Search by name or country..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="cricket-input flex-1 text-sm"
                    />
                  </div>
                  
                  <div className="mb-2 text-sm text-dark-muted">
                    Showing {Math.min(50, filteredPlayers.length)} of {filteredPlayers.length} players
                  </div>
                  
                  <div className="max-h-64 overflow-y-auto border border-dark-border rounded-lg bg-dark-card">
                    {filteredPlayers.slice(0, 50).map(player => (
                      <motion.button
                        key={player.player_id}
                        whileHover={{ backgroundColor: '#00C85120' }}
                        onClick={() => handlePlayerAdd(player)}
                        className="w-full text-left p-3 hover:bg-cricket-green/10 border-b border-dark-border last:border-b-0 transition-colors"
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex-1">
                            <div className="font-medium text-dark-text">
                              {player.player_name}
                            </div>
                            <div className="text-xs text-dark-muted flex items-center gap-2">
                              <span>{player.country}</span>
                              {player.player_role && <span>• {player.player_role}</span>}
                              {player.batting_avg > 0 && (
                                <span>• Avg: {player.batting_avg.toFixed(1)}</span>
                              )}
                            </div>
                          </div>
                        </div>
                      </motion.button>
                    ))}
                    {filteredPlayers.length === 0 && (
                      <div className="p-6 text-center text-dark-muted">
                        No players found. Try adjusting your search.
                      </div>
                    )}
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Selected Players */}
          <div className="space-y-2">
            <AnimatePresence>
              {team.players.map((player, index) => (
                <motion.div
                  key={player.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                  className="flex items-center justify-between p-3 bg-dark-card border border-dark-border rounded-lg"
                >
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 bg-cricket-green/20 rounded-full flex items-center justify-center text-sm font-semibold text-cricket-green">
                      {index + 1}
                    </div>
                    <div>
                      <div className="font-medium text-dark-text">
                        {player.name}
                      </div>
                      <div className="text-sm text-dark-muted">
                        {player.country}
                      </div>
                    </div>
                  </div>
                  <motion.button
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                    onClick={() => onRemovePlayer(teamType, player.id)}
                    className="text-cricket-red hover:text-red-400 transition-colors"
                  >
                    <X className="h-4 w-4" />
                  </motion.button>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        </div>
      )}
    </motion.div>
  );
};

export default TeamSelector;
