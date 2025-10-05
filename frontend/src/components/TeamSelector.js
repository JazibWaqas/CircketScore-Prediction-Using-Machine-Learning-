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

  const handlePlayerAdd = (playerId, playerName) => {
    if (team.players.length < 11) {
      onPlayerSelect(teamType, playerId, playerName);
      setShowPlayerDropdown(false);
      setSearchQuery('');
    }
  };

  const filteredPlayers = players.filter(player =>
    player.player_name.toLowerCase().includes(searchQuery.toLowerCase()) &&
    !team.players.some(p => p.id === player.player_id)
  );

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
          Team {teamType}
        </h3>
      </div>

      {/* Team Selection */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-dark-muted mb-2">
          Select Team
        </label>
        <select
          value={team.id || ''}
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
      {team.name && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-4 p-3 bg-cricket-green/10 border border-cricket-green/30 rounded-lg"
        >
          <div className="text-cricket-green font-semibold">{team.name}</div>
          <div className="text-sm text-dark-muted">
            {team.players.length}/11 players selected
          </div>
        </motion.div>
      )}

      {/* Player Selection */}
      {team.name && (
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
                      placeholder="Search players..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="cricket-input flex-1 text-sm"
                    />
                  </div>
                  
                  <div className="max-h-48 overflow-y-auto border border-dark-border rounded-lg bg-dark-card">
                    {filteredPlayers.slice(0, 20).map(player => (
                      <motion.button
                        key={player.player_id}
                        whileHover={{ backgroundColor: '#00C85120' }}
                        onClick={() => handlePlayerAdd(player.player_id, player.player_name)}
                        className="w-full text-left p-3 hover:bg-cricket-green/10 border-b border-dark-border last:border-b-0 transition-colors"
                      >
                        <div className="font-medium text-dark-text">
                          {player.player_name}
                        </div>
                        <div className="text-sm text-dark-muted">
                          {player.country} • {player.player_role}
                        </div>
                      </motion.button>
                    ))}
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

      {/* Team Stats Preview */}
      {team.players.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-4 p-3 bg-dark-card border border-dark-border rounded-lg"
        >
          <div className="text-sm text-dark-muted mb-2">Team Composition</div>
          <div className="flex flex-wrap gap-2">
            {['Batsman', 'Bowler', 'All-rounder', 'Wicket-keeper'].map(role => {
              const count = team.players.filter(p => 
                players.find(player => player.player_id === p.id)?.player_role === role
              ).length;
              return (
                <span
                  key={role}
                  className="px-2 py-1 bg-cricket-green/20 text-cricket-green text-xs rounded-full"
                >
                  {role}: {count}
                </span>
              );
            })}
          </div>
        </motion.div>
      )}
    </motion.div>
  );
};

export default TeamSelector;
