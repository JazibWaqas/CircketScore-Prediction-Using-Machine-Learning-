import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Users, Plus, X, Search, Settings } from 'lucide-react';

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
  const [showFilters, setShowFilters] = useState(false);
  const [filters, setFilters] = useState({
    countries: [],
    roles: [],
    battingStyles: [],
    bowlingStyles: []
  });

  const handleTeamChange = (e) => {
    const teamId = parseInt(e.target.value);
    const selectedTeam = teams.find(t => t.team_id === teamId);
    if (selectedTeam) {
      onTeamSelect(teamType, teamId, selectedTeam.team_name);
    }
  };

  const handlePlayerAdd = (playerId, playerName, playerCountry, playerRole) => {
    if (team.players.length < 11) {
      onPlayerSelect(teamType, playerId, playerName, playerCountry, playerRole);
      setShowPlayerDropdown(false);
      setSearchQuery('');
    }
  };

  const handleFilterChange = (filterType, value) => {
    setFilters(prev => ({
      ...prev,
      [filterType]: prev[filterType].includes(value)
        ? prev[filterType].filter(item => item !== value)
        : [...prev[filterType], value]
    }));
  };

  const clearFilters = () => {
    setFilters({
      countries: [],
      roles: [],
      battingStyles: [],
      bowlingStyles: []
    });
  };

  // Get unique values for filter options
  const uniqueCountries = [...new Set(players.map(p => p.country).filter(Boolean))].sort();
  const uniqueRoles = [...new Set(players.map(p => p.player_role).filter(Boolean))].sort();
  const uniqueBattingStyles = [...new Set(players.map(p => p.batting_style).filter(Boolean))].sort();
  const uniqueBowlingStyles = [...new Set(players.map(p => p.bowling_style).filter(Boolean))].sort();

  const filteredPlayers = players.filter(player => {
    const query = searchQuery.toLowerCase();
    const name = player.player_name.toLowerCase();
    const country = (player.country || '').toLowerCase();
    const role = (player.player_role || '').toLowerCase();
    const battingStyle = (player.batting_style || '').toLowerCase();
    const bowlingStyle = (player.bowling_style || '').toLowerCase();
    
    // Text search - matches any part of name, country, or role
    const matchesText = query === '' || 
      name.includes(query) || 
      country.includes(query) || 
      role.includes(query) ||
      battingStyle.includes(query) ||
      bowlingStyle.includes(query);
    
    // Filter by countries
    const matchesCountryFilter = filters.countries.length === 0 || 
      filters.countries.includes(player.country);
    
    // Filter by roles
    const matchesRoleFilter = filters.roles.length === 0 || 
      filters.roles.includes(player.player_role);
    
    // Filter by batting styles
    const matchesBattingFilter = filters.battingStyles.length === 0 || 
      filters.battingStyles.includes(player.batting_style);
    
    // Filter by bowling styles
    const matchesBowlingFilter = filters.bowlingStyles.length === 0 || 
      filters.bowlingStyles.includes(player.bowling_style);
    
    return matchesText && matchesCountryFilter && matchesRoleFilter && 
           matchesBattingFilter && matchesBowlingFilter &&
           !team.players.some(p => p.id === player.player_id);
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
          Team {teamType}
        </h3>
      </div>

      {/* Team Selection */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-dark-muted mb-2">
          Select Team
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
                      placeholder="Search by name, country, or role (e.g., 'Wasim', 'Pakistan', 'Bowler')..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="cricket-input flex-1 text-sm"
                    />
                  </div>
                  
                  {/* Filter Toggle Button */}
                  <div className="flex items-center justify-between mb-2">
                    <button
                      onClick={() => setShowFilters(!showFilters)}
                      className="flex items-center gap-2 text-sm text-cricket-green hover:text-green-400 transition-colors"
                    >
                      <Settings className="h-4 w-4" />
                      Filters {showFilters ? '▼' : '▶'}
                    </button>
                    {(filters.countries.length > 0 || filters.roles.length > 0 || filters.battingStyles.length > 0 || filters.bowlingStyles.length > 0) && (
                      <button
                        onClick={clearFilters}
                        className="text-xs text-cricket-red hover:text-red-400 transition-colors"
                      >
                        Clear Filters
                      </button>
                    )}
                  </div>

                  {/* Filter Options */}
                  {showFilters && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      exit={{ opacity: 0, height: 0 }}
                      className="mb-4 p-3 bg-dark-card border border-dark-border rounded-lg"
                    >
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {/* Country Filter */}
                        <div>
                          <label className="block text-xs font-medium text-dark-muted mb-2">Country</label>
                          <div className="max-h-24 overflow-y-auto space-y-1">
                            {uniqueCountries.slice(0, 10).map(country => (
                              <label key={country} className="flex items-center gap-2 text-xs">
                                <input
                                  type="checkbox"
                                  checked={filters.countries.includes(country)}
                                  onChange={() => handleFilterChange('countries', country)}
                                  className="rounded border-dark-border"
                                />
                                <span className="text-dark-text">{country}</span>
                              </label>
                            ))}
                          </div>
                        </div>

                        {/* Role Filter */}
                        <div>
                          <label className="block text-xs font-medium text-dark-muted mb-2">Role</label>
                          <div className="space-y-1">
                            {uniqueRoles.map(role => (
                              <label key={role} className="flex items-center gap-2 text-xs">
                                <input
                                  type="checkbox"
                                  checked={filters.roles.includes(role)}
                                  onChange={() => handleFilterChange('roles', role)}
                                  className="rounded border-dark-border"
                                />
                                <span className="text-dark-text">{role}</span>
                              </label>
                            ))}
                          </div>
                        </div>

                        {/* Batting Style Filter */}
                        <div>
                          <label className="block text-xs font-medium text-dark-muted mb-2">Batting Style</label>
                          <div className="space-y-1">
                            {uniqueBattingStyles.slice(0, 5).map(style => (
                              <label key={style} className="flex items-center gap-2 text-xs">
                                <input
                                  type="checkbox"
                                  checked={filters.battingStyles.includes(style)}
                                  onChange={() => handleFilterChange('battingStyles', style)}
                                  className="rounded border-dark-border"
                                />
                                <span className="text-dark-text">{style}</span>
                              </label>
                            ))}
                          </div>
                        </div>

                        {/* Bowling Style Filter */}
                        <div>
                          <label className="block text-xs font-medium text-dark-muted mb-2">Bowling Style</label>
                          <div className="space-y-1">
                            {uniqueBowlingStyles.slice(0, 5).map(style => (
                              <label key={style} className="flex items-center gap-2 text-xs">
                                <input
                                  type="checkbox"
                                  checked={filters.bowlingStyles.includes(style)}
                                  onChange={() => handleFilterChange('bowlingStyles', style)}
                                  className="rounded border-dark-border"
                                />
                                <span className="text-dark-text">{style}</span>
                              </label>
                            ))}
                          </div>
                        </div>
                      </div>
                    </motion.div>
                  )}
                  
                  <div className="mb-2 text-sm text-dark-muted">
                    {filteredPlayers.length} players found
                    {(filters.countries.length > 0 || filters.roles.length > 0 || filters.battingStyles.length > 0 || filters.bowlingStyles.length > 0) && (
                      <span className="ml-2 text-cricket-green">
                        (Filtered)
                      </span>
                    )}
                  </div>
                  <div className="max-h-48 overflow-y-auto border border-dark-border rounded-lg bg-dark-card">
                    {filteredPlayers.slice(0, 100).map(player => (
                      <motion.button
                        key={player.player_id}
                        whileHover={{ backgroundColor: '#00C85120' }}
                        onClick={() => onPlayerSelect(teamType, player.player_id, player.player_name, player.country, player.player_role)}
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
                    {filteredPlayers.length > 100 && (
                      <div className="p-3 text-center text-sm text-dark-muted border-t border-dark-border">
                        Showing first 100 of {filteredPlayers.length} results. 
                        <br />Try a more specific search to narrow down results.
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
                        {player.country} • {player.role}
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
