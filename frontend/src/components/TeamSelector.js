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
    tiers: [],
    hasImpact: null  // null = all, true = only with impact, false = only without
  });

  const handleTeamChange = (e) => {
    const teamId = parseInt(e.target.value);
    const selectedTeam = teams.find(t => t.team_id === teamId);
    if (selectedTeam) {
      onTeamSelect(teamType, teamId, selectedTeam.team_name);
    }
  };

  const handlePlayerAdd = (playerId, playerName, playerCountry) => {
    if (team.players.length < 11) {
      onPlayerSelect(teamType, playerId, playerName, playerCountry);
      // DON'T close dropdown - keep it open for adding more players
      // setShowPlayerDropdown(false);
      // Clear search to show all again
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
      tiers: [],
      hasImpact: null
    });
  };

  // Get unique values for filter options
  const uniqueCountries = [...new Set(players.map(p => p.country).filter(c => c && c !== 'Unknown'))].sort();
  const uniqueRoles = [...new Set(players.map(p => p.role || p.player_role).filter(Boolean))].sort();
  const uniqueTiers = ['elite', 'star', 'good', 'regular'];

  const filteredPlayers = players.filter(player => {
    const query = searchQuery.toLowerCase();
    const name = (player.player_name || player.name || '').toLowerCase();
    const country = (player.country || '').toLowerCase();
    const role = (player.role || player.player_role || '').toLowerCase();
    
    // Text search - matches name, country, or role
    const matchesText = query === '' || 
      name.includes(query) || 
      country.includes(query) || 
      role.includes(query);
    
    // Filter by countries
    const matchesCountryFilter = filters.countries.length === 0 || 
      filters.countries.includes(player.country);
    
    // Filter by roles
    const matchesRoleFilter = filters.roles.length === 0 || 
      filters.roles.includes(player.role) ||
      filters.roles.includes(player.player_role);
    
    // Filter by tier (elite/star/good/regular)
    const matchesTierFilter = filters.tiers.length === 0 ||
      filters.tiers.includes(player.tier);
    
    // Filter by has impact
    const matchesImpactFilter = filters.hasImpact === null ||
      (filters.hasImpact === true && player.has_impact) ||
      (filters.hasImpact === false && !player.has_impact);
    
    // Not already selected
    const notSelected = !team.players.some(p => p.id === player.player_id || p.id === player.id);
    
    return matchesText && matchesCountryFilter && matchesRoleFilter && 
           matchesTierFilter && matchesImpactFilter && notSelected;
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
                    {(filters.countries.length > 0 || filters.roles.length > 0 || filters.tiers.length > 0 || filters.hasImpact !== null) && (
                      <button
                        onClick={clearFilters}
                        className="text-xs text-cricket-red hover:text-red-400 transition-colors"
                      >
                        Clear Filters
                      </button>
                    )}
                    <span className="text-xs text-dark-muted">
                      {filteredPlayers.length} players
                    </span>
                  </div>

                  {/* Filter Options */}
                  {showFilters && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      exit={{ opacity: 0, height: 0 }}
                      className="mb-4 p-3 bg-dark-card border border-dark-border rounded-lg"
                    >
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        {/* Country Filter */}
                        <div>
                          <label className="block text-xs font-medium text-dark-muted mb-2">
                            Country ({uniqueCountries.length})
                          </label>
                          <div className="max-h-32 overflow-y-auto space-y-1 pr-2">
                            {uniqueCountries.map(country => (
                              <label key={country} className="flex items-center gap-2 text-xs hover:bg-dark-border/30 p-1 rounded">
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
                          <label className="block text-xs font-medium text-dark-muted mb-2">
                            Role
                          </label>
                          <div className="space-y-1">
                            {uniqueRoles.map(role => (
                              <label key={role} className="flex items-center gap-2 text-xs hover:bg-dark-border/30 p-1 rounded">
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

                        {/* Tier & Impact Filter */}
                        <div>
                          <label className="block text-xs font-medium text-dark-muted mb-2">
                            Player Quality
                          </label>
                          <div className="space-y-1">
                            <label className="flex items-center gap-2 text-xs hover:bg-dark-border/30 p-1 rounded">
                              <input
                                type="radio"
                                checked={filters.hasImpact === true}
                                onChange={() => setFilters(prev => ({...prev, hasImpact: true}))}
                                className="rounded border-dark-border"
                              />
                              <span className="text-cricket-green">⭐ Stars Only (with impact)</span>
                            </label>
                            <label className="flex items-center gap-2 text-xs hover:bg-dark-border/30 p-1 rounded">
                              <input
                                type="radio"
                                checked={filters.hasImpact === null}
                                onChange={() => setFilters(prev => ({...prev, hasImpact: null}))}
                                className="rounded border-dark-border"
                              />
                              <span className="text-dark-text">All Players</span>
                            </label>
                            
                            <div className="mt-2 pt-2 border-t border-dark-border">
                              <label className="block text-xs font-medium text-dark-muted mb-1">Tier</label>
                              {uniqueTiers.map(tier => (
                                <label key={tier} className="flex items-center gap-2 text-xs hover:bg-dark-border/30 p-1 rounded">
                                  <input
                                    type="checkbox"
                                    checked={filters.tiers.includes(tier)}
                                    onChange={() => handleFilterChange('tiers', tier)}
                                    className="rounded border-dark-border"
                                  />
                                  <span className="text-dark-text capitalize">
                                    {tier} {tier === 'elite' ? '(+15-20)' : tier === 'star' ? '(+8-12)' : tier === 'good' ? '(+3-8)' : '(0)'}
                                  </span>
                                </label>
                              ))}
                            </div>
                          </div>
                        </div>
                      </div>
                    </motion.div>
                  )}
                  
                  <div className="mb-2 flex items-center justify-between text-sm text-dark-muted">
                    <span>
                      Showing {Math.min(50, filteredPlayers.length)} of {filteredPlayers.length} players
                      {(filters.countries.length > 0 || filters.roles.length > 0 || filters.tiers.length > 0 || filters.hasImpact !== null) && (
                        <span className="ml-2 text-cricket-green">(filtered)</span>
                      )}
                    </span>
                    {team.players.length >= 11 && (
                      <span className="text-cricket-gold font-medium">Team Full ✓</span>
                    )}
                  </div>
                  <div className="max-h-64 overflow-y-auto border border-dark-border rounded-lg bg-dark-card">
                    {filteredPlayers.slice(0, 100).map(player => {
                      const battingImpact = player.batting_impact || 0;
                      const bowlingImpact = player.bowling_impact || 0;
                      const totalImpact = battingImpact + bowlingImpact;
                      const hasImpact = player.has_impact || Math.abs(totalImpact) > 0.5;
                      
                      return (
                        <motion.button
                          key={player.player_id || player.id}
                          whileHover={{ backgroundColor: '#00C85120' }}
                          onClick={() => handlePlayerAdd(
                            player.player_id || player.id, 
                            player.player_name || player.name, 
                            player.country
                          )}
                          className="w-full text-left p-3 hover:bg-cricket-green/10 border-b border-dark-border last:border-b-0 transition-colors"
                        >
                          <div className="flex items-center justify-between">
                            <div className="flex-1">
                              <div className="font-medium text-dark-text">
                                {player.player_name || player.name}
                                {hasImpact && (
                                  <span className="ml-2 text-xs">
                                    {totalImpact > 12 ? '⭐⭐' : totalImpact > 5 || totalImpact < -5 ? '⭐' : ''}
                                  </span>
                                )}
                              </div>
                              <div className="text-xs text-dark-muted flex items-center gap-2">
                                <span>{player.country}</span>
                                {player.role && <span>• {player.role}</span>}
                                {player.batting_avg > 0 && (
                                  <span>• Avg: {player.batting_avg.toFixed(1)}</span>
                                )}
                              </div>
                            </div>
                            {hasImpact && (
                              <div className={`text-xs font-semibold ml-2 ${
                                totalImpact > 0 ? 'text-cricket-green' : 'text-red-400'
                              }`}>
                                {totalImpact > 0 ? '+' : ''}{totalImpact.toFixed(1)}
                              </div>
                            )}
                          </div>
                        </motion.button>
                      );
                    })}
                    {filteredPlayers.length === 0 && (
                      <div className="p-6 text-center text-dark-muted">
                        No players found. Try adjusting your filters or search.
                      </div>
                    )}
                    {filteredPlayers.length > 100 && (
                      <div className="p-3 text-center text-sm text-dark-muted border-t border-dark-border bg-dark-bg">
                        Showing first 100 of {filteredPlayers.length} results. 
                        <br />Use filters or search to narrow down.
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
