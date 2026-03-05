import React from 'react';
import { motion } from 'framer-motion';
import { Users, TrendingUp, Shield, Award } from 'lucide-react';

const TeamFormationDisplay = ({ team, teamType, players }) => {
    if (!team.team_name || team.players.length === 0) {
        return null;
    }

    // Get full player data with stats
    const teamPlayersWithStats = team.players.map(selectedPlayer => {
        const fullPlayerData = players.find(p => p.player_id === selectedPlayer.id);
        return {
            ...selectedPlayer,
            role: fullPlayerData?.player_role || 'All-rounder',
            batting_avg: fullPlayerData?.batting_avg || 0,
            bowling_economy: fullPlayerData?.bowling_economy || 0
        };
    });

    // Categorize players
    const batsmen = teamPlayersWithStats.filter(p => p.role === 'Batsman');
    const allRounders = teamPlayersWithStats.filter(p => p.role === 'All-rounder');
    const bowlers = teamPlayersWithStats.filter(p => p.role === 'Bowler');

    // Calculate team stats
    const avgBattingAvg = teamPlayersWithStats.reduce((sum, p) => sum + (p.batting_avg || 0), 0) / teamPlayersWithStats.length;
    const eliteBatsmen = teamPlayersWithStats.filter(p => p.batting_avg >= 40).length;
    const battingDepth = teamPlayersWithStats.filter(p => p.batting_avg >= 30).length;

    const bowlingPlayers = teamPlayersWithStats.filter(p => p.bowling_economy > 0);
    const avgBowlingEconomy = bowlingPlayers.length > 0
        ? bowlingPlayers.reduce((sum, p) => sum + p.bowling_economy, 0) / bowlingPlayers.length
        : 0;
    const eliteBowlers = teamPlayersWithStats.filter(p => p.bowling_economy > 0 && p.bowling_economy < 4.8).length;

    // Role icons
    const getRoleIcon = (role) => {
        switch (role) {
            case 'Batsman': return '🏏';
            case 'Bowler': return '🎳';
            case 'All-rounder': return '⚖️';
            default: return '👤';
        }
    };

    // Strength bars
    const battingStrength = Math.max(0, Math.min(10, Math.round((avgBattingAvg / 50) * 10)));
    const bowlingStrength = avgBowlingEconomy > 0 ? Math.max(0, Math.min(10, Math.round((6 - avgBowlingEconomy) * 2))) : 5;
    const depthStrength = Math.max(0, Math.min(10, battingDepth));

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="cricket-card"
        >
            <div className="flex items-center gap-2 mb-6">
                <Users className="h-6 w-6 text-cricket-green" />
                <h3 className="text-xl font-bold text-cricket-green">
                    {team.team_name} - Team Formation
                </h3>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Batting Order */}
                <div>
                    <div className="flex items-center gap-2 mb-4">
                        <TrendingUp className="h-5 w-5 text-cricket-green" />
                        <h4 className="font-semibold text-dark-text">Batting Order</h4>
                    </div>

                    <div className="space-y-2">
                        {teamPlayersWithStats.map((player, idx) => (
                            <motion.div
                                key={player.id}
                                initial={{ opacity: 0, x: -10 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: idx * 0.05 }}
                                className="flex items-center gap-3 p-2 bg-dark-card border border-dark-border rounded-lg hover:border-cricket-green/50 transition-colors"
                            >
                                <div className="w-8 h-8 bg-cricket-green/20 rounded-full flex items-center justify-center text-sm font-bold text-cricket-green">
                                    {idx + 1}
                                </div>
                                <div className="flex-1">
                                    <div className="flex items-center gap-2">
                                        <span className="text-sm">{getRoleIcon(player.role)}</span>
                                        <span className="font-medium text-dark-text">{player.name}</span>
                                    </div>
                                    <div className="text-xs text-dark-muted">
                                        {player.batting_avg > 0 && `Avg: ${player.batting_avg.toFixed(1)}`}
                                        {player.batting_avg > 0 && player.bowling_economy > 0 && ' | '}
                                        {player.bowling_economy > 0 && `Econ: ${player.bowling_economy.toFixed(1)}`}
                                    </div>
                                </div>
                                {player.batting_avg >= 40 && (
                                    <Award className="h-4 w-4 text-yellow-500" title="Elite Batsman" />
                                )}
                            </motion.div>
                        ))}
                    </div>
                </div>

                {/* Team Stats & Composition */}
                <div>
                    <div className="flex items-center gap-2 mb-4">
                        <Shield className="h-5 w-5 text-cricket-green" />
                        <h4 className="font-semibold text-dark-text">Team Analysis</h4>
                    </div>

                    {/* Composition */}
                    <div className="mb-6 p-4 bg-dark-card border border-dark-border rounded-lg">
                        <div className="text-sm font-semibold text-dark-muted mb-3">Team Composition</div>
                        <div className="space-y-2">
                            <div className="flex items-center justify-between">
                                <span className="text-sm text-dark-text">🏏 Batsmen</span>
                                <span className="font-semibold text-cricket-green">{batsmen.length}</span>
                            </div>
                            <div className="flex items-center justify-between">
                                <span className="text-sm text-dark-text">⚖️ All-rounders</span>
                                <span className="font-semibold text-cricket-green">{allRounders.length}</span>
                            </div>
                            <div className="flex items-center justify-between">
                                <span className="text-sm text-dark-text">🎳 Bowlers</span>
                                <span className="font-semibold text-cricket-green">{bowlers.length}</span>
                            </div>
                        </div>

                        {/* Composition Warning */}
                        {bowlers.length + allRounders.length < 5 && (
                            <div className="mt-3 p-2 bg-yellow-900/20 border border-yellow-500/50 rounded text-xs text-yellow-400">
                                ⚠️ Consider adding more bowling options
                            </div>
                        )}
                        {batsmen.length + allRounders.length < 6 && (
                            <div className="mt-3 p-2 bg-yellow-900/20 border border-yellow-500/50 rounded text-xs text-yellow-400">
                                ⚠️ Batting lineup looks thin
                            </div>
                        )}
                        {bowlers.length + allRounders.length >= 5 && batsmen.length + allRounders.length >= 6 && (
                            <div className="mt-3 p-2 bg-cricket-green/20 border border-cricket-green/50 rounded text-xs text-cricket-green">
                                ✓ Well-balanced team composition
                            </div>
                        )}
                    </div>

                    {/* Team Strength Metrics */}
                    <div className="space-y-4">
                        <div>
                            <div className="flex items-center justify-between text-sm mb-2">
                                <span className="text-dark-muted">Batting Strength</span>
                                <span className="text-cricket-green font-semibold">{battingStrength}/10</span>
                            </div>
                            <div className="h-2 bg-dark-card rounded-full overflow-hidden border border-dark-border">
                                <div
                                    className="h-full bg-gradient-to-r from-cricket-green to-green-400 transition-all"
                                    style={{ width: `${battingStrength * 10}%` }}
                                />
                            </div>
                            <div className="text-xs text-dark-muted mt-1">
                                Avg: {avgBattingAvg.toFixed(1)} | Elite: {eliteBatsmen} | Depth: {battingDepth}
                            </div>
                        </div>

                        <div>
                            <div className="flex items-center justify-between text-sm mb-2">
                                <span className="text-dark-muted">Bowling Strength</span>
                                <span className="text-cricket-green font-semibold">{bowlingStrength}/10</span>
                            </div>
                            <div className="h-2 bg-dark-card rounded-full overflow-hidden border border-dark-border">
                                <div
                                    className="h-full bg-gradient-to-r from-blue-500 to-blue-400 transition-all"
                                    style={{ width: `${bowlingStrength * 10}%` }}
                                />
                            </div>
                            <div className="text-xs text-dark-muted mt-1">
                                {avgBowlingEconomy > 0 ? `Avg Econ: ${avgBowlingEconomy.toFixed(1)} | Elite: ${eliteBowlers}` : 'No bowling data'}
                            </div>
                        </div>

                        <div>
                            <div className="flex items-center justify-between text-sm mb-2">
                                <span className="text-dark-muted">Team Depth</span>
                                <span className="text-cricket-green font-semibold">{depthStrength}/10</span>
                            </div>
                            <div className="h-2 bg-dark-card rounded-full overflow-hidden border border-dark-border">
                                <div
                                    className="h-full bg-gradient-to-r from-purple-500 to-purple-400 transition-all"
                                    style={{ width: `${depthStrength * 10}%` }}
                                />
                            </div>
                            <div className="text-xs text-dark-muted mt-1">
                                {battingDepth} players with avg ≥ 30
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </motion.div>
    );
};

export default TeamFormationDisplay;
