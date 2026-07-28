import React from 'react';
import { motion } from 'framer-motion';
import { MapPin, Target } from 'lucide-react';

const MatchContextDisplay = ({ teamA, teamB, scenario, venues }) => {
    if (!teamA.team_name || !teamB.team_name) {
        return null;
    }

    const currentScore = Number(scenario.current_score) || 0;
    const wickets = Number(scenario.wickets_fallen) || 0;
    const overs = Number(scenario.overs) || 0;
    const runsLast10 = Number(scenario.runs_last_10) || 0;

    const currentRunRate = overs > 0 ? (currentScore / overs).toFixed(2) : '0.00';
    const remainingOvers = 50 - overs;
    const progressPercentage = (overs / 50) * 100;

    const selectedVenue = venues.find(v => v.venue_name === scenario.venue);
    const venueAvg = selectedVenue?.avg_score || scenario.venue_avg_score || 250;

    // Wickets visualization
    const wicketsArray = Array(10).fill(0).map((_, i) => i < wickets);

    return (
        <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="cricket-card mb-8 border-2 border-cricket-green/30"
        >
            {/* Match Header */}
            <div className="bg-gradient-to-r from-cricket-green/20 to-transparent p-6 border-b border-cricket-green/30">
                <div className="flex items-center justify-between mb-4">
                    <div className="flex-1 text-center">
                        <div className="text-2xl font-bold text-cricket-green">{teamA.team_name}</div>
                        <div className="text-sm text-dark-muted">Batting</div>
                    </div>

                    <div className="px-6 text-dark-muted font-semibold">vs</div>

                    <div className="flex-1 text-center">
                        <div className="text-2xl font-bold text-dark-text">{teamB.team_name}</div>
                        <div className="text-sm text-dark-muted">Bowling</div>
                    </div>
                </div>

                {/* Venue */}
                {scenario.venue && (
                    <div className="flex items-center justify-center gap-2 text-dark-muted">
                        <MapPin className="h-4 w-4" />
                        <span className="text-sm">
                            {scenario.venue} <span className="text-cricket-green">(Avg: {venueAvg.toFixed(0)})</span>
                        </span>
                    </div>
                )}
            </div>

            {/* Score Display */}
            <div className="p-6">
                <div className="text-center mb-6">
                    <div className="flex items-center justify-center gap-4">
                        <div>
                            <div className="text-5xl font-bold text-cricket-green">
                                {currentScore}/{wickets}
                            </div>
                            <div className="text-lg text-dark-muted mt-1">
                                ({overs}.0 overs)
                            </div>
                        </div>
                    </div>

                    {/* Wickets Visualization */}
                    <div className="flex items-center justify-center gap-2 mt-4">
                        {wicketsArray.map((fallen, idx) => (
                            <div
                                key={idx}
                                className={`w-6 h-6 rounded-full border-2 flex items-center justify-center text-xs font-bold transition-all ${fallen
                                        ? 'bg-cricket-red border-cricket-red text-white'
                                        : 'bg-cricket-green/20 border-cricket-green text-cricket-green'
                                    }`}
                            >
                                {fallen ? '✕' : '○'}
                            </div>
                        ))}
                    </div>
                    <div className="text-xs text-dark-muted mt-2">
                        {wickets} down, {10 - wickets} wickets remaining
                    </div>
                </div>

                {/* Stats Row */}
                <div className="grid grid-cols-3 gap-4 mb-6">
                    <div className="text-center p-3 bg-dark-card rounded-lg border border-dark-border">
                        <div className="text-xs text-dark-muted mb-1">Current RR</div>
                        <div className="text-xl font-bold text-cricket-green">{currentRunRate}</div>
                    </div>

                    <div className="text-center p-3 bg-dark-card rounded-lg border border-dark-border">
                        <div className="text-xs text-dark-muted mb-1">Last 10 Overs</div>
                        <div className="text-xl font-bold text-cricket-green">{runsLast10}</div>
                    </div>

                    <div className="text-center p-3 bg-dark-card rounded-lg border border-dark-border">
                        <div className="text-xs text-dark-muted mb-1">Remaining</div>
                        <div className="text-xl font-bold text-dark-text">{remainingOvers} overs</div>
                    </div>
                </div>

                {/* Progress Bar */}
                <div className="mb-4">
                    <div className="flex items-center justify-between text-xs text-dark-muted mb-2">
                        <span>Match Progress</span>
                        <span>{overs}/50 overs ({progressPercentage.toFixed(0)}%)</span>
                    </div>
                    <div className="h-3 bg-dark-card rounded-full overflow-hidden border border-dark-border">
                        <motion.div
                            initial={{ width: 0 }}
                            animate={{ width: `${progressPercentage}%` }}
                            transition={{ duration: 0.8, ease: 'easeOut' }}
                            className="h-full bg-gradient-to-r from-cricket-green to-green-400"
                        />
                    </div>
                </div>

                {/* Current Batsmen */}
                {(scenario.batsman_1 || scenario.batsman_2) && (
                    <div className="mt-6 p-4 bg-cricket-green/10 border border-cricket-green/30 rounded-lg">
                        <div className="flex items-center gap-2 mb-3">
                            <Target className="h-4 w-4 text-cricket-green" />
                            <div className="text-sm font-semibold text-cricket-green">At The Crease</div>
                        </div>
                        <div className="grid grid-cols-2 gap-4">
                            {scenario.batsman_1 && (
                                <div className="text-center">
                                    <div className="text-sm font-medium text-dark-text">{scenario.batsman_1}</div>
                                    <div className="text-xs text-dark-muted">Striker</div>
                                </div>
                            )}
                            {scenario.batsman_2 && (
                                <div className="text-center">
                                    <div className="text-sm font-medium text-dark-text">{scenario.batsman_2}</div>
                                    <div className="text-xs text-dark-muted">Non-Striker</div>
                                </div>
                            )}
                        </div>
                    </div>
                )}
            </div>
        </motion.div>
    );
};

export default MatchContextDisplay;
