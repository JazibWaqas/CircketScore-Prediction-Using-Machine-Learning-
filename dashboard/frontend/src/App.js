import React, { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import Header from './components/Header';
import TeamSelector from './components/TeamSelector';
import MatchScenario from './components/MatchScenario';
import PredictionDisplay from './components/PredictionDisplay';
import LoadingSpinner from './components/LoadingSpinner';
import MatchContextDisplay from './components/MatchContextDisplay';
import TeamFormationDisplay from './components/TeamFormationDisplay';
import api from './utils/api';

function App() {
  const [teams, setTeams] = useState([]);
  const [players, setPlayers] = useState([]);
  const [venues, setVenues] = useState([]);
  const [loading, setLoading] = useState(true);
  const [predicting, setPredicting] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);

  // Team selection - matches root frontend structure
  const [teamA, setTeamA] = useState({
    team_id: null,
    team_name: '',
    players: []
  });

  const [teamB, setTeamB] = useState({
    team_id: null,
    team_name: '',
    players: []
  });

  // Match scenario
  const [matchScenario, setMatchScenario] = useState({
    venue: '',
    venue_avg_score: 250,
    current_score: '',
    wickets_fallen: '',
    overs: '',
    runs_last_10: '',
    batsman_1: '',
    batsman_2: ''
  });
  // Show players from any country when true (What-if scenario)
  const [whatIfAllPlayers, setWhatIfAllPlayers] = useState(false);

  const selectedModel = 'xgboost';
  const predictionRef = useRef(null);

  // Load initial data
  useEffect(() => {
    const loadData = async () => {
      try {
        // Load essential data first
        const [teamsRes, playersRes, venuesRes] = await Promise.all([
          api.getTeams(),
          api.getPlayers(),
          api.getVenues()
        ]);

        setTeams(teamsRes.data.teams);
        setPlayers(playersRes.data.players);
        setVenues(venuesRes.data.venues);

        // Model loading logic removed inline with simplifying to XGBoost only

        setLoading(false);
      } catch (err) {
        setError('Failed to load prediction data. The API may be starting up; please try again shortly.');
        console.error('Load error:', err);
        setLoading(false);
      }
    };

    loadData();
  }, []);

  // Handle team selection
  const handleTeamSelect = (teamType, teamId, teamName) => {
    if (teamType === 'A') {
      setTeamA({ team_id: teamId, team_name: teamName, players: [] });
    } else {
      setTeamB({ team_id: teamId, team_name: teamName, players: [] });
    }
  };

  // Handle player selection
  const handlePlayerSelect = (teamType, playerId, playerName, playerCountry) => {
    const player = { id: playerId, name: playerName, country: playerCountry };

    if (teamType === 'A') {
      if (teamA.players.length < 11) {
        setTeamA(prev => ({ ...prev, players: [...prev.players, player] }));
      }
    } else {
      if (teamB.players.length < 11) {
        setTeamB(prev => ({ ...prev, players: [...prev.players, player] }));
      }
    }
  };

  // Handle player removal
  const handleRemovePlayer = (teamType, playerId) => {
    if (teamType === 'A') {
      setTeamA(prev => ({ ...prev, players: prev.players.filter(p => p.id !== playerId) }));
    } else {
      setTeamB(prev => ({ ...prev, players: prev.players.filter(p => p.id !== playerId) }));
    }
  };

  // Handle prediction
  const handlePredict = async () => {
    console.log('Prediction request - Team A players:', teamA.players.length);
    console.log('Prediction request - Team B players:', teamB.players.length);
    console.log('Prediction request - Venue:', matchScenario.venue);

    if (teamA.players.length < 11) {
      setError('Please select 11 players for Team A (Batting Team)');
      return;
    }

    if (teamB.players.length < 11) {
      setError('Please select 11 players for Team B (Opposition)');
      return;
    }

    if (!matchScenario.venue) {
      setError('Please select a venue');
      return;
    }

    setPredicting(true);
    setError(null);

    try {
      const current_score = Number(matchScenario.current_score) || 0;
      const wickets_fallen = Number(matchScenario.wickets_fallen) || 0;
      const oversNumber = Number(matchScenario.overs) || 0;
      const runs_last_10 = Number(matchScenario.runs_last_10) || 0;
      const balls_bowled = oversNumber * 6;

      const requestData = {
        batting_team_players: teamA.players.map(p => p.name),
        bowling_team_players: teamB.players.map(p => p.name),
        venue: matchScenario.venue,
        venue_avg_score: matchScenario.venue_avg_score,
        current_score: current_score,
        wickets_fallen: wickets_fallen,
        balls_bowled: balls_bowled,
        runs_last_10_overs: runs_last_10,
        batsman_1: matchScenario.batsman_1,
        batsman_2: matchScenario.batsman_2,
        model: selectedModel  // Include selected model
      };

      console.log('Sending prediction request:', requestData);

      const response = await api.predict(requestData);

      setPrediction(response.data);
      setTimeout(() => {
        predictionRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 100);
    } catch (err) {
      setError(err.response?.data?.error || 'Prediction failed. Check console for details.');
      console.error('Prediction error:', err);
    } finally {
      setPredicting(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <LoadingSpinner />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-dark-bg">
      <Header />

      <main className="max-w-7xl mx-auto px-6 py-8">
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-6 bg-red-900/20 border border-red-500/50 rounded-lg p-4 text-red-400"
          >
            {error}
            <button
              onClick={() => setError(null)}
              className="ml-4 text-sm underline"
            >
              Dismiss
            </button>
          </motion.div>
        )}

        {/* Info Banner & Features */}
        <div className="mb-8 bg-dark-card border border-dark-border rounded-xl p-6 shadow-lg">
          <h2 className="text-xl font-bold text-cricket-green mb-3">Welcome to ODI Predictor</h2>
          <p className="text-dark-text mb-4 text-sm leading-relaxed">
            Choose your custom fantasy team or use real squads to predict upcoming or live International ODI final scores. Our platform runs your scenarios through an advanced machine learning engine trained entirely on international ODI matches.
          </p>
          <div className="flex flex-col md:flex-row gap-4 text-sm text-dark-muted items-start md:items-center">
            <span className="flex items-center gap-2"><span className="flex items-center justify-center w-5 h-5 rounded-full bg-cricket-green/20 text-cricket-green font-bold text-xs">✓</span> Build Fantasy Teams</span>
            <span className="flex items-center gap-2"><span className="flex items-center justify-center w-5 h-5 rounded-full bg-cricket-green/20 text-cricket-green font-bold text-xs">✓</span> Compare Player Impact</span>
            <span className="flex items-center gap-2"><span className="flex items-center justify-center w-5 h-5 rounded-full bg-cricket-green/20 text-cricket-green font-bold text-xs">✓</span> Live Match Context</span>

            <div className="mt-2 md:mt-0 md:ml-auto inline-flex items-center bg-black/30 px-4 py-2 rounded-lg border border-dark-border py-2 text-sm">
              <input
                type="checkbox"
                id="whatif-toggle"
                checked={whatIfAllPlayers}
                onChange={(e) => setWhatIfAllPlayers(e.target.checked)}
                className="h-4 w-4 rounded text-cricket-green border-gray-600 focus:ring-cricket-green/50 bg-dark-card mr-3 cursor-pointer"
              />
              <label htmlFor="whatif-toggle" className="cursor-pointer text-white">Enable What-If (All Countries)</label>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* Team A - Batting Team */}
          <TeamSelector
            teamType="A"
            team={teamA}
            teams={teams}
            players={players}
            whatIfAllPlayers={whatIfAllPlayers}
            onTeamSelect={handleTeamSelect}
            onPlayerSelect={handlePlayerSelect}
            onRemovePlayer={handleRemovePlayer}
          />

          {/* Team B - Opposition */}
          <TeamSelector
            teamType="B"
            team={teamB}
            teams={teams}
            players={players}
            whatIfAllPlayers={whatIfAllPlayers}
            onTeamSelect={handleTeamSelect}
            onPlayerSelect={handlePlayerSelect}
            onRemovePlayer={handleRemovePlayer}
          />
        </div>

        {/* Team Formation Displays */}
        {(teamA.players.length > 0 || teamB.players.length > 0) && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
            {teamA.players.length > 0 && (
              <TeamFormationDisplay
                team={teamA}
                teamType="A"
                players={players}
              />
            )}
            {teamB.players.length > 0 && (
              <TeamFormationDisplay
                team={teamB}
                teamType="B"
                players={players}
              />
            )}
          </div>
        )}

        {/* Match Scenario */}
        <MatchScenario
          scenario={matchScenario}
          onChange={setMatchScenario}
          venues={venues}
          battingPlayers={teamA.players}
        />

        {/* Match Context Display */}
        <MatchContextDisplay
          teamA={teamA}
          teamB={teamB}
          scenario={matchScenario}
          venues={venues}
        />

        {/* Predict Button */}
        <div className="text-center my-8 flex flex-col items-center">
          <motion.button
            whileHover={{ scale: (predicting || teamA.players.length < 11 || teamB.players.length < 11) ? 1 : 1.05 }}
            whileTap={{ scale: (predicting || teamA.players.length < 11 || teamB.players.length < 11) ? 1 : 0.95 }}
            onClick={handlePredict}
            disabled={predicting || teamA.players.length < 11 || teamB.players.length < 11}
            className={`text-xl px-12 py-4 rounded-lg font-bold shadow-lg transition-all duration-300 ${(predicting || teamA.players.length < 11 || teamB.players.length < 11)
                ? 'bg-gray-800 text-gray-500 cursor-not-allowed border border-gray-700'
                : 'bg-cricket-green text-white hover:bg-green-600 hover:scale-105'
              }`}
          >
            {predicting ? 'Predicting...' : '🏏 Predict Final Score'}
          </motion.button>

          {(teamA.players.length < 11 || teamB.players.length < 11) && (
            <div className="mt-3 text-sm text-yellow-500/80 bg-yellow-900/10 px-4 py-2 rounded-lg border border-yellow-700/30">
              Select <b>{Math.max(0, 11 - teamA.players.length)}</b> more players for Team A and <b>{Math.max(0, 11 - teamB.players.length)}</b> for Team B to enable.
            </div>
          )}
        </div>

        {/* Prediction Results */}
        <div ref={predictionRef}>
          {prediction && (
            <PredictionDisplay prediction={prediction} scenario={matchScenario} />
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
