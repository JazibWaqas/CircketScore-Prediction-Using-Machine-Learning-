import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import Header from './components/Header';
import TeamSelector from './components/TeamSelector';
import MatchScenario from './components/MatchScenario';
import PredictionDisplay from './components/PredictionDisplay';
import LoadingSpinner from './components/LoadingSpinner';
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
    current_score: 0,
    wickets_fallen: 0,
    overs: 0,
    runs_last_10: 0,
    batsman_1: '',
    batsman_2: ''
  });
  
  // Load initial data
  useEffect(() => {
    const loadData = async () => {
      try {
        const [teamsRes, playersRes, venuesRes] = await Promise.all([
          api.getTeams(),
          api.getPlayers(),
          api.getVenues()
        ]);
        
        setTeams(teamsRes.data.teams);
        setPlayers(playersRes.data.players);
        setVenues(venuesRes.data.venues);
        setLoading(false);
      } catch (err) {
        setError('Failed to load data. Make sure backend is running on port 5002');
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
      const balls_bowled = matchScenario.overs * 6;
      
      const requestData = {
        batting_team_players: teamA.players.map(p => p.name),
        bowling_team_players: teamB.players.map(p => p.name),
        venue: matchScenario.venue,
        venue_avg_score: matchScenario.venue_avg_score,
        current_score: matchScenario.current_score,
        wickets_fallen: matchScenario.wickets_fallen,
        balls_bowled: balls_bowled,
        runs_last_10_overs: matchScenario.runs_last_10,
        batsman_1: matchScenario.batsman_1,
        batsman_2: matchScenario.batsman_2
      };
      
      console.log('Sending prediction request:', requestData);
      
      const response = await api.predict(requestData);
      
      setPrediction(response.data);
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
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* Team A - Batting Team */}
          <TeamSelector
            teamType="A"
            team={teamA}
            teams={teams}
            players={players}
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
            onTeamSelect={handleTeamSelect}
            onPlayerSelect={handlePlayerSelect}
            onRemovePlayer={handleRemovePlayer}
          />
        </div>
        
        {/* Match Scenario */}
        <MatchScenario
          scenario={matchScenario}
          onChange={setMatchScenario}
          venues={venues}
          battingPlayers={teamA.players}
        />
        
        {/* Predict Button */}
        <div className="text-center my-8">
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={handlePredict}
            disabled={predicting}
            className="cricket-button text-xl px-12 py-4 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {predicting ? 'Predicting...' : '🏏 Predict Final Score'}
          </motion.button>
        </div>
        
        {/* Prediction Results */}
        {prediction && (
          <PredictionDisplay prediction={prediction} scenario={matchScenario} />
        )}
      </main>
    </div>
  );
}

export default App;
