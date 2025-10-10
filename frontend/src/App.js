import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';
import Header from './components/Header';
import TeamSelector from './components/TeamSelector';
import MatchContext from './components/MatchContext';
import PredictionResults from './components/PredictionResults';
import ModelSelector from './components/ModelSelector';
import LoadingSpinner from './components/LoadingSpinner';

function App() {
  // Format selection: T20 or ODI
  const [format, setFormat] = useState('T20'); // 'T20' or 'ODI'
  const API_BASE_URL = format === 'T20' 
    ? 'http://localhost:5000/api' 
    : 'http://localhost:5001/api/odi';
  
  const [teams, setTeams] = useState([]);
  const [venues, setVenues] = useState([]);
  const [players, setPlayers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [prediction, setPrediction] = useState(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [selectedModel, setSelectedModel] = useState('xgboost');
  
  // Team selection state
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
  
  // Match context state - ONLY features the model actually uses
  const [matchContext, setMatchContext] = useState({
    venue: null,
    date: new Date(),
    battingFirst: null,
    tossWinner: null,
    tossDecision: null,
    tournamentType: '',
    seasonYear: 2025,
    seasonMonth: 1,
    isWinter: false,
    isSummer: true,
    isMonsoon: false,
    gender: 'male'
  });

  useEffect(() => {
    loadInitialData();
  }, [format]); // Reload data when format changes

  const loadInitialData = async () => {
    try {
      setLoading(true);
      
      const [teamsRes, venuesRes, playersRes] = await Promise.all([
        axios.get(`${API_BASE_URL}/teams`),
        axios.get(`${API_BASE_URL}/venues`),
        axios.get(`${API_BASE_URL}/players`)
      ]);
      
      // Ensure data is array
      setTeams(Array.isArray(teamsRes.data) ? teamsRes.data : []);
      setVenues(Array.isArray(venuesRes.data) ? venuesRes.data : []);
      setPlayers(Array.isArray(playersRes.data) ? playersRes.data : []);
      
      console.log(`Loaded ${format} data:`, {
        teams: teamsRes.data?.length || 0,
        venues: venuesRes.data?.length || 0,
        players: playersRes.data?.length || 0
      });
      
    } catch (error) {
      console.error('Error loading data:', error);
      // Set empty arrays on error
      setTeams([]);
      setVenues([]);
      setPlayers([]);
      alert(`Error loading ${format} data: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleTeamSelection = (teamType, teamId, teamName) => {
    if (teamType === 'A') {
      setTeamA(prev => ({ ...prev, team_id: teamId, team_name: teamName }));
    } else {
      setTeamB(prev => ({ ...prev, team_id: teamId, team_name: teamName }));
    }
  };

  const handlePlayerSelection = (teamType, playerId, playerName, playerCountry) => {
    if (teamType === 'A') {
      setTeamA(prev => ({
        ...prev,
        players: [...prev.players, { id: playerId, name: playerName, country: playerCountry, role: 'player' }]
      }));
    } else {
      setTeamB(prev => ({
        ...prev,
        players: [...prev.players, { id: playerId, name: playerName, country: playerCountry, role: 'player' }]
      }));
    }
  };

  const removePlayer = (teamType, playerId) => {
    if (teamType === 'A') {
      setTeamA(prev => ({
        ...prev,
        players: prev.players.filter(p => p.id !== playerId)
      }));
    } else {
      setTeamB(prev => ({
        ...prev,
        players: prev.players.filter(p => p.id !== playerId)
      }));
    }
  };

  const handlePredict = async () => {
    if (!teamA.team_id || !teamB.team_id || !matchContext.venue || !matchContext.venue.venue_id) {
      alert('Please select both teams and venue');
      return;
    }

    setIsPredicting(true);
    setPrediction(null);

    try {
      let response;
      
      if (format === 'ODI') {
        // ODI API expects player names, not IDs
        const teamAPlayerNames = teamA.players.map(p => p.name || p.player_name);
        const teamBPlayerNames = teamB.players.map(p => p.name || p.player_name);
        
        // Prepare match context with ALL required features
        const odiContext = {
          // Venue features (from selected venue)
          venue_avg: matchContext.venue?.venue_avg || matchContext.venue?.avg_runs_scored || 240,
          venue_high: matchContext.venue?.venue_high || 380,
          venue_low: matchContext.venue?.venue_low || 120,
          venue_std: matchContext.venue?.venue_std || 50,
          venue_matches: matchContext.venue?.venue_matches || matchContext.venue?.total_matches || 50,
          
          // Toss
          toss_won: matchContext.tossWinner?.team_name === teamA.team_name ? 'team_a' : 'team_b',
          toss_decision: matchContext.tossDecision || 'bat',
          
          // Temporal
          year: matchContext.seasonYear || 2024,
          month: matchContext.seasonMonth || 10,
          match_number: 1,
          
          // Pitch & Weather (estimated from month/location)
          temperature: matchContext.temperature || 25,
          humidity: matchContext.humidity || 60,
          pitch_bounce: matchContext.pitch_bounce || 1.0,
          pitch_swing: matchContext.pitch_swing || 0.8,
          
          // Form (defaults - could be enhanced later)
          team_recent_avg: 240,
          team_form_matches: 5,
          opposition_recent_avg: 240,
          
          // H2H (defaults - could be enhanced later)
          h2h_avg_runs: 240,
          h2h_matches: 10,
          h2h_win_rate: 0.5
        };
        
        response = await axios.post(`${API_BASE_URL}/predict`, {
          team_a_players: teamAPlayerNames,
          team_b_players: teamBPlayerNames,
          team_a_name: teamA.team_name,
          team_b_name: teamB.team_name,
          venue_name: matchContext.venue?.venue_name || 'Unknown Venue',
          match_context: odiContext
        });
      } else {
        // T20 API (existing)
        response = await axios.post(`${API_BASE_URL}/predict`, {
          team_a_id: teamA.team_id,
          team_b_id: teamB.team_id,
          venue_id: matchContext.venue.venue_id,
          team_a_players: teamA.players.map(p => p.id),
          team_b_players: teamB.players.map(p => p.id),
          match_context: matchContext,
          model: selectedModel
        });
      }

      if (format === 'ODI' && response.data.success) {
        // Transform ODI response to show both teams
        setPrediction({
          // Both team scores
          predicted_score_a: response.data.final_prediction_a,
          predicted_score_b: response.data.final_prediction_b,
          team_a: response.data.team_a,
          team_b: response.data.team_b,
          predicted_winner: response.data.predicted_winner,
          
          // Impact details
          base_prediction_a: response.data.base_prediction_a,
          base_prediction_b: response.data.base_prediction_b,
          player_adjustment_a: response.data.player_adjustment_a,
          player_adjustment_b: response.data.player_adjustment_b,
          team_a_impact: response.data.team_a_batting_impact,
          team_b_impact: response.data.team_b_batting_impact,
          
          // Meta
          model_used: 'XGBoost + Player Impact',
          model_accuracy: '76%',
          venue: matchContext.venue?.venue_name || '',
          format: 'ODI'
        });
      } else if (response.data.prediction) {
        setPrediction(response.data.prediction);
      } else {
        alert('Error: ' + (response.data.error || 'Unknown error'));
      }
    } catch (error) {
      console.error('Prediction error:', error);
      console.error('Error response:', error.response?.data);
      alert('Error making prediction: ' + (error.response?.data?.error || error.message));
    } finally {
      setIsPredicting(false);
    }
  };

  const resetAll = () => {
    setTeamA({ team_id: null, team_name: '', players: [] });
    setTeamB({ team_id: null, team_name: '', players: [] });
    setMatchContext({
      venue: null,
      date: new Date(),
      battingFirst: null,
      tossWinner: null,
      tossDecision: null,
      tournamentType: '',
      seasonYear: 2025,
      seasonMonth: 1,
      isWinter: false,
      isSummer: true,
      isMonsoon: false,
      gender: 'male'
    });
    setPrediction(null);
  };

  const handleFormatChange = (newFormat) => {
    if (newFormat !== format) {
      setFormat(newFormat);
      resetAll(); // Reset everything when changing format
    }
  };

  if (loading) {
    return <LoadingSpinner />;
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-dark-bg via-dark-card to-dark-bg">
      <Header />
      
      <main className="container mx-auto px-4 py-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="max-w-7xl mx-auto"
        >
          {/* Format Toggle */}
          <div className="flex justify-center mb-8">
            <div className="bg-dark-card rounded-lg p-2 flex gap-2">
              <button
                onClick={() => handleFormatChange('T20')}
                className={`px-8 py-3 rounded-lg font-semibold transition-all duration-300 ${
                  format === 'T20'
                    ? 'bg-cricket-accent text-white shadow-lg'
                    : 'bg-dark-border text-dark-muted hover:bg-dark-muted hover:text-white'
                }`}
              >
                🏏 T20 Cricket
              </button>
              <button
                onClick={() => handleFormatChange('ODI')}
                className={`px-8 py-3 rounded-lg font-semibold transition-all duration-300 ${
                  format === 'ODI'
                    ? 'bg-cricket-accent text-white shadow-lg'
                    : 'bg-dark-border text-dark-muted hover:bg-dark-muted hover:text-white'
                }`}
              >
                🏏 ODI Cricket
              </button>
            </div>
          </div>

          {/* Model Selector */}
          <ModelSelector 
            selectedModel={selectedModel}
            onModelChange={setSelectedModel}
            format={format}
          />

          {/* Team Selection */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
            <TeamSelector
              teamType="A"
              team={teamA}
              teams={teams}
              players={players}
              onTeamSelect={handleTeamSelection}
              onPlayerSelect={handlePlayerSelection}
              onRemovePlayer={removePlayer}
              format={format}
            />
            
            <TeamSelector
              teamType="B"
              team={teamB}
              teams={teams}
              players={players}
              onTeamSelect={handleTeamSelection}
              onPlayerSelect={handlePlayerSelection}
              onRemovePlayer={removePlayer}
              format={format}
            />
          </div>

          {/* Match Context */}
          <MatchContext
            venues={venues}
            teams={[teamA, teamB]}
            context={matchContext}
            onContextChange={setMatchContext}
            format={format}
          />

          {/* Action Buttons */}
          <div className="flex justify-center gap-4 mt-8">
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={handlePredict}
              disabled={isPredicting || !teamA.team_id || !teamB.team_id || !matchContext.venue}
              className="cricket-button disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isPredicting ? (
                <div className="flex items-center gap-2">
                  <div className="loading-spinner"></div>
                  Predicting...
                </div>
              ) : (
                <>
                  <span className="mr-2">🏏</span>
                  Predict Match Outcome
                </>
              )}
            </motion.button>
            
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={resetAll}
              className="bg-dark-border hover:bg-dark-muted text-dark-text px-6 py-3 rounded-lg transition-all duration-300"
            >
              Reset All
            </motion.button>
          </div>

          {/* Prediction Results */}
          <AnimatePresence>
            {prediction && (
              <motion.div
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -30 }}
                transition={{ duration: 0.5 }}
                className="mt-8"
              >
                <PredictionResults prediction={prediction} />
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      </main>
    </div>
  );
}

export default App;
