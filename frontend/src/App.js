import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';
import Header from './components/Header';
import TeamSelector from './components/TeamSelector';
import MatchContext from './components/MatchContext';
import PredictionResults from './components/PredictionResults';
import ModelSelector from './components/ModelSelector';
import LoadingSpinner from './components/LoadingSpinner';

const API_BASE_URL = 'http://localhost:5000/api';

function App() {
  const [teams, setTeams] = useState([]);
  const [venues, setVenues] = useState([]);
  const [players, setPlayers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [prediction, setPrediction] = useState(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [selectedModel, setSelectedModel] = useState('random_forest');
  
  // Team selection state
  const [teamA, setTeamA] = useState({
    id: null,
    name: '',
    players: []
  });
  const [teamB, setTeamB] = useState({
    id: null,
    name: '',
    players: []
  });
  
  // Match context state
  const [matchContext, setMatchContext] = useState({
    venue: null,
    date: new Date(),
    battingFirst: null,
    tossWinner: null,
    tossDecision: null,
    isHomeTeam: false,
    isFinal: false,
    isIPL: false,
    isT20WorldCup: false
  });

  useEffect(() => {
    loadInitialData();
  }, []);

  const loadInitialData = async () => {
    try {
      setLoading(true);
      
      const [teamsRes, venuesRes, playersRes] = await Promise.all([
        axios.get(`${API_BASE_URL}/teams`),
        axios.get(`${API_BASE_URL}/venues`),
        axios.get(`${API_BASE_URL}/players`)
      ]);
      
      setTeams(teamsRes.data);
      setVenues(venuesRes.data);
      setPlayers(playersRes.data);
      
    } catch (error) {
      console.error('Error loading data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleTeamSelection = (teamType, teamId, teamName) => {
    if (teamType === 'A') {
      setTeamA(prev => ({ ...prev, id: teamId, name: teamName }));
    } else {
      setTeamB(prev => ({ ...prev, id: teamId, name: teamName }));
    }
  };

  const handlePlayerSelection = (teamType, playerId, playerName) => {
    if (teamType === 'A') {
      setTeamA(prev => ({
        ...prev,
        players: [...prev.players, { id: playerId, name: playerName }]
      }));
    } else {
      setTeamB(prev => ({
        ...prev,
        players: [...prev.players, { id: playerId, name: playerName }]
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
    if (!teamA.id || !teamB.id || !matchContext.venue || !matchContext.venue.venue_id) {
      alert('Please select both teams and venue');
      return;
    }

    setIsPredicting(true);
    setPrediction(null);

    try {
      const response = await axios.post(`${API_BASE_URL}/predict`, {
        team_a_id: teamA.id,
        team_b_id: teamB.id,
        venue_id: matchContext.venue.venue_id,
        team_a_players: teamA.players.map(p => p.id),
        team_b_players: teamB.players.map(p => p.id),
        match_context: matchContext,
        model: selectedModel
      });

      if (response.data.prediction) {
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
    setTeamA({ id: null, name: '', players: [] });
    setTeamB({ id: null, name: '', players: [] });
    setMatchContext({
      venue: null,
      date: new Date(),
      battingFirst: null,
      tossWinner: null,
      tossDecision: null,
      isHomeTeam: false,
      isFinal: false,
      isIPL: false,
      isT20WorldCup: false
    });
    setPrediction(null);
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
          {/* Model Selector */}
          <ModelSelector 
            selectedModel={selectedModel}
            onModelChange={setSelectedModel}
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
            />
            
            <TeamSelector
              teamType="B"
              team={teamB}
              teams={teams}
              players={players}
              onTeamSelect={handleTeamSelection}
              onPlayerSelect={handlePlayerSelection}
              onRemovePlayer={removePlayer}
            />
          </div>

          {/* Match Context */}
          <MatchContext
            venues={venues}
            teams={[teamA, teamB]}
            context={matchContext}
            onContextChange={setMatchContext}
          />

          {/* Action Buttons */}
          <div className="flex justify-center gap-4 mt-8">
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={handlePredict}
              disabled={isPredicting || !teamA.id || !teamB.id || !matchContext.venue}
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
