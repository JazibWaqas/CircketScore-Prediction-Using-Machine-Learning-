import axios from 'axios';

// Use an explicit production API when configured by the frontend host.
const getApiUrl = () => {
  const configuredUrl = process.env.REACT_APP_API_URL;
  if (configuredUrl) {
    return configuredUrl.replace(/\/$/, '');
  }

  // Local React development and the deployed portfolio fallback.
  if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
    return 'http://localhost:5002/api';
  }

  return 'https://cricket-score-predictor-api.onrender.com/api';
};

const API_BASE_URL = getApiUrl();

// Free-tier hosting sleeps between visits, so a cold request can take a while.
// The ceiling is generous on purpose: timing out early would surface an error
// for a request that was about to succeed.
const client = axios.create({ baseURL: API_BASE_URL, timeout: 90000 });

export const api = {
  // Health check, also used on mount to spin the container up.
  health: () => client.get('/health'),
  
  // Get teams
  getTeams: () => client.get('/teams'),

  // Get all players
  getPlayers: () => client.get('/players'),

  // Get venues
  getVenues: () => client.get('/venues'),

  // Get available models
  getModels: () => client.get('/models'),

  // Make prediction
  predict: (data) => client.post('/predict', data),

  // What-if analysis
  whatif: (data) => client.post('/whatif', data),

  // Progressive predictions
  progressive: (data) => client.post('/progressive', data)
};

export default api;
