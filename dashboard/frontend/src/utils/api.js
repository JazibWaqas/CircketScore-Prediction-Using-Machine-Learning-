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

export const api = {
  // Health check
  health: () => axios.get(`${API_BASE_URL}/health`),
  
  // Get teams
  getTeams: () => axios.get(`${API_BASE_URL}/teams`),
  
  // Get all players
  getPlayers: () => axios.get(`${API_BASE_URL}/players`),
  
  // Get venues
  getVenues: () => axios.get(`${API_BASE_URL}/venues`),
  
  // Get available models
  getModels: () => axios.get(`${API_BASE_URL}/models`),
  
  // Make prediction
  predict: (data) => axios.post(`${API_BASE_URL}/predict`, data),
  
  // What-if analysis
  whatif: (data) => axios.post(`${API_BASE_URL}/whatif`, data),
  
  // Progressive predictions
  progressive: (data) => axios.post(`${API_BASE_URL}/progressive`, data)
};

export default api;
