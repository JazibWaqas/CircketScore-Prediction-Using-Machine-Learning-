import axios from 'axios';

// Auto-detect API URL - works everywhere automatically!
const getApiUrl = () => {
  // If accessing from ngrok or any remote domain, use relative URL (same domain as frontend)
  if (window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
    // Use the same host as the frontend - works when frontend is served from backend
    return `${window.location.protocol}//${window.location.host}/api`;
  }
  // For local development, connect to backend on port 5002
  return 'http://localhost:5002/api';
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
  
  // Make prediction
  predict: (data) => axios.post(`${API_BASE_URL}/predict`, data),
  
  // What-if analysis
  whatif: (data) => axios.post(`${API_BASE_URL}/whatif`, data),
  
  // Progressive predictions
  progressive: (data) => axios.post(`${API_BASE_URL}/progressive`, data)
};

export default api;

