import axios from 'axios';

const API_BASE_URL = 'http://localhost:5002/api';

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

