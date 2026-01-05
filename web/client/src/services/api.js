import axios from "axios";

const API_BASE_URL = "http://localhost:5000";

export const stockAPI = {
  // Get Top 100 list
  getTop100List: async () => {
    const response = await axios.get(`${API_BASE_URL}/top100-list`);
    return response.data;
  },

  // Get Top 100 history
  getTop100History: async (days = 30) => {
    const response = await axios.get(
      `${API_BASE_URL}/top100-history?days=${days}`
    );

    return response.data;
  },

  // Get single stock history
  getStockHistory: async (symbol, days = 30) => {
    const response = await axios.get(
      `${API_BASE_URL}/stock/${symbol}?days=${days}`
    );
    return response.data;
  },

  // Get top recommendations from saved CSV (server-side)
  getTopRecommendationsCsv: async (limit = 100, sort = true) => {
    const response = await axios.get(`${API_BASE_URL}/predict-top100-csv`, {
      params: { limit, sort },
    });
    return response.data;
  },

  // Get top recommendations by computing on server now
  // source: 'VNStock' (live) or 'local'
  getTopRecommendations: async ({ source = "VNStock", limit = 100, save_csv = false, days = 60 } = {}) => {
    const response = await axios.get(`${API_BASE_URL}/predict-top100`, {
      params: { source, limit, save_csv, days },
    });
    return response.data;
  },
};
