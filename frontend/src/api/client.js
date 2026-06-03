const API_BASE_URL = import.meta.env.DEV 
  ? "http://localhost:8000" 
  : "https://recsys-backend-1dge.onrender.com"; // Your Render backend URL

export async function fetchFromAPI(endpoint, options = {}) {
  const url = `${API_BASE_URL}${endpoint}`;
  const response = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
      ...options.headers,
    },
    ...options,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `API error: ${response.statusText}`);
  }

  return response.json();
}

export const apiClient = {
  // GET /movies?genre=Action&limit=50
  getMovies: (genre = "", query = "", limit = 50) => {
    const params = new URLSearchParams();
    if (genre) params.append("genre", genre);
    if (query) params.append("query", query);
    params.append("limit", limit);
    return fetchFromAPI(`/movies?${params.toString()}`);
  },

  // GET /users/{user_id}/history
  getUserHistory: (userId) => {
    return fetchFromAPI(`/users/${userId}/history`);
  },

  // POST /recommend
  getRecommendations: (userId, model, topK = 10, debias = false) => {
    return fetchFromAPI("/recommend", {
      method: "POST",
      body: JSON.stringify({
        user_id: parseInt(userId, 10),
        model,
        top_k: topK,
        debias,
      }),
    });
  },

  // POST /compare-models
  compareModels: (userId, topK = 10) => {
    return fetchFromAPI("/compare-models", {
      method: "POST",
      body: JSON.stringify({
        user_id: parseInt(userId, 10),
        top_k: topK,
      }),
    });
  },

  // GET /metrics
  getMetrics: () => {
    return fetchFromAPI("/metrics");
  },
};
