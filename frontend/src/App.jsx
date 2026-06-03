import React, { useState, useEffect } from "react";
import { Sparkles, BarChart3, GitCompare, Scale, Flame, RefreshCw } from "lucide-react";
import { apiClient } from "./api/client";
import UserSelector from "./components/UserSelector";
import RecommendationList from "./components/RecommendationList";
import ModelComparison from "./components/ModelComparison";
import MetricsDashboard from "./components/MetricsDashboard";
import BiasVisualizer from "./components/BiasVisualizer";

export default function App() {
  const [selectedUserId, setSelectedUserId] = useState(1);
  const [selectedModel, setSelectedModel] = useState("svd");
  const [debias, setDebias] = useState(false);

  // Data states
  const [recommendations, setRecommendations] = useState([]);
  const [userHistory, setUserHistory] = useState(null);
  const [comparisonData, setComparisonData] = useState(null);
  const [metrics, setMetrics] = useState(null);

  // Loading states
  const [loadingRecs, setLoadingRecs] = useState(false);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [loadingCompare, setLoadingCompare] = useState(false);

  // Load offline benchmark metrics on mount
  useEffect(() => {
    apiClient
      .getMetrics()
      .then((data) => setMetrics(data))
      .catch((err) => console.error("Failed to load benchmark metrics", err));
  }, []);

  // Fetch recommendations whenever user, model, or debias flag changes
  useEffect(() => {
    setLoadingRecs(true);
    apiClient
      .getRecommendations(selectedUserId, selectedModel, 10, debias)
      .then((res) => {
        setRecommendations(res.recommendations);
        setLoadingRecs(false);
      })
      .catch((err) => {
        console.error("Failed to fetch recommendations", err);
        setLoadingRecs(false);
      });
  }, [selectedUserId, selectedModel, debias]);

  // Fetch user history and model comparisons when user ID changes
  useEffect(() => {
    setLoadingHistory(true);
    apiClient
      .getUserHistory(selectedUserId)
      .then((res) => {
        setUserHistory(res);
        setLoadingHistory(false);
      })
      .catch((err) => {
        console.error("Failed to fetch user history", err);
        setLoadingHistory(false);
      });

    setLoadingCompare(true);
    apiClient
      .compareModels(selectedUserId, 10)
      .then((res) => {
        setComparisonData(res);
        setLoadingCompare(false);
      })
      .catch((err) => {
        console.error("Failed to fetch comparison data", err);
        setLoadingCompare(false);
      });
  }, [selectedUserId]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col justify-between">
      {/* Premium Header */}
      <header className="border-b border-slate-900 bg-slate-900/30 backdrop-blur-md sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-orange-500/10 border border-orange-500/20 rounded-2xl flex items-center justify-center">
              <Flame className="w-6 h-6 text-orange-500" />
            </div>
            <div>
              <h1 className="text-lg font-extrabold tracking-tight text-white leading-tight flex items-center gap-2">
                RecSys Engine
                <span className="bg-orange-500/10 text-orange-400 text-[10px] font-extrabold px-2 py-0.5 rounded border border-orange-500/20 tracking-wider uppercase">
                  Hybrid
                </span>
              </h1>
              <p className="text-[10px] text-slate-500 font-bold uppercase tracking-wider mt-0.5">
                Recommendation engine with causal popularity debiasing
              </p>
            </div>
          </div>

          <div className="hidden sm:flex items-center gap-4 text-xs font-semibold text-slate-400">
            <span className="flex items-center gap-1.5 hover:text-white transition-colors cursor-pointer">
              <span className="w-1.5 h-1.5 rounded-full bg-teal-400 animate-pulse"></span>
              FastAPI: Online
            </span>
            <span className="w-1 h-4 bg-slate-800"></span>
            <span>Dataset: MovieLens 1M</span>
          </div>
        </div>
      </header>

      {/* Core Dashboard Content */}
      <main className="max-w-7xl mx-auto px-6 py-8 flex-1 w-full space-y-8">
        {/* Row 1: Profile Selection & Ranked Recommendations */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 items-start">
          <div className="lg:col-span-1">
            <UserSelector
              selectedUserId={selectedUserId}
              onUserChange={setSelectedUserId}
              userHistory={userHistory}
              loadingHistory={loadingHistory}
            />
          </div>

          <div className="lg:col-span-2">
            <RecommendationList
              recommendations={recommendations}
              loading={loadingRecs}
              model={selectedModel}
              onModelChange={setSelectedModel}
              debias={debias}
              onDebiasToggle={setDebias}
            />
          </div>
        </div>

        {/* Row 2: Side-by-Side Model Comparison Grid */}
        <ModelComparison
          comparisonData={comparisonData}
          metrics={metrics}
          loading={loadingCompare}
        />

        {/* Row 3: Metrics Radar & Latency Performance Analytics */}
        <MetricsDashboard metrics={metrics} />

        {/* Row 4: Causal Debiasing & Popularity Bias visualizer */}
        <BiasVisualizer />
      </main>

      {/* Simple Footer */}
      <footer className="border-t border-slate-900 bg-slate-950 py-6 text-center text-xs text-slate-600">
        <p className="max-w-7xl mx-auto px-6 font-medium">
          RecSys Engine — Collaborative Filtering Matrix Factorization (SVD) → Deep NeuMF (NCF) → Metadata-Aware Dual Tower Retrieval.
        </p>
      </footer>
    </div>
  );
}
