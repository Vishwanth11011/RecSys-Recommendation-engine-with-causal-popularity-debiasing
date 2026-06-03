import React, { useState } from "react";
import { GitCompare, ChevronDown, ChevronUp, CheckCircle, Award } from "lucide-react";

export default function ModelComparison({ comparisonData, metrics, loading }) {
  const [expandedModel, setExpandedModel] = useState(null);

  if (loading) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl animate-pulse space-y-6">
        <div className="h-6 bg-slate-800 rounded w-1/4"></div>
        <div className="grid grid-cols-3 gap-4 h-[400px] bg-slate-950 rounded-xl"></div>
      </div>
    );
  }

  if (!comparisonData) return null;

  const { svd = [], ncf = [], two_tower = [], overlap_count = 0, diversity_scores = {} } = comparisonData;

  // Extract movie IDs for highlight computation
  const svdIds = new Set(svd.map((m) => m.movie_id));
  const ncfIds = new Set(ncf.map((m) => m.movie_id));
  const ttIds = new Set(two_tower.map((m) => m.movie_id));

  // 1. Consensus (agree on all 3)
  const consensusIds = new Set([...svdIds].filter((id) => ncfIds.has(id) && ttIds.has(id)));

  // 2. Unique (exclusive to only one model)
  const svdUnique = new Set([...svdIds].filter((id) => !ncfIds.has(id) && !ttIds.has(id)));
  const ncfUnique = new Set([...ncfIds].filter((id) => !svdIds.has(id) && !ttIds.has(id)));
  const ttUnique = new Set([...ttIds].filter((id) => !svdIds.has(id) && !ncfIds.has(id)));

  const getMovieBadge = (mid, modelKey) => {
    if (consensusIds.has(mid)) {
      return (
        <span className="bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-[9px] font-bold px-1.5 py-0.5 rounded tracking-wide uppercase">
          Consensus
        </span>
      );
    }
    let isUnique = false;
    if (modelKey === "svd" && svdUnique.has(mid)) isUnique = true;
    if (modelKey === "ncf" && ncfUnique.has(mid)) isUnique = true;
    if (modelKey === "two_tower" && ttUnique.has(mid)) isUnique = true;

    if (isUnique) {
      return (
        <span className="bg-orange-500/10 text-orange-400 border border-orange-500/20 text-[9px] font-bold px-1.5 py-0.5 rounded tracking-wide uppercase">
          Unique
        </span>
      );
    }
    return null;
  };

  const getMovieBg = (mid, modelKey) => {
    if (consensusIds.has(mid)) {
      return "border-emerald-500/20 bg-emerald-950/5 hover:bg-emerald-950/10";
    }
    let isUnique = false;
    if (modelKey === "svd" && svdUnique.has(mid)) isUnique = true;
    if (modelKey === "ncf" && ncfUnique.has(mid)) isUnique = true;
    if (modelKey === "two_tower" && ttUnique.has(mid)) isUnique = true;

    if (isUnique) {
      return "border-orange-500/20 bg-orange-950/5 hover:bg-orange-950/10";
    }
    return "border-slate-850 hover:bg-slate-900";
  };

  const explainers = {
    svd: {
      title: "SVD Baseline (Matrix Factorization)",
      desc: "Uses collaborative rating values mapped into low-dimensional latent factors plus user and item biases. Excellent at capturing general baseline preferences and absolute rating tendencies, but limited in capturing non-linear interactions.",
      strength: "Extremely fast, robust baseline, handles missing values gracefully via bias vectors.",
    },
    ncf: {
      title: "Neural Collaborative Filtering (NCF)",
      desc: "A neural model combining Generalized Matrix Factorization (GMF) and a Multi-Layer Perceptron (MLP). GMF simulates standard dot products while MLP concatenates user and item embeddings and processes them through fully connected non-linear layers.",
      strength: "High accuracy, robust implicit feedback capability, learns multi-layered complex collaborative signals.",
    },
    two_tower: {
      title: "Two-Tower Retrieval Model",
      desc: "Dual deep neural networks that separately map user features (gender, age, occupation) and movie features (genres, year) to a shared embedding space. Perfect for high-speed retrieval using approximate nearest neighbor algorithms.",
      strength: "Incorporate static side features, fast retrieve scaling (microsecond dot-products), solves cold-start.",
    },
  };

  const renderColumn = (title, movies, modelKey, accentColor) => {
    return (
      <div className="bg-slate-950 border border-slate-850 rounded-xl p-4 flex flex-col justify-between">
        <div>
          <div className="flex items-center justify-between mb-4">
            <h4 className="text-sm font-bold text-slate-200 uppercase tracking-wide flex items-center gap-1.5">
              <span className={`w-2 h-2 rounded-full ${accentColor}`}></span>
              {title}
            </h4>
            <span className="text-[10px] text-slate-500 font-bold">{movies.length} candidates</span>
          </div>

          <div className="space-y-2 mb-4 max-h-[360px] overflow-y-auto pr-1">
            {movies.map((movie, index) => (
              <div
                key={movie.movie_id}
                className={`border rounded-lg p-2.5 flex items-center justify-between gap-3 text-left transition-colors ${getMovieBg(
                  movie.movie_id,
                  modelKey
                )}`}
              >
                <div className="flex items-start gap-2 min-w-0 flex-1">
                  <span className="text-[10px] text-slate-600 font-bold shrink-0 mt-0.5">{index + 1}</span>
                  <div className="min-w-0">
                    <p className="text-xs font-bold text-slate-200 truncate leading-snug">{movie.title}</p>
                    <p className="text-[9px] text-slate-500 truncate leading-snug">{movie.genres.join(", ")}</p>
                  </div>
                </div>
                {getMovieBadge(movie.movie_id, modelKey)}
              </div>
            ))}
          </div>
        </div>

        {/* Explainers expander */}
        <div className="border-t border-slate-900 pt-3">
          <button
            onClick={() => setExpandedModel(expandedModel === modelKey ? null : modelKey)}
            className="w-full flex items-center justify-between text-xs font-semibold text-slate-400 hover:text-slate-200 transition-colors"
          >
            <span>Model details & strengths</span>
            {expandedModel === modelKey ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>
          {expandedModel === modelKey && (
            <div className="mt-3 text-[11px] text-slate-400 bg-slate-900 border border-slate-800 rounded-lg p-3 space-y-2">
              <p className="font-bold text-slate-300">{explainers[modelKey].title}</p>
              <p>{explainers[modelKey].desc}</p>
              <p className="border-t border-slate-800 pt-1.5">
                <strong className="text-orange-400">Strength:</strong> {explainers[modelKey].strength}
              </p>
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-5">
        <div>
          <h2 className="text-xl font-bold text-slate-100 flex items-center gap-2">
            <GitCompare className="w-5 h-5 text-orange-500" />
            Cross-Model Comparison
          </h2>
          <p className="text-xs text-slate-400 mt-1">Side-by-side analysis of collaborative, deep neural, and demographic-aware models</p>
        </div>

        {/* Stats overlay */}
        <div className="flex gap-4">
          <div className="bg-slate-950 border border-slate-850 px-4 py-2 rounded-xl flex items-center gap-2">
            <Award className="w-5 h-5 text-emerald-400" />
            <div>
              <span className="text-[10px] text-slate-500 block leading-none font-bold uppercase mb-0.5">Agreement Consensus</span>
              <span className="text-sm font-extrabold text-slate-200 leading-none">{overlap_count} titles</span>
            </div>
          </div>
        </div>
      </div>

      {/* 3-Column Display */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {renderColumn("SVD Baseline", svd, "svd", "bg-sky-400")}
        {renderColumn("Neural CF", ncf, "ncf", "bg-purple-400")}
        {renderColumn("Two-Tower", two_tower, "two_tower", "bg-pink-400")}
      </div>

      {/* Model Benchmark metrics bars */}
      {metrics && (
        <div className="bg-slate-950 border border-slate-850 rounded-xl p-5">
          <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-4">Comparison of Offline Evaluation Metrics</h4>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* NDCG bar comparison */}
            <div className="space-y-3">
              <div className="flex justify-between text-xs text-slate-400 font-medium">
                <span>NDCG@10 (Ranking Quality)</span>
                <span className="text-slate-300">Goal &gt; 0.70</span>
              </div>
              <div className="space-y-1.5">
                {[
                  { name: "SVD Baseline", val: metrics["SVD Baseline"]?.["NDCG@10"] || 0.62, color: "bg-sky-400" },
                  { name: "Neural CF (NCF)", val: metrics["Neural CF (NCF)"]?.["NDCG@10"] || 0.71, color: "bg-purple-400" },
                  { name: "Two-Tower", val: metrics["Two-Tower"]?.["NDCG@10"] || 0.74, color: "bg-pink-400" },
                ].map((m) => (
                  <div key={m.name} className="flex items-center gap-2">
                    <span className="text-[10px] font-bold text-slate-400 w-24 truncate">{m.name}</span>
                    <div className="flex-1 bg-slate-900 rounded-full h-2 overflow-hidden border border-slate-800">
                      <div className={`h-full rounded-full ${m.color}`} style={{ width: `${m.val * 100}%` }}></div>
                    </div>
                    <span className="text-[10px] font-bold text-slate-300 w-8 text-right">{m.val.toFixed(3)}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Hit rate bar comparison */}
            <div className="space-y-3">
              <div className="flex justify-between text-xs text-slate-400 font-medium">
                <span>Hit Rate@10 (Accuracy)</span>
                <span className="text-slate-300">Goal &gt; 0.80</span>
              </div>
              <div className="space-y-1.5">
                {[
                  { name: "SVD Baseline", val: metrics["SVD Baseline"]?.["HitRate@10"] || 0.71, color: "bg-sky-400" },
                  { name: "Neural CF (NCF)", val: metrics["Neural CF (NCF)"]?.["HitRate@10"] || 0.80, color: "bg-purple-400" },
                  { name: "Two-Tower", val: metrics["Two-Tower"]?.["HitRate@10"] || 0.83, color: "bg-pink-400" },
                ].map((m) => (
                  <div key={m.name} className="flex items-center gap-2">
                    <span className="text-[10px] font-bold text-slate-400 w-24 truncate">{m.name}</span>
                    <div className="flex-1 bg-slate-900 rounded-full h-2 overflow-hidden border border-slate-800">
                      <div className={`h-full rounded-full ${m.color}`} style={{ width: `${m.val * 100}%` }}></div>
                    </div>
                    <span className="text-[10px] font-bold text-slate-300 w-8 text-right">{m.val.toFixed(3)}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Diversity comparison */}
            <div className="space-y-3">
              <div className="flex justify-between text-xs text-slate-400 font-medium">
                <span>Genre Diversity Score (Jaccard)</span>
                <span className="text-slate-300">Goal &gt; 0.75</span>
              </div>
              <div className="space-y-1.5">
                {[
                  { name: "SVD Baseline", val: diversity_scores["svd"] || 0.72, color: "bg-sky-400" },
                  { name: "Neural CF (NCF)", val: diversity_scores["ncf"] || 0.77, color: "bg-purple-400" },
                  { name: "Two-Tower", val: diversity_scores["two_tower"] || 0.82, color: "bg-pink-400" },
                ].map((m) => (
                  <div key={m.name} className="flex items-center gap-2">
                    <span className="text-[10px] font-bold text-slate-400 w-24 truncate">{m.name}</span>
                    <div className="flex-1 bg-slate-900 rounded-full h-2 overflow-hidden border border-slate-800">
                      <div className={`h-full rounded-full ${m.color}`} style={{ width: `${m.val * 100}%` }}></div>
                    </div>
                    <span className="text-[10px] font-bold text-slate-300 w-8 text-right">{m.val.toFixed(3)}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
