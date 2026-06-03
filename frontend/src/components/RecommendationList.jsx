import React from "react";
import { Sparkles, Info, ShieldAlert } from "lucide-react";

export const GENRE_COLORS = {
  Action: "bg-red-950/30 text-red-400 border-red-900/30",
  Adventure: "bg-emerald-950/30 text-emerald-400 border-emerald-900/30",
  Animation: "bg-teal-950/30 text-teal-400 border-teal-900/30",
  "Children's": "bg-amber-950/30 text-amber-400 border-amber-900/30",
  Comedy: "bg-yellow-950/30 text-yellow-400 border-yellow-900/30",
  Crime: "bg-rose-950/30 text-rose-400 border-rose-900/30",
  Documentary: "bg-sky-950/30 text-sky-400 border-sky-900/30",
  Drama: "bg-blue-950/30 text-blue-400 border-blue-900/30",
  Fantasy: "bg-indigo-950/30 text-indigo-400 border-indigo-900/30",
  "Film-Noir": "bg-neutral-950/40 text-neutral-400 border-neutral-900/30",
  Horror: "bg-orange-950/30 text-orange-400 border-orange-900/30",
  Musical: "bg-fuchsia-950/30 text-fuchsia-400 border-fuchsia-900/30",
  Mystery: "bg-purple-950/30 text-purple-400 border-purple-900/30",
  Romance: "bg-pink-950/30 text-pink-400 border-pink-900/30",
  "Sci-Fi": "bg-violet-950/30 text-violet-400 border-violet-900/30",
  Thriller: "bg-cyan-950/30 text-cyan-400 border-cyan-900/30",
  War: "bg-stone-950/30 text-stone-400 border-stone-900/30",
  Western: "bg-lime-950/30 text-lime-400 border-lime-900/30",
};

export default function RecommendationList({
  recommendations,
  loading,
  model,
  onModelChange,
  debias,
  onDebiasToggle,
}) {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl h-full flex flex-col">
      {/* Header controls */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-5 mb-5">
        <div>
          <h2 className="text-xl font-bold text-slate-100 flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-orange-500 fill-orange-500/20" />
            Ranked Recommendations
          </h2>
          <p className="text-xs text-slate-400 mt-1">Personalized recommendations filtered to exclude rated titles</p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          {/* Model Selector */}
          <div className="flex bg-slate-950 p-1 rounded-xl border border-slate-850">
            {["svd", "ncf", "two_tower"].map((m) => (
              <button
                key={m}
                onClick={() => onModelChange(m)}
                className={`px-3 py-1.5 text-xs font-semibold rounded-lg capitalize transition-all ${
                  model === m ? "bg-orange-500 text-white shadow-md" : "text-slate-400 hover:text-slate-200"
                }`}
              >
                {m === "svd" ? "SVD" : m === "ncf" ? "Neural CF" : "Two-Tower"}
              </button>
            ))}
          </div>

          {/* Debias Toggle */}
          <button
            onClick={() => onDebiasToggle(!debias)}
            className={`flex items-center gap-2 px-3 py-2 text-xs font-semibold rounded-xl border transition-all ${
              debias
                ? "bg-teal-950/30 text-teal-400 border-teal-800 shadow-[0_0_15px_rgba(20,184,166,0.1)]"
                : "bg-slate-950 text-slate-400 border-slate-850 hover:text-slate-200"
            }`}
          >
            <span className={`w-2 h-2 rounded-full ${debias ? "bg-teal-400 animate-pulse" : "bg-slate-600"}`}></span>
            Causal Debiasing
          </button>
        </div>
      </div>

      {/* Recommendations Display */}
      {loading ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 flex-1 items-center justify-center min-h-[300px]">
          {[...Array(6)].map((_, idx) => (
            <div key={idx} className="bg-slate-950 border border-slate-850 rounded-xl p-4 animate-pulse space-y-3">
              <div className="h-4 bg-slate-800 rounded w-3/4"></div>
              <div className="h-3 bg-slate-800 rounded w-1/2"></div>
              <div className="h-6 bg-slate-800 rounded w-1/4"></div>
            </div>
          ))}
        </div>
      ) : recommendations.length === 0 ? (
        <div className="flex flex-col items-center justify-center text-center p-8 flex-1 min-h-[300px]">
          <ShieldAlert className="w-12 h-12 text-slate-600 mb-3" />
          <h3 className="text-slate-300 font-semibold mb-1">No Recommendations Available</h3>
          <p className="text-xs text-slate-500 max-w-xs">Try selecting a different user profile or models.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 flex-1">
          {recommendations.map((movie, index) => (
            <div
              key={movie.movie_id}
              className="group bg-slate-950 hover:bg-slate-900 border border-slate-850 hover:border-slate-800 rounded-xl p-4 flex flex-col justify-between hover-scale relative overflow-hidden transition-all shadow-sm"
            >
              {/* Badge Overlay */}
              {movie.debiased_badge && (
                <div className="absolute top-0 right-0">
                  <span className="bg-teal-500/10 text-teal-400 border-l border-b border-teal-500/20 text-[10px] font-bold px-2 py-0.5 rounded-bl-lg tracking-wider uppercase block">
                    Long-Tail Corrected
                  </span>
                </div>
              )}

              <div>
                <div className="flex items-start justify-between gap-2 mb-2">
                  <span className="text-[10px] font-bold text-orange-500 w-5 h-5 flex items-center justify-center bg-orange-950/20 border border-orange-900/20 rounded-full shrink-0">
                    {index + 1}
                  </span>
                  <h4 className="text-sm font-bold text-slate-100 leading-tight flex-1 group-hover:text-orange-400 transition-colors pr-2">
                    {movie.title}
                  </h4>
                </div>

                <div className="flex flex-wrap gap-1 mb-4">
                  {movie.genres.map((g) => (
                    <span
                      key={g}
                      className={`px-2 py-0.5 text-[10px] font-medium rounded border ${
                        GENRE_COLORS[g] || "bg-slate-900 text-slate-400 border-slate-800"
                      }`}
                    >
                      {g}
                    </span>
                  ))}
                </div>
              </div>

              <div>
                {/* Custom Confidence Score Bar */}
                <div className="space-y-1">
                  <div className="flex justify-between items-center text-[10px] font-bold text-slate-500">
                    <span>MATCH CONFIDENCE</span>
                    <span className="text-slate-300">
                      {model === "svd"
                        ? `${movie.score.toFixed(2)} ★`
                        : `${(movie.score * 100).toFixed(1)}%`}
                    </span>
                  </div>
                  <div className="w-full bg-slate-900 rounded-full h-1.5 overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ${
                        movie.debiased_badge ? "bg-teal-500" : "bg-orange-500"
                      }`}
                      style={{
                        width: `${
                          model === "svd"
                            ? Math.min(100, Math.max(0, (movie.score / 5.0) * 100))
                            : Math.min(100, Math.max(0, movie.score * 100))
                        }%`,
                      }}
                    ></div>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
