import React from "react";
import { User, RefreshCw, Star, Film } from "lucide-react";

export const PRESET_USERS = [
  { id: 1, label: "User 1 (18M, Student - Sci-Fi / Action)" },
  { id: 10, label: "User 10 (35F, Educator - Drama / Romance)" },
  { id: 33, label: "User 33 (50M, Executive - Film-Noir / Mystery)" },
  { id: 115, label: "User 115 (25F, Artist - Comedy / Animation)" },
  { id: 400, label: "User 400 (45M, Engineer - Thriller / Sci-Fi)" },
  { id: 600, label: "User 600 (56F, Retired - Musical / Drama)" },
  { id: 1010, label: "User 1010 (21M, Writer - Adventure / Fantasy)" },
  { id: 2020, label: "User 2020 (30F, Healthcare - Comedy / Romance)" },
  { id: 3030, label: "User 3030 (25M, Sales - Action / Thriller)" },
  { id: 5050, label: "User 5050 (40M, Executive - Drama / Crime)" },
];

export default function UserSelector({ selectedUserId, onUserChange, userHistory, loadingHistory }) {
  const handleRandomize = () => {
    const randomIdx = Math.floor(Math.random() * PRESET_USERS.length);
    onUserChange(PRESET_USERS[randomIdx].id);
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-slate-100 flex items-center gap-2">
          <User className="w-5 h-5 text-orange-500" />
          Select User Persona
        </h3>
        <button
          onClick={handleRandomize}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-300 hover:text-white bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors border border-slate-700"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          Randomize
        </button>
      </div>

      <div className="mb-6">
        <label className="block text-xs font-medium text-slate-400 mb-2">Simulated User ID</label>
        <select
          value={selectedUserId}
          onChange={(e) => onUserChange(parseInt(e.target.value, 10))}
          className="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-3 text-sm text-slate-200 focus:outline-none focus:ring-2 focus:ring-orange-500/50 focus:border-orange-500 cursor-pointer"
        >
          {PRESET_USERS.map((u) => (
            <option key={u.id} value={u.id} className="bg-slate-950">
              {u.label}
            </option>
          ))}
        </select>
      </div>

      {loadingHistory ? (
        <div className="animate-pulse space-y-3">
          <div className="h-4 bg-slate-800 rounded w-1/3"></div>
          <div className="h-20 bg-slate-800 rounded-xl"></div>
        </div>
      ) : userHistory ? (
        <div className="space-y-4">
          <div className="border-t border-slate-800 pt-4">
            <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">User Profile Stats</h4>
            <div className="grid grid-cols-2 gap-3 mb-4">
              <div className="bg-slate-950/50 border border-slate-850 p-3 rounded-xl flex flex-col justify-center">
                <span className="text-xs text-slate-500 mb-1 flex items-center gap-1">
                  <Film className="w-3 h-3 text-slate-400" /> Total Rated
                </span>
                <span className="text-xl font-bold text-slate-100">{userHistory.total_rated_count} films</span>
              </div>
              <div className="bg-slate-950/50 border border-slate-850 p-3 rounded-xl flex flex-col justify-center">
                <span className="text-xs text-slate-500 mb-1 flex items-center gap-1">
                  <Star className="w-3 h-3 text-yellow-500 fill-yellow-500" /> Avg Rating
                </span>
                <span className="text-xl font-bold text-slate-100">{userHistory.avg_rating} / 5</span>
              </div>
            </div>

            <div>
              <span className="text-xs text-slate-500 block mb-2">Preferred Genres</span>
              <div className="flex flex-wrap gap-1.5">
                {userHistory.top_genres && userHistory.top_genres.length > 0 ? (
                  userHistory.top_genres.map((genre) => (
                    <span
                      key={genre}
                      className="px-2.5 py-1 text-xs font-medium rounded-full bg-orange-950/30 text-orange-400 border border-orange-900/30"
                    >
                      {genre}
                    </span>
                  ))
                ) : (
                  <span className="text-xs text-slate-600">None detected</span>
                )}
              </div>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
