import React from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from "recharts";
import { Scale, Info, Award } from "lucide-react";

export default function BiasVisualizer() {
  // Recommendation frequency mock data illustrating popularity bias correction
  const dataWithoutIPS = [
    { name: "Toy Story (Blockbuster)", freq: 48, fill: "#ef4444" },
    { name: "Star Wars IV (Blockbuster)", freq: 42, fill: "#ef4444" },
    { name: "Jurassic Park (Blockbuster)", freq: 36, fill: "#ef4444" },
    { name: "Titanic (Blockbuster)", freq: 30, fill: "#ef4444" },
    { name: "Forrest Gump (Blockbuster)", freq: 24, fill: "#ef4444" },
    { name: "Run Lola Run (Indie)", freq: 3, fill: "#f97316" },
    { name: "Waking Life (Indie)", freq: 2, fill: "#f97316" },
    { name: "Chungking Express (Indie)", freq: 1, fill: "#f97316" },
    { name: "Pi (Indie)", freq: 0, fill: "#f97316" },
    { name: "Cube (Indie)", freq: 0, fill: "#f97316" },
  ];

  const dataWithIPS = [
    { name: "Toy Story (Blockbuster)", freq: 20, fill: "#38bdf8" },
    { name: "Star Wars IV (Blockbuster)", freq: 18, fill: "#38bdf8" },
    { name: "Jurassic Park (Blockbuster)", freq: 16, fill: "#38bdf8" },
    { name: "Titanic (Blockbuster)", freq: 14, fill: "#38bdf8" },
    { name: "Forrest Gump (Blockbuster)", freq: 12, fill: "#38bdf8" },
    { name: "Run Lola Run (Indie)", freq: 11, fill: "#2dd4bf" },
    { name: "Waking Life (Indie)", freq: 10, fill: "#2dd4bf" },
    { name: "Chungking Express (Indie)", freq: 9, fill: "#2dd4bf" },
    { name: "Pi (Indie)", freq: 8, fill: "#2dd4bf" },
    { name: "Cube (Indie)", freq: 7, fill: "#2dd4bf" },
  ];

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-5">
        <div>
          <h2 className="text-xl font-bold text-slate-100 flex items-center gap-2">
            <Scale className="w-5 h-5 text-orange-500" />
            Popularity Bias & Causal Debiasing (IPS)
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Visualizing recommendation frequencies for blockbusters vs long-tail indie films
          </p>
        </div>

        {/* Gini Coefficient comparisons */}
        <div className="flex gap-4">
          <div className="bg-slate-950 border border-slate-850 px-4 py-2 rounded-xl text-center">
            <span className="text-[10px] text-slate-500 block leading-none font-bold uppercase mb-1">Standard Gini</span>
            <span className="text-sm font-extrabold text-red-400 leading-none">0.76 (High Inequality)</span>
          </div>
          <div className="bg-slate-950 border border-slate-850 px-4 py-2 rounded-xl text-center">
            <span className="text-[10px] text-slate-500 block leading-none font-bold uppercase mb-1">Debiased Gini</span>
            <span className="text-sm font-extrabold text-teal-400 leading-none">0.31 (Highly Balanced)</span>
          </div>
        </div>
      </div>

      <div className="bg-slate-950 border border-slate-850 rounded-xl p-4 flex gap-2 items-start text-xs text-slate-400">
        <Info className="w-4 h-4 text-orange-500 shrink-0 mt-0.5" />
        <p>
          <strong>Popularity Bias:</strong> Standard collaborative filtering models over-recommend popular "blockbusters" because they dominate ratings.
          <strong> Inverse Propensity Scoring (IPS)</strong> corrects this causal confounder by scaling loss weights inversely to item observation frequency.
          As shown below, IPS successfully surfaces the "long-tail" indie movies, providing a much more diverse and personalized recommendation spread.
        </p>
      </div>

      {/* Side by Side Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Without IPS */}
        <div className="space-y-4">
          <div className="flex items-center justify-between border-b border-slate-900 pb-2">
            <h3 className="text-xs font-bold uppercase tracking-wider text-red-400">Without IPS (Standard Collaborative)</h3>
            <span className="text-[10px] text-slate-500 font-bold">Gini: 0.76</span>
          </div>
          <div className="h-[340px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={dataWithoutIPS}
                layout="vertical"
                margin={{ top: 5, right: 30, left: 40, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
                <XAxis type="number" tick={{ fill: "#64748b", fontSize: 9 }} label={{ value: "Recommendation Count", position: "insideBottom", fill: "#64748b", fontSize: 10, offset: -5 }} />
                <YAxis type="category" dataKey="name" tick={{ fill: "#94a3b8", fontSize: 9 }} width={120} />
                <Tooltip
                  contentStyle={{ backgroundColor: "#0f172a", borderColor: "#1e293b", borderRadius: "12px" }}
                  itemStyle={{ fontSize: "11px" }}
                />
                <Bar dataKey="freq" radius={[0, 4, 4, 0]} barSize={16}>
                  {dataWithoutIPS.map((entry, index) => (
                    <Bar key={`cell-${index}`} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* With IPS */}
        <div className="space-y-4">
          <div className="flex items-center justify-between border-b border-slate-900 pb-2">
            <h3 className="text-xs font-bold uppercase tracking-wider text-teal-400">With IPS (Causal Debiased)</h3>
            <span className="text-[10px] text-slate-500 font-bold">Gini: 0.31</span>
          </div>
          <div className="h-[340px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={dataWithIPS}
                layout="vertical"
                margin={{ top: 5, right: 30, left: 40, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
                <XAxis type="number" tick={{ fill: "#64748b", fontSize: 9 }} label={{ value: "Recommendation Count", position: "insideBottom", fill: "#64748b", fontSize: 10, offset: -5 }} />
                <YAxis type="category" dataKey="name" tick={{ fill: "#94a3b8", fontSize: 9 }} width={120} />
                <Tooltip
                  contentStyle={{ backgroundColor: "#0f172a", borderColor: "#1e293b", borderRadius: "12px" }}
                  itemStyle={{ fontSize: "11px" }}
                />
                <Bar dataKey="freq" radius={[0, 4, 4, 0]} barSize={16}>
                  {dataWithIPS.map((entry, index) => (
                    <Bar key={`cell-${index}`} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
