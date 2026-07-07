"use client";

import { useState, useRef } from "react";
import { useRouter } from "next/navigation";
import { Header } from "@/components/Header";
import { uploadDataset, loadFromHuggingFace, createJob } from "@/lib/api";

type Tab = "upload" | "huggingface" | "kaggle";

export default function HomePage() {
  const router = useRouter();
  const [tab, setTab] = useState<Tab>("upload");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<{
    file_path: string;
    columns: string[];
    n_rows: number;
    preview: string[][];
  } | null>(null);

  // Upload
  const fileRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  // HuggingFace
  const [hfId, setHfId] = useState("scikit-learn/iris");
  const [hfSplit, setHfSplit] = useState("train");

  // Form
  const [description, setDescription] = useState("");
  const [targetCol, setTargetCol] = useState("");

  async function handleFileSelect(file: File) {
    setLoading(true);
    setError(null);
    try {
      const result = await uploadDataset(file);
      setPreview(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setLoading(false);
    }
  }

  async function handleHFLoad() {
    if (!hfId.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const result = await loadFromHuggingFace(hfId.trim(), hfSplit);
      setPreview(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : "HuggingFace load failed");
    } finally {
      setLoading(false);
    }
  }

  async function handleStartAnalysis() {
    if (!preview || !description.trim()) {
      setError("Please provide a dataset and describe your prediction goal.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const result = await createJob({
        dataset_source: "upload",
        uploaded_file_path: preview.file_path,
        user_description: description.trim(),
        target_column: targetCol.trim() || undefined,
      });
      router.push(`/jobs/${result.job_id}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to start analysis");
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-white">
      <Header />

      <main className="max-w-4xl mx-auto px-4 py-14">
        <div className="text-center mb-12">
          <div className="inline-flex items-center gap-1.5 text-xs font-semibold text-brand-600 bg-brand-50 border border-brand-100 rounded-full px-3 py-1 mb-4 uppercase tracking-wide">
            <span className="w-1.5 h-1.5 rounded-full bg-brand-500 animate-pulse" />
            Self-correcting multi-agent pipeline
          </div>
          <h1 className="text-4xl md:text-5xl font-bold text-slate-900 mb-4 tracking-tight">
            Autonomous Data Science
          </h1>
          <p className="text-lg text-slate-500 max-w-2xl mx-auto leading-relaxed">
            Upload any tabular dataset — a Planner, Critic, and Evaluator agent plan a strategy,
            train models, critique their own results, and hand you a report with statistical
            confidence, not just a bare accuracy number.
          </p>
        </div>

        {/* Tab selector */}
        <div className="bg-white rounded-2xl shadow-soft border border-slate-200 p-6 mb-6">
          <div className="flex gap-1 mb-6 bg-slate-100/80 rounded-xl p-1">
            {(["upload", "huggingface", "kaggle"] as Tab[]).map((t) => (
              <button
                key={t}
                onClick={() => { setTab(t); setPreview(null); setError(null); }}
                className={`flex-1 py-2 text-sm font-medium rounded-lg transition-all ${
                  tab === t ? "bg-white shadow-soft text-brand-700" : "text-slate-500 hover:text-slate-800"
                }`}
              >
                {t === "upload" ? "📁 Upload" : t === "huggingface" ? "🤗 HuggingFace" : "🏆 Kaggle"}
              </button>
            ))}
          </div>

          {tab === "upload" && (
            <div>
              <div
                className={`border-2 border-dashed rounded-2xl p-10 text-center cursor-pointer transition-all ${
                  dragOver ? "border-brand-400 bg-brand-50" : "border-slate-200 hover:border-slate-300 hover:bg-slate-50/50"
                }`}
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setDragOver(false);
                  const f = e.dataTransfer.files[0];
                  if (f) handleFileSelect(f);
                }}
                onClick={() => fileRef.current?.click()}
              >
                <div className="text-4xl mb-3">📊</div>
                <p className="text-slate-600 font-medium">
                  Drop your dataset here or <span className="text-brand-600">click to browse</span>
                </p>
                <p className="text-sm text-slate-400 mt-1">
                  CSV, Parquet, Excel, JSON, TSV, ZIP(CSV) — up to 500MB
                </p>
                <input
                  ref={fileRef}
                  type="file"
                  className="hidden"
                  accept=".csv,.parquet,.xlsx,.xls,.json,.tsv,.zip"
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) handleFileSelect(f);
                  }}
                />
              </div>
            </div>
          )}

          {tab === "huggingface" && (
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">
                  Dataset ID
                </label>
                <input
                  value={hfId}
                  onChange={(e) => setHfId(e.target.value)}
                  placeholder="e.g. scikit-learn/iris"
                  className="w-full px-4 py-2.5 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-brand-500"
                />
              </div>
              <div className="flex gap-4">
                <div className="flex-1">
                  <label className="block text-sm font-medium text-slate-700 mb-1">Split</label>
                  <input
                    value={hfSplit}
                    onChange={(e) => setHfSplit(e.target.value)}
                    className="w-full px-4 py-2.5 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-brand-500"
                  />
                </div>
              </div>
              <button
                onClick={handleHFLoad}
                disabled={loading}
                className="px-6 py-2.5 bg-gradient-to-br from-amber-500 to-orange-500 hover:from-amber-600 hover:to-orange-600 text-white rounded-xl font-medium shadow-soft"
              >
                {loading ? "Loading..." : "Load Dataset →"}
              </button>
            </div>
          )}

          {tab === "kaggle" && (
            <div className="bg-amber-50 border border-amber-200 rounded-2xl p-6 text-center">
              <p className="text-amber-800 text-sm">
                Kaggle integration requires KAGGLE_USERNAME and KAGGLE_KEY environment variables.
                Use the HuggingFace tab or direct upload instead.
              </p>
            </div>
          )}
        </div>

        {/* Loading indicator */}
        {loading && (
          <div className="flex items-center justify-center gap-2 py-4 text-slate-500">
            <div className="w-5 h-5 border-2 border-brand-500 border-t-transparent rounded-full animate-spin" />
            Processing...
          </div>
        )}

        {error && (
          <div className="bg-rose-50 border border-rose-200 text-rose-700 rounded-2xl p-4 mb-4 text-sm">
            {error}
          </div>
        )}

        {/* Dataset preview */}
        {preview && (
          <div className="bg-white rounded-2xl shadow-soft border border-slate-200 p-6 mb-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="font-semibold text-slate-900">Dataset Preview</h2>
              <span className="text-sm text-slate-500">
                {preview.n_rows.toLocaleString()} rows × {preview.columns.length} columns
              </span>
            </div>
            <div className="overflow-x-auto rounded-xl border border-slate-200">
              <table className="text-xs w-full">
                <thead className="bg-slate-50">
                  <tr>
                    {preview.columns.slice(0, 12).map((col) => (
                      <th key={col} className="px-3 py-2 text-left font-medium text-slate-500 whitespace-nowrap uppercase tracking-wide">
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {preview.preview.slice(1, 6).map((row, i) => (
                    <tr key={i} className={i % 2 === 0 ? "bg-white" : "bg-slate-50/60"}>
                      {row.slice(0, 12).map((cell, j) => (
                        <td key={j} className="px-3 py-1.5 text-slate-600 whitespace-nowrap max-w-xs truncate">
                          {cell}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Analysis form */}
            <div className="mt-6 space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">
                  Describe your prediction goal <span className="text-rose-500">*</span>
                </label>
                <textarea
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder="e.g. Predict whether a customer will churn based on their usage patterns"
                  rows={2}
                  className="w-full px-4 py-2.5 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-brand-500 resize-none"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">
                  Target column{" "}
                  <span className="text-amber-600 font-normal">
                    (recommended — auto-detect can guess wrong)
                  </span>
                </label>
                <select
                  value={targetCol}
                  onChange={(e) => setTargetCol(e.target.value)}
                  className={`w-full px-4 py-2.5 border rounded-xl focus:outline-none focus:ring-2 focus:ring-brand-500 ${
                    targetCol ? "border-emerald-300 bg-emerald-50" : "border-amber-300 bg-amber-50"
                  }`}
                >
                  <option value="">⚠ Auto-detect from description (less reliable)</option>
                  {preview.columns.map((col) => (
                    <option key={col} value={col}>{col}</option>
                  ))}
                </select>
                {!targetCol && (
                  <p className="text-xs text-amber-700 mt-1">
                    Pick the exact column you want to predict — the agent works with any
                    column, but a wrong guess here means a wrong analysis.
                  </p>
                )}
              </div>

              <button
                onClick={handleStartAnalysis}
                disabled={loading || !description.trim()}
                className="w-full py-3.5 bg-gradient-to-br from-brand-600 to-violet-600 hover:from-brand-700 hover:to-violet-700 disabled:from-slate-300 disabled:to-slate-300 text-white font-semibold rounded-xl transition-colors flex items-center justify-center gap-2 shadow-soft"
              >
                {loading ? (
                  <><div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />Starting...</>
                ) : (
                  "Start Analysis →"
                )}
              </button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
