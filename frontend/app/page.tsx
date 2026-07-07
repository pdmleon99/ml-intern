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
    <div className="min-h-screen bg-gray-50">
      <Header />

      <main className="max-w-4xl mx-auto px-4 py-10">
        <div className="text-center mb-10">
          <h1 className="text-4xl font-bold text-gray-900 mb-3">
            Autonomous Data Science
          </h1>
          <p className="text-lg text-gray-600">
            Upload any tabular dataset and get automated EDA, feature engineering,
            model training, and a PDF report.
          </p>
        </div>

        {/* Tab selector */}
        <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6 mb-6">
          <div className="flex gap-1 mb-6 bg-gray-100 rounded-lg p-1">
            {(["upload", "huggingface", "kaggle"] as Tab[]).map((t) => (
              <button
                key={t}
                onClick={() => { setTab(t); setPreview(null); setError(null); }}
                className={`flex-1 py-2 text-sm font-medium rounded-md transition-all ${
                  tab === t ? "bg-white shadow text-blue-600" : "text-gray-600 hover:text-gray-900"
                }`}
              >
                {t === "upload" ? "📁 Upload" : t === "huggingface" ? "🤗 HuggingFace" : "🏆 Kaggle"}
              </button>
            ))}
          </div>

          {tab === "upload" && (
            <div>
              <div
                className={`border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-all ${
                  dragOver ? "border-blue-400 bg-blue-50" : "border-gray-300 hover:border-gray-400"
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
                <p className="text-gray-600 font-medium">
                  Drop your dataset here or <span className="text-blue-500">click to browse</span>
                </p>
                <p className="text-sm text-gray-400 mt-1">
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
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Dataset ID
                </label>
                <input
                  value={hfId}
                  onChange={(e) => setHfId(e.target.value)}
                  placeholder="e.g. scikit-learn/iris"
                  className="w-full px-4 py-2.5 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
              <div className="flex gap-4">
                <div className="flex-1">
                  <label className="block text-sm font-medium text-gray-700 mb-1">Split</label>
                  <input
                    value={hfSplit}
                    onChange={(e) => setHfSplit(e.target.value)}
                    className="w-full px-4 py-2.5 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
              </div>
              <button
                onClick={handleHFLoad}
                disabled={loading}
                className="px-6 py-2.5 bg-orange-500 hover:bg-orange-600 text-white rounded-lg font-medium"
              >
                {loading ? "Loading..." : "Load Dataset →"}
              </button>
            </div>
          )}

          {tab === "kaggle" && (
            <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-6 text-center">
              <p className="text-yellow-800 text-sm">
                Kaggle integration requires KAGGLE_USERNAME and KAGGLE_KEY environment variables.
                Use the HuggingFace tab or direct upload instead.
              </p>
            </div>
          )}
        </div>

        {/* Loading indicator */}
        {loading && (
          <div className="flex items-center justify-center gap-2 py-4 text-gray-500">
            <div className="w-5 h-5 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
            Processing...
          </div>
        )}

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 rounded-xl p-4 mb-4 text-sm">
            {error}
          </div>
        )}

        {/* Dataset preview */}
        {preview && (
          <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6 mb-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="font-semibold text-gray-900">Dataset Preview</h2>
              <span className="text-sm text-gray-500">
                {preview.n_rows.toLocaleString()} rows × {preview.columns.length} columns
              </span>
            </div>
            <div className="overflow-x-auto rounded-lg border border-gray-200">
              <table className="text-xs w-full">
                <thead className="bg-gray-50">
                  <tr>
                    {preview.columns.slice(0, 12).map((col) => (
                      <th key={col} className="px-3 py-2 text-left font-medium text-gray-600 whitespace-nowrap">
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {preview.preview.slice(1, 6).map((row, i) => (
                    <tr key={i} className={i % 2 === 0 ? "bg-white" : "bg-gray-50"}>
                      {row.slice(0, 12).map((cell, j) => (
                        <td key={j} className="px-3 py-1.5 text-gray-700 whitespace-nowrap max-w-xs truncate">
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
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Describe your prediction goal <span className="text-red-500">*</span>
                </label>
                <textarea
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder="e.g. Predict whether a customer will churn based on their usage patterns"
                  rows={2}
                  className="w-full px-4 py-2.5 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Target column{" "}
                  <span className="text-gray-400 font-normal">(leave empty to auto-detect)</span>
                </label>
                <select
                  value={targetCol}
                  onChange={(e) => setTargetCol(e.target.value)}
                  className="w-full px-4 py-2.5 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="">Auto-detect from description</option>
                  {preview.columns.map((col) => (
                    <option key={col} value={col}>{col}</option>
                  ))}
                </select>
              </div>

              <button
                onClick={handleStartAnalysis}
                disabled={loading || !description.trim()}
                className="w-full py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-300 text-white font-semibold rounded-lg transition-colors flex items-center justify-center gap-2"
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
