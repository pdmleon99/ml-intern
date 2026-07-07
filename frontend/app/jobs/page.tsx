"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Header } from "@/components/Header";
import { listJobs } from "@/lib/api";

interface JobSummary {
  job_id: string;
  status: string;
  progress_pct: number;
  dataset_name: string;
  problem_type: string | null;
  best_model_score: number | null;
  created_at: string | null;
}

const STATUS_COLORS: Record<string, string> = {
  completed: "bg-emerald-100 text-emerald-700",
  running: "bg-brand-100 text-brand-700",
  failed: "bg-rose-100 text-rose-700",
  queued: "bg-amber-100 text-amber-700",
};

export default function JobsPage() {
  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listJobs()
      .then(setJobs)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-white">
      <Header />
      <main className="max-w-4xl mx-auto px-4 py-10">
        <div className="flex items-center justify-between mb-8">
          <div>
            <p className="text-xs font-semibold text-brand-600 uppercase tracking-wider mb-1">History</p>
            <h1 className="text-3xl font-bold text-slate-900 tracking-tight">My Jobs</h1>
          </div>
          <Link
            href="/"
            className="px-4 py-2.5 bg-gradient-to-br from-brand-600 to-violet-600 hover:from-brand-700 hover:to-violet-700 text-white rounded-xl text-sm font-semibold shadow-soft transition-colors"
          >
            + New Analysis
          </Link>
        </div>

        {loading && (
          <div className="flex items-center justify-center py-12">
            <div className="w-6 h-6 border-2 border-brand-500 border-t-transparent rounded-full animate-spin" />
          </div>
        )}

        {error && (
          <div className="bg-rose-50 border border-rose-200 text-rose-700 rounded-2xl p-4 text-sm">
            {error}
          </div>
        )}

        {!loading && jobs.length === 0 && (
          <div className="text-center py-20 text-slate-400">
            <div className="text-4xl mb-3">🤖</div>
            <p>No jobs yet. Upload a dataset to get started!</p>
          </div>
        )}

        <div className="space-y-3">
          {jobs.map((job) => (
            <Link
              key={job.job_id}
              href={job.status === "completed" ? `/jobs/${job.job_id}/report` : `/jobs/${job.job_id}`}
              className="block bg-white rounded-2xl border border-slate-200 shadow-soft p-4 hover:border-brand-300 hover:shadow-card transition-all"
            >
              <div className="flex items-center justify-between">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-slate-900">{job.dataset_name || "Dataset"}</span>
                    <span
                      className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                        STATUS_COLORS[job.status] ?? "bg-slate-100 text-slate-600"
                      }`}
                    >
                      {job.status}
                    </span>
                    {job.problem_type && (
                      <span className="text-xs text-slate-400">{job.problem_type}</span>
                    )}
                  </div>
                  <p className="text-xs text-slate-400 mt-1 font-mono">{job.job_id.slice(0, 16)}...</p>
                </div>

                <div className="text-right">
                  {job.best_model_score != null && (
                    <div className="text-lg font-bold text-brand-600 tracking-tight">
                      {(job.best_model_score * 100).toFixed(1)}%
                    </div>
                  )}
                  {job.status === "running" && (
                    <div className="text-sm text-slate-500">{job.progress_pct}%</div>
                  )}
                  {job.created_at && (
                    <div className="text-xs text-slate-400 mt-1">
                      {new Date(job.created_at).toLocaleDateString()}
                    </div>
                  )}
                </div>
              </div>

              {job.status === "running" && (
                <div className="mt-3 h-1.5 bg-slate-100 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-brand-500 to-violet-500 rounded-full transition-all"
                    style={{ width: `${job.progress_pct}%` }}
                  />
                </div>
              )}
            </Link>
          ))}
        </div>
      </main>
    </div>
  );
}
