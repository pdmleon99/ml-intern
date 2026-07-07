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
  completed: "bg-green-100 text-green-700",
  running: "bg-blue-100 text-blue-700",
  failed: "bg-red-100 text-red-700",
  queued: "bg-yellow-100 text-yellow-700",
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
    <div className="min-h-screen bg-gray-50">
      <Header />
      <main className="max-w-4xl mx-auto px-4 py-8">
        <div className="flex items-center justify-between mb-6">
          <h1 className="text-2xl font-bold text-gray-900">My Jobs</h1>
          <Link
            href="/"
            className="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700"
          >
            + New Analysis
          </Link>
        </div>

        {loading && (
          <div className="flex items-center justify-center py-12">
            <div className="w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
          </div>
        )}

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 rounded-xl p-4 text-sm">
            {error}
          </div>
        )}

        {!loading && jobs.length === 0 && (
          <div className="text-center py-16 text-gray-400">
            <div className="text-4xl mb-3">🤖</div>
            <p>No jobs yet. Upload a dataset to get started!</p>
          </div>
        )}

        <div className="space-y-3">
          {jobs.map((job) => (
            <Link
              key={job.job_id}
              href={job.status === "completed" ? `/jobs/${job.job_id}/report` : `/jobs/${job.job_id}`}
              className="block bg-white rounded-xl border border-gray-200 p-4 hover:border-blue-300 hover:shadow-sm transition-all"
            >
              <div className="flex items-center justify-between">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-gray-900">{job.dataset_name || "Dataset"}</span>
                    <span
                      className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                        STATUS_COLORS[job.status] ?? "bg-gray-100 text-gray-600"
                      }`}
                    >
                      {job.status}
                    </span>
                    {job.problem_type && (
                      <span className="text-xs text-gray-400">{job.problem_type}</span>
                    )}
                  </div>
                  <p className="text-xs text-gray-400 mt-1 font-mono">{job.job_id.slice(0, 16)}...</p>
                </div>

                <div className="text-right">
                  {job.best_model_score != null && (
                    <div className="text-lg font-bold text-blue-600">
                      {(job.best_model_score * 100).toFixed(1)}%
                    </div>
                  )}
                  {job.status === "running" && (
                    <div className="text-sm text-gray-500">{job.progress_pct}%</div>
                  )}
                  {job.created_at && (
                    <div className="text-xs text-gray-400 mt-1">
                      {new Date(job.created_at).toLocaleDateString()}
                    </div>
                  )}
                </div>
              </div>

              {job.status === "running" && (
                <div className="mt-2 h-1.5 bg-gray-200 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-blue-500 rounded-full transition-all"
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
