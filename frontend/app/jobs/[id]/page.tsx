"use client";

import { useParams, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { Header } from "@/components/Header";
import { AgentGraph } from "@/components/AgentGraph";
import { AgentConsole } from "@/components/AgentConsole";
import { useJobStream } from "@/lib/useJobStream";

export default function JobPage() {
  const params = useParams<{ id: string }>();
  const router = useRouter();
  const jobId = params.id;
  const stream = useJobStream(jobId);
  const [showConfetti, setShowConfetti] = useState(false);
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);

  useEffect(() => {
    if (stream.status === "completed") {
      setShowConfetti(true);
      setTimeout(() => setShowConfetti(false), 4000);
    }
  }, [stream.status]);

  const totalTokens = stream.tokenUsage.reduce(
    (s, t) => s + (t.input_tokens ?? 0) + (t.output_tokens ?? 0), 0
  );
  const totalCost = stream.tokenUsage.reduce((s, t) => s + (t.cost_usd ?? 0), 0);

  const selectedTrace = selectedAgent
    ? stream.trace.filter((t) => t.agent === selectedAgent)
    : [];

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-white">
      <Header />

      {showConfetti && (
        <div className="fixed inset-0 pointer-events-none z-50 overflow-hidden">
          {Array.from({ length: 40 }).map((_, i) => (
            <div
              key={i}
              className="absolute text-2xl"
              style={{
                left: `${Math.random() * 100}%`,
                animation: `confetti-fall ${1.5 + Math.random() * 2}s ease-in ${Math.random() * 1}s forwards`,
              }}
            >
              {["🎉", "✨", "🌟", "🎊"][Math.floor(Math.random() * 4)]}
            </div>
          ))}
        </div>
      )}

      <main className="max-w-6xl mx-auto px-4 py-10">
        <div className="mb-8 flex items-start justify-between">
          <div>
            <p className="text-xs font-semibold text-brand-600 uppercase tracking-wider mb-1">
              Agent run
            </p>
            <h1 className="text-3xl font-bold text-slate-900 tracking-tight">Analysis in Progress</h1>
            <p className="text-slate-400 text-xs mt-1.5 font-mono">{jobId}</p>
          </div>
          {stream.tokenUsage.length > 0 && (
            <div className="text-right bg-white border border-slate-200 rounded-xl px-4 py-2.5 shadow-soft">
              <div className="font-mono text-sm font-semibold text-slate-800">
                {totalTokens.toLocaleString()} tok · ${totalCost.toFixed(4)}
              </div>
              <div className="text-[11px] text-slate-400 uppercase tracking-wide">live LLM usage</div>
            </div>
          )}
        </div>

        {/* Progress bar */}
        <div className="mb-8">
          <div className="flex justify-between text-sm mb-2">
            <span className="font-medium text-slate-600">
              {stream.status === "completed"
                ? "Complete!"
                : stream.status === "failed"
                ? "Failed"
                : `Running: ${stream.currentAgent || "starting"}…`}
            </span>
            <span className="font-semibold text-slate-800">{stream.progress}%</span>
          </div>
          <div className="h-2.5 bg-slate-100 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-700 ease-out ${
                stream.status === "failed"
                  ? "bg-rose-500"
                  : stream.status === "completed"
                  ? "bg-emerald-500"
                  : "bg-gradient-to-r from-brand-500 to-violet-500"
              }`}
              style={{ width: `${stream.progress}%` }}
            />
          </div>
        </div>

        {/* Live agent graph — the "watch the agents think" view */}
        <div className="mb-6">
          <h2 className="font-semibold text-slate-800 mb-3 text-sm uppercase tracking-wide">
            Agent Graph
          </h2>
          <AgentGraph
            currentAgent={stream.currentAgent}
            status={stream.status}
            trace={stream.trace}
            activeRetry={stream.activeRetry}
            onSelectAgent={setSelectedAgent}
          />
          {selectedAgent && selectedTrace.length > 0 && (
            <div className="mt-2 bg-brand-50 border border-brand-100 rounded-lg p-3 text-xs text-brand-800">
              <span className="font-semibold capitalize">{selectedAgent}</span>: {selectedTrace.length} trace event(s) — click again to deselect
            </div>
          )}
        </div>

        <div className="mb-6">
          <AgentConsole events={stream.events} trace={stream.trace} />
        </div>

        {/* Error state */}
        {stream.status === "failed" && (
          <div className="mt-6 bg-rose-50 border border-rose-200 rounded-2xl p-6">
            <h3 className="font-semibold text-rose-700 mb-2">Pipeline Failed</h3>
            <p className="text-rose-600 text-sm">{stream.error}</p>
            <button
              onClick={() => router.push("/")}
              className="mt-4 px-4 py-2 bg-rose-600 text-white rounded-lg text-sm hover:bg-rose-700 transition-colors"
            >
              Try Again
            </button>
          </div>
        )}

        {/* Completed state */}
        {stream.status === "completed" && (
          <div className="mt-6 bg-gradient-to-br from-emerald-50 to-teal-50 border border-emerald-200 rounded-2xl p-10 text-center">
            <div className="text-5xl mb-4">🎉</div>
            <h3 className="text-2xl font-bold text-emerald-800 mb-2">Analysis Complete!</h3>
            <p className="text-emerald-700/80 mb-6">Your ML report is ready to view.</p>
            <button
              onClick={() => router.push(`/jobs/${jobId}/report`)}
              className="px-8 py-3 bg-emerald-600 hover:bg-emerald-700 text-white font-semibold rounded-xl text-lg transition-colors shadow-soft"
            >
              View Full Report →
            </button>
          </div>
        )}
      </main>
    </div>
  );
}
