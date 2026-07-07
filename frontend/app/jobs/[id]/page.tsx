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
    <div className="min-h-screen bg-gray-50">
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

      <main className="max-w-6xl mx-auto px-4 py-8">
        <div className="mb-6 flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Analysis Running</h1>
            <p className="text-gray-500 text-sm mt-1 font-mono">{jobId}</p>
          </div>
          {stream.tokenUsage.length > 0 && (
            <div className="text-right text-sm text-gray-500">
              <div className="font-mono">{totalTokens.toLocaleString()} tokens · ${totalCost.toFixed(4)}</div>
              <div className="text-xs">live LLM usage</div>
            </div>
          )}
        </div>

        {/* Progress bar */}
        <div className="mb-6">
          <div className="flex justify-between text-sm mb-1">
            <span className="text-gray-600">
              {stream.status === "completed"
                ? "Complete!"
                : stream.status === "failed"
                ? "Failed"
                : `Running: ${stream.currentAgent || "starting"}...`}
            </span>
            <span className="font-medium">{stream.progress}%</span>
          </div>
          <div className="h-3 bg-gray-200 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-500 ${
                stream.status === "failed"
                  ? "bg-red-500"
                  : stream.status === "completed"
                  ? "bg-green-500"
                  : "bg-blue-500"
              }`}
              style={{ width: `${stream.progress}%` }}
            />
          </div>
        </div>

        {/* Live agent graph — the "watch the agents think" view */}
        <div className="mb-6">
          <h2 className="font-semibold text-gray-800 mb-2">Agent Graph</h2>
          <AgentGraph
            currentAgent={stream.currentAgent}
            status={stream.status}
            trace={stream.trace}
            activeRetry={stream.activeRetry}
            onSelectAgent={setSelectedAgent}
          />
          {selectedAgent && selectedTrace.length > 0 && (
            <div className="mt-2 bg-indigo-50 border border-indigo-100 rounded-lg p-3 text-xs text-indigo-800">
              <span className="font-semibold capitalize">{selectedAgent}</span>: {selectedTrace.length} trace event(s) — click again to deselect
            </div>
          )}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-3">
            <AgentConsole events={stream.events} trace={stream.trace} />
          </div>
        </div>

        {/* Error state */}
        {stream.status === "failed" && (
          <div className="mt-6 bg-red-50 border border-red-200 rounded-xl p-6">
            <h3 className="font-semibold text-red-700 mb-2">Pipeline Failed</h3>
            <p className="text-red-600 text-sm">{stream.error}</p>
            <button
              onClick={() => router.push("/")}
              className="mt-4 px-4 py-2 bg-red-600 text-white rounded-lg text-sm hover:bg-red-700"
            >
              Try Again
            </button>
          </div>
        )}

        {/* Completed state */}
        {stream.status === "completed" && (
          <div className="mt-6 bg-green-50 border border-green-200 rounded-xl p-8 text-center">
            <div className="text-5xl mb-4">🎉</div>
            <h3 className="text-2xl font-bold text-green-700 mb-2">Analysis Complete!</h3>
            <p className="text-green-600 mb-6">Your ML report is ready to view.</p>
            <button
              onClick={() => router.push(`/jobs/${jobId}/report`)}
              className="px-8 py-3 bg-green-600 hover:bg-green-700 text-white font-semibold rounded-xl text-lg transition-colors"
            >
              View Full Report →
            </button>
          </div>
        )}
      </main>
    </div>
  );
}
