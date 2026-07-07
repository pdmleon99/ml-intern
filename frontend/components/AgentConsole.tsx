"use client";

import { useEffect, useRef } from "react";
import { StreamEvent, TraceEvent } from "@/lib/useJobStream";

interface Props {
  events: StreamEvent[];
  trace: TraceEvent[];
}

type Line = { key: string; ts: number; render: () => JSX.Element };

const AGENT_COLORS: Record<string, string> = {
  eda: "text-blue-500",
  planner: "text-violet-500",
  features: "text-purple-500",
  experiments: "text-orange-500",
  critic: "text-pink-600",
  evaluation: "text-green-500",
  report: "text-pink-500",
};

export function AgentConsole({ events, trace }: Props) {
  const logRef = useRef<HTMLDivElement>(null);

  const lines: Line[] = [
    ...events.map((e, i) => ({
      key: `log-${i}`,
      ts: new Date(e.timestamp).getTime(),
      render: () => (
        <div className="flex gap-2">
          <span className={`flex-shrink-0 w-20 ${AGENT_COLORS[e.agent] ?? "text-gray-400"}`}>[{e.agent}]</span>
          <span className="text-gray-700">{e.message}</span>
        </div>
      ),
    })),
    ...trace.map((t, i) => ({
      key: `trace-${i}`,
      ts: new Date(t.ts).getTime(),
      render: () => renderTrace(t),
    })),
  ].sort((a, b) => a.ts - b.ts);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [lines.length]);

  function renderTrace(t: TraceEvent) {
    const color = AGENT_COLORS[t.agent] ?? "text-gray-400";
    if (t.kind === "thought") {
      return (
        <div className="flex gap-2 italic">
          <span className={`flex-shrink-0 w-20 ${color}`}>[{t.agent}]</span>
          <span className="text-indigo-600">💭 {t.payload?.reasoning}</span>
        </div>
      );
    }
    if (t.kind === "plan") {
      return (
        <div className="flex gap-2">
          <span className={`flex-shrink-0 w-20 ${color}`}>[{t.agent}]</span>
          <span className="text-violet-700">
            🧠 plan → train {JSON.stringify(t.payload?.model_shortlist ?? [])}
            {t.payload?.columns_to_drop?.length > 0 &&
              ` · drop [${t.payload.columns_to_drop.map((d: any) => d.column).join(", ")}]`}
          </span>
        </div>
      );
    }
    if (t.kind === "critic") {
      const isRetry = String(t.payload?.verdict).startsWith("retry");
      return (
        <div className="flex gap-2">
          <span className={`flex-shrink-0 w-20 ${color}`}>[{t.agent}]</span>
          <span className={isRetry ? "text-amber-600" : "text-emerald-700"}>
            🔬 verdict: {t.payload?.verdict} ({t.payload?.confidence}) — {t.payload?.reasoning}
          </span>
        </div>
      );
    }
    if (t.kind === "retry") {
      return (
        <div className="flex gap-2 font-semibold">
          <span className={`flex-shrink-0 w-20 ${color}`}>[{t.agent}]</span>
          <span className="text-red-500">
            ↺ retrying: {t.payload?.from} → {t.payload?.to} — {t.payload?.reason}
          </span>
        </div>
      );
    }
    if (t.kind === "token_usage") {
      return (
        <div className="flex gap-2 text-gray-400">
          <span className="flex-shrink-0 w-20">[{t.agent}]</span>
          <span>
            {t.payload?.input_tokens}+{t.payload?.output_tokens} tok · ${t.payload?.cost_usd?.toFixed(5)} ·{" "}
            {t.payload?.latency_s?.toFixed(2)}s
          </span>
        </div>
      );
    }
    return <span />;
  }

  return (
    <div className="bg-white rounded-xl border border-gray-200 h-full flex flex-col">
      <div className="p-4 border-b border-gray-100 flex items-center justify-between">
        <h2 className="font-semibold text-gray-800">Live Agent Console</h2>
        <span className="text-xs text-gray-400">{lines.length} events</span>
      </div>
      <div ref={logRef} className="flex-1 p-4 overflow-y-auto font-mono text-xs space-y-1 max-h-96">
        {lines.length === 0 ? (
          <p className="text-gray-400 italic">Waiting for agents to start...</p>
        ) : (
          lines.map((l) => <div key={l.key}>{l.render()}</div>)
        )}
      </div>
    </div>
  );
}
