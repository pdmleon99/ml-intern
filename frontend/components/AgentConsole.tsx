"use client";

import { useEffect, useRef } from "react";
import { StreamEvent, TraceEvent } from "@/lib/useJobStream";

interface Props {
  events: StreamEvent[];
  trace: TraceEvent[];
}

type Line = { key: string; ts: number; agent: string; render: () => JSX.Element };

const AGENT_COLORS: Record<string, string> = {
  eda: "text-sky-400 bg-sky-400/10",
  planner: "text-violet-400 bg-violet-400/10",
  features: "text-fuchsia-400 bg-fuchsia-400/10",
  experiments: "text-amber-400 bg-amber-400/10",
  critic: "text-pink-400 bg-pink-400/10",
  evaluation: "text-emerald-400 bg-emerald-400/10",
  report: "text-rose-400 bg-rose-400/10",
};

function AgentTag({ agent }: { agent: string }) {
  return (
    <span
      className={`flex-shrink-0 w-[76px] text-center text-[10px] font-semibold uppercase tracking-wider rounded px-1.5 py-0.5 ${
        AGENT_COLORS[agent] ?? "text-slate-400 bg-slate-400/10"
      }`}
    >
      {agent || "—"}
    </span>
  );
}

function formatTime(ts: number) {
  return new Date(ts).toLocaleTimeString([], { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export function AgentConsole({ events, trace }: Props) {
  const logRef = useRef<HTMLDivElement>(null);

  const lines: Line[] = [
    ...events.map((e, i) => ({
      key: `log-${i}`,
      ts: new Date(e.timestamp).getTime(),
      agent: e.agent,
      render: () => (
        <div className="flex gap-3">
          <AgentTag agent={e.agent} />
          <span className="text-slate-300">{e.message}</span>
        </div>
      ),
    })),
    ...trace.map((t, i) => ({
      key: `trace-${i}`,
      ts: new Date(t.ts).getTime(),
      agent: t.agent,
      render: () => renderTrace(t),
    })),
  ].sort((a, b) => a.ts - b.ts);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [lines.length]);

  function renderTrace(t: TraceEvent) {
    if (t.kind === "thought") {
      return (
        <div className="flex gap-3 italic">
          <AgentTag agent={t.agent} />
          <span className="text-indigo-300">💭 {t.payload?.reasoning}</span>
        </div>
      );
    }
    if (t.kind === "plan") {
      return (
        <div className="flex gap-3">
          <AgentTag agent={t.agent} />
          <span className="text-violet-300">
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
        <div className="flex gap-3">
          <AgentTag agent={t.agent} />
          <span className={isRetry ? "text-amber-300" : "text-emerald-300"}>
            🔬 verdict: <b>{t.payload?.verdict}</b> ({t.payload?.confidence}) — {t.payload?.reasoning}
          </span>
        </div>
      );
    }
    if (t.kind === "retry") {
      return (
        <div className="flex gap-3 font-semibold">
          <AgentTag agent={t.agent} />
          <span className="text-rose-400">
            ↺ retrying: {t.payload?.from} → {t.payload?.to} — {t.payload?.reason}
          </span>
        </div>
      );
    }
    if (t.kind === "token_usage") {
      return (
        <div className="flex gap-3 text-slate-500">
          <AgentTag agent={t.agent} />
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
    <div className="bg-slate-900 rounded-2xl border border-slate-800 h-full flex flex-col shadow-card overflow-hidden">
      <div className="px-4 py-3 border-b border-slate-800 flex items-center justify-between bg-slate-900/80">
        <div className="flex items-center gap-3">
          <div className="flex gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-rose-500/70" />
            <span className="w-2.5 h-2.5 rounded-full bg-amber-500/70" />
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-500/70" />
          </div>
          <h2 className="font-semibold text-slate-200 text-sm">Live Agent Console</h2>
        </div>
        <span className="text-xs text-slate-500 font-mono">{lines.length} events</span>
      </div>
      <div ref={logRef} className="flex-1 p-4 overflow-y-auto font-mono text-[12.5px] leading-relaxed space-y-1.5 max-h-96">
        {lines.length === 0 ? (
          <p className="text-slate-500 italic">Waiting for agents to start...</p>
        ) : (
          lines.map((l) => (
            <div key={l.key} className="flex gap-2 group">
              <span className="text-slate-600 text-[10px] w-[64px] flex-shrink-0 pt-0.5 group-hover:text-slate-500">
                {formatTime(l.ts)}
              </span>
              {l.render()}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
