"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { getJob, streamJobEvents } from "./api";

export interface StreamEvent {
  agent: string;
  progress_pct: number;
  message: string;
  timestamp: string;
}

export interface TraceEvent {
  kind: "thought" | "plan" | "critic" | "retry" | "token_usage";
  agent: string;
  payload: any;
  ts: string;
}

export interface JobStreamState {
  events: StreamEvent[];
  trace: TraceEvent[];
  plan: any | null;
  criticHistory: any[];
  tokenUsage: any[];
  activeRetry: { from: string; to: string; reason: string } | null;
  progress: number;
  currentAgent: string;
  status: "idle" | "running" | "completed" | "failed";
  error: string | null;
  reportUrl: string | null;
}

// job.messages from the DB is a flat list of strings with no per-message agent tag (only
// the live SSE "update" events carry that). Each agent's first message is prefixed with a
// distinct emoji, so the polling fallback can reconstruct which agent produced each line.
const AGENT_EMOJI_MARKERS: [string, string][] = [
  ["🔍", "eda"],
  ["🧠", "planner"],
  ["🔧", "features"],
  ["🏋️", "experiments"],
  ["🔬", "critic"],
  ["📊", "evaluation"],
  ["📄", "report"],
];

function inferAgentFor(message: string, lastKnown: string): string {
  for (const [emoji, agent] of AGENT_EMOJI_MARKERS) {
    if (message.includes(emoji)) return agent;
  }
  return lastKnown;
}

const INITIAL_STATE: JobStreamState = {
  events: [],
  trace: [],
  plan: null,
  criticHistory: [],
  tokenUsage: [],
  activeRetry: null,
  progress: 0,
  currentAgent: "",
  status: "idle",
  error: null,
  reportUrl: null,
};

export function useJobStream(jobId: string | null) {
  const [state, setState] = useState<JobStreamState>(INITIAL_STATE);
  const retriesRef = useRef(0);
  const MAX_RETRIES = 3;

  const connect = useCallback((signal: AbortSignal) => {
    if (!jobId) return;

    function handleEvent(eventType: string, data: any) {
      retriesRef.current = 0;

      if (eventType === "update") {
        setState((prev) => ({
          ...prev,
          status: "running",
          progress: data.progress_pct ?? prev.progress,
          currentAgent: data.agent ?? prev.currentAgent,
          events: [
            ...prev.events,
            {
              agent: data.agent ?? "",
              progress_pct: data.progress_pct ?? 0,
              message: data.message ?? "",
              timestamp: new Date().toISOString(),
            },
          ],
        }));
      } else if (["thought", "plan", "critic", "retry", "token_usage"].includes(eventType)) {
        setState((prev) => {
          const traceEvent: TraceEvent = {
            kind: eventType as TraceEvent["kind"],
            agent: data.agent,
            payload: data.payload,
            ts: data.ts,
          };
          return {
            ...prev,
            trace: [...prev.trace, traceEvent],
            plan: eventType === "plan" ? data.payload : prev.plan,
            criticHistory: eventType === "critic" ? [...prev.criticHistory, data.payload] : prev.criticHistory,
            tokenUsage: eventType === "token_usage" ? [...prev.tokenUsage, data.payload] : prev.tokenUsage,
            activeRetry: eventType === "retry" ? data.payload : prev.activeRetry,
          };
        });
      } else if (eventType === "completed") {
        setState((prev) => ({
          ...prev,
          status: "completed",
          progress: 100,
          reportUrl: data.report_url ?? null,
        }));
      } else if (eventType === "failed") {
        setState((prev) => ({ ...prev, status: "failed", error: data.error ?? "Pipeline failed" }));
      }
      // "ping"/"done" — no state change needed
    }

    streamJobEvents(jobId, handleEvent, signal).catch((err) => {
      if (signal.aborted) return;
      if (retriesRef.current < MAX_RETRIES) {
        retriesRef.current++;
        const delay = Math.pow(2, retriesRef.current) * 1000;
        setTimeout(() => {
          if (!signal.aborted) connect(signal);
        }, delay);
      } else {
        setState((prev) => ({
          ...prev,
          status: "failed",
          error: err instanceof Error ? err.message : "Connection lost after multiple retries",
        }));
      }
    });
  }, [jobId]);

  useEffect(() => {
    if (!jobId) return;
    const controller = new AbortController();
    connect(controller.signal);
    return () => controller.abort();
  }, [jobId, connect]);

  // Polling fallback: heavy synchronous CPU work in the pipeline (real model training,
  // real LLM calls) can starve the server's single event loop for stretches long enough
  // that the already-open SSE connection doesn't get a turn to flush — the UI looks frozen
  // even though the job is genuinely progressing. Short-lived poll requests get through
  // more reliably than one long-held stream, so this keeps the UI honest as a safety net
  // even if SSE stalls. Never regresses progress/status, and only appends messages not
  // already seen from the stream.
  const seenMessageCountRef = useRef(0);
  const lastInferredAgentRef = useRef("eda");
  useEffect(() => {
    if (!jobId) return;
    const interval = setInterval(async () => {
      try {
        const job = await getJob(jobId);
        setState((prev) => {
          if (prev.status === "completed" || prev.status === "failed") return prev;

          const messages: string[] = job.messages ?? [];
          const newMessages = messages.slice(seenMessageCountRef.current);
          seenMessageCountRef.current = messages.length;

          const newEvents = newMessages.map((m) => {
            const agent = inferAgentFor(m, lastInferredAgentRef.current);
            lastInferredAgentRef.current = agent;
            return {
              agent,
              progress_pct: job.progress_pct ?? prev.progress,
              message: m,
              timestamp: new Date().toISOString(),
            };
          });

          const nextStatus =
            job.status === "completed" ? "completed" :
            job.status === "failed" ? "failed" :
            prev.status === "idle" ? "running" : prev.status;

          return {
            ...prev,
            status: nextStatus,
            progress: Math.max(prev.progress, job.progress_pct ?? prev.progress),
            currentAgent: job.current_agent ?? prev.currentAgent,
            events: newEvents.length > 0 ? [...prev.events, ...newEvents] : prev.events,
            error: job.status === "failed" ? (job.errors?.[0] ?? "Pipeline failed") : prev.error,
            reportUrl: job.status === "completed" ? `/api/reports/${jobId}/pdf` : prev.reportUrl,
          };
        });
      } catch {
        // transient — SSE or the next poll will catch up
      }
    }, 3000);
    return () => clearInterval(interval);
  }, [jobId]);

  return state;
}
