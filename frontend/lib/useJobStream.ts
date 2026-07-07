"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { streamJobEvents } from "./api";

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

  return state;
}
