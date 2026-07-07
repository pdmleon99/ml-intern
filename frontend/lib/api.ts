import { getLLMConfig, LLMConfig } from "./storage";

const BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface CreateJobPayload {
  dataset_source: "upload" | "huggingface" | "kaggle";
  uploaded_file_path?: string;
  hf_dataset_id?: string;
  hf_split?: string;
  hf_max_rows?: number;
  kaggle_dataset?: string;
  kaggle_filename?: string;
  user_description: string;
  target_column?: string;
}

function getLLMHeaders(): HeadersInit {
  const config = getLLMConfig();
  if (!config) throw new Error("No API key configured. Go to Settings.");
  return {
    "X-LLM-Api-Key": config.apiKey,
    "X-LLM-Provider": config.provider,
    "X-LLM-Model": config.model,
    "Content-Type": "application/json",
  };
}

export async function createJob(payload: CreateJobPayload) {
  const res = await fetch(`${BASE_URL}/api/jobs`, {
    method: "POST",
    headers: getLLMHeaders(),
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(err.detail ?? "Failed to create job");
  }
  return res.json();
}

export async function getJob(jobId: string) {
  const res = await fetch(`${BASE_URL}/api/jobs/${jobId}`, {
    headers: getLLMHeaders(),
  });
  if (!res.ok) throw new Error("Failed to fetch job");
  return res.json();
}

export async function listJobs() {
  const res = await fetch(`${BASE_URL}/api/jobs`, {
    headers: getLLMHeaders(),
  });
  if (!res.ok) throw new Error("Failed to list jobs");
  return res.json();
}

/**
 * Streams job events via fetch() + a manually-parsed SSE body, instead of the native
 * EventSource API — EventSource cannot set custom headers, which previously forced the
 * API key into the URL query string (`?k=...`), leaking it into server logs and browser
 * history. Headers are the only place a credential should travel.
 */
export async function streamJobEvents(
  jobId: string,
  onEvent: (event: string, data: any) => void,
  signal: AbortSignal
): Promise<void> {
  const res = await fetch(`${BASE_URL}/api/jobs/${jobId}/stream`, {
    headers: getLLMHeaders(),
    signal,
  });
  if (!res.ok || !res.body) {
    throw new Error(`Failed to open stream (${res.status})`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const chunks = buffer.split("\n\n");
    buffer = chunks.pop() ?? "";

    for (const chunk of chunks) {
      if (!chunk.trim()) continue;
      let eventType = "message";
      const dataLines: string[] = [];
      for (const line of chunk.split("\n")) {
        if (line.startsWith("event:")) eventType = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
      }
      const raw = dataLines.join("\n");
      let data: any = raw;
      try {
        data = JSON.parse(raw);
      } catch {
        // keep raw string
      }
      onEvent(eventType, data);
      if (eventType === "completed" || eventType === "failed" || eventType === "done") {
        return;
      }
    }
  }
}

export async function uploadDataset(file: File) {
  const config = getLLMConfig();
  if (!config) throw new Error("No API key configured");
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${BASE_URL}/api/datasets/upload`, {
    method: "POST",
    headers: {
      "X-LLM-Api-Key": config.apiKey,
      "X-LLM-Provider": config.provider,
      "X-LLM-Model": config.model,
    },
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Upload failed" }));
    throw new Error(err.detail ?? "Upload failed");
  }
  return res.json();
}

export async function loadFromHuggingFace(
  datasetId: string,
  split: string = "train",
  maxRows: number = 500_000
) {
  const res = await fetch(`${BASE_URL}/api/datasets/from-huggingface`, {
    method: "POST",
    headers: getLLMHeaders(),
    body: JSON.stringify({ dataset_id: datasetId, split, max_rows: maxRows }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "HuggingFace load failed" }));
    throw new Error(err.detail ?? "HuggingFace load failed");
  }
  return res.json();
}

export async function downloadReport(jobId: string): Promise<Blob> {
  const res = await fetch(`${BASE_URL}/api/reports/${jobId}/pdf`, {
    headers: getLLMHeaders(),
  });
  if (!res.ok) throw new Error("Failed to download report");
  return res.blob();
}

export async function getReportJson(jobId: string) {
  const res = await fetch(`${BASE_URL}/api/reports/${jobId}/json`, {
    headers: getLLMHeaders(),
  });
  if (!res.ok) throw new Error("Report not available");
  return res.json();
}

export async function validateKey(config: LLMConfig): Promise<{ valid: boolean; error?: string }> {
  const res = await fetch(`${BASE_URL}/api/validate-key`, {
    method: "POST",
    headers: {
      "X-LLM-Api-Key": config.apiKey,
      "X-LLM-Provider": config.provider,
      "X-LLM-Model": config.model,
      "Content-Type": "application/json",
    },
  });
  return res.json();
}
