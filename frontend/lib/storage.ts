const STORAGE_KEY = "ml_intern_llm_config";

export interface LLMConfig {
  provider: "anthropic" | "openai" | "groq";
  model: string;
  apiKey: string;
}

export function getLLMConfig(): LLMConfig | null {
  if (typeof window === "undefined") return null;
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) return null;
  try {
    return JSON.parse(raw) as LLMConfig;
  } catch {
    return null;
  }
}

export function saveLLMConfig(config: LLMConfig): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(config));
}

export function clearLLMConfig(): void {
  localStorage.removeItem(STORAGE_KEY);
}
