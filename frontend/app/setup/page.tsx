"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { saveLLMConfig, LLMConfig } from "@/lib/storage";
import { validateKey } from "@/lib/api";

const PROVIDERS = [
  {
    id: "anthropic" as const,
    name: "Anthropic Claude",
    recommended: true,
    description: "Best quality. $5 free credits on signup.",
    link: "console.anthropic.com/keys",
    models: [
      { id: "claude-haiku-4-5", label: "Claude Haiku 4.5 (cheaper, faster)" },
      { id: "claude-sonnet-4-6", label: "Claude Sonnet 4.6 (best quality)" },
    ],
  },
  {
    id: "openai" as const,
    name: "OpenAI",
    description: "GPT-4o Mini is fast and affordable.",
    link: "platform.openai.com/api-keys",
    models: [
      { id: "gpt-4o-mini", label: "GPT-4o Mini (affordable)" },
      { id: "gpt-4o", label: "GPT-4o (best)" },
    ],
  },
  {
    id: "groq" as const,
    name: "Groq",
    free: true,
    description: "Llama 3.3 70B. No credit card needed.",
    link: "console.groq.com",
    models: [
      { id: "llama-3.3-70b-versatile", label: "Llama 3.3 70B (free tier)" },
      { id: "mixtral-8x7b-32768", label: "Mixtral 8x7B (free tier)" },
    ],
  },
];

export default function SetupPage() {
  const router = useRouter();
  const [provider, setProvider] = useState<"anthropic" | "openai" | "groq">("anthropic");
  const [model, setModel] = useState("claude-haiku-4-5");
  const [apiKey, setApiKey] = useState("");
  const [showKey, setShowKey] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const selectedProvider = PROVIDERS.find((p) => p.id === provider)!;

  function handleProviderChange(id: "anthropic" | "openai" | "groq") {
    setProvider(id);
    const p = PROVIDERS.find((p) => p.id === id)!;
    setModel(p.models[0].id);
    setError(null);
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!apiKey.trim()) {
      setError("Please enter your API key.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const config: LLMConfig = { provider, model, apiKey: apiKey.trim() };
      const result = await validateKey(config);
      if (result.valid) {
        saveLLMConfig(config);
        router.push("/");
      } else {
        setError(result.error ?? "Key validation failed. Check your key and try again.");
      }
    } catch (err) {
      setError("Could not connect to the server. Is it running?");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-brand-950 flex items-center justify-center p-4">
      <div className="w-full max-w-lg bg-white rounded-2xl shadow-2xl p-8">
        <div className="text-center mb-6">
          <span className="inline-flex items-center justify-center w-12 h-12 rounded-2xl bg-gradient-to-br from-brand-600 to-violet-600 text-white text-lg font-bold shadow-soft mb-3">
            ML
          </span>
          <h1 className="text-2xl font-bold text-slate-900 tracking-tight">ML Intern</h1>
          <p className="text-slate-500 mt-1">Autonomous Data Science Agent</p>
        </div>

        <hr className="my-6 border-slate-100" />

        <form onSubmit={handleSubmit}>
          <p className="font-semibold text-slate-700 mb-3">Choose your AI provider</p>

          <div className="space-y-3 mb-6">
            {PROVIDERS.map((p) => (
              <label
                key={p.id}
                className={`flex items-start gap-3 p-3.5 rounded-xl border-2 cursor-pointer transition-all ${
                  provider === p.id
                    ? "border-brand-500 bg-brand-50"
                    : "border-slate-200 hover:border-slate-300"
                }`}
              >
                <input
                  type="radio"
                  name="provider"
                  value={p.id}
                  checked={provider === p.id}
                  onChange={() => handleProviderChange(p.id)}
                  className="mt-1 accent-brand-600"
                />
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-slate-900">{p.name}</span>
                    {p.recommended && (
                      <span className="text-xs bg-brand-100 text-brand-700 px-2 py-0.5 rounded-full font-medium">
                        Recommended
                      </span>
                    )}
                    {p.free && (
                      <span className="text-xs bg-emerald-100 text-emerald-700 px-2 py-0.5 rounded-full font-medium">
                        FREE
                      </span>
                    )}
                  </div>
                  <p className="text-sm text-slate-500 mt-0.5">{p.description}</p>
                  <a
                    href={`https://${p.link}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-xs text-brand-600 hover:underline"
                    onClick={(e) => e.stopPropagation()}
                  >
                    → {p.link}
                  </a>
                </div>
              </label>
            ))}
          </div>

          <hr className="my-6 border-slate-100" />

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">
                Your API key
              </label>
              <div className="relative">
                <input
                  type={showKey ? "text" : "password"}
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  placeholder={
                    provider === "anthropic"
                      ? "sk-ant-..."
                      : provider === "openai"
                      ? "sk-..."
                      : "gsk_..."
                  }
                  className="w-full px-4 py-2.5 border border-slate-200 rounded-xl pr-10 focus:outline-none focus:ring-2 focus:ring-brand-500 font-mono text-sm"
                  autoComplete="off"
                />
                <button
                  type="button"
                  onClick={() => setShowKey((v) => !v)}
                  className="absolute right-3 top-2.5 text-slate-400 hover:text-slate-600"
                >
                  {showKey ? "🙈" : "👁"}
                </button>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Model</label>
              <select
                value={model}
                onChange={(e) => setModel(e.target.value)}
                className="w-full px-4 py-2.5 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-brand-500 text-sm"
              >
                {selectedProvider.models.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.label}
                  </option>
                ))}
              </select>
            </div>

            {error && (
              <div className="bg-rose-50 border border-rose-200 text-rose-700 rounded-xl p-3 text-sm">
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full py-3 bg-gradient-to-br from-brand-600 to-violet-600 hover:from-brand-700 hover:to-violet-700 disabled:from-slate-300 disabled:to-slate-300 text-white font-semibold rounded-xl transition-colors flex items-center justify-center gap-2 shadow-soft"
            >
              {loading ? (
                <>
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  Verifying...
                </>
              ) : (
                "Verify key and continue →"
              )}
            </button>
          </div>
        </form>

        <hr className="my-6 border-slate-100" />
        <div className="flex items-start gap-2 text-sm text-slate-500">
          <span aria-hidden>🔒</span>
          <p>
            Your key is stored only in your browser. It is never sent to our servers or logged.
            <span className="font-medium text-slate-700"> We pay $0 to run this app.</span>
          </p>
        </div>
      </div>
    </div>
  );
}
