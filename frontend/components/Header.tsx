"use client";

import Link from "next/link";
import { getLLMConfig } from "@/lib/storage";
import { useEffect, useState } from "react";

export function Header() {
  const [config, setConfig] = useState<{ provider: string; model: string } | null>(null);

  useEffect(() => {
    const c = getLLMConfig();
    if (c) setConfig({ provider: c.provider, model: c.model });
  }, []);

  return (
    <header className="bg-white/80 backdrop-blur-md border-b border-slate-200/70 sticky top-0 z-50">
      <div className="max-w-6xl mx-auto px-4 h-16 flex items-center justify-between">
        <Link href="/" className="flex items-center gap-2.5 font-bold text-slate-900">
          <span className="flex items-center justify-center w-8 h-8 rounded-lg bg-gradient-to-br from-brand-600 to-violet-600 text-white text-sm shadow-soft">
            ML
          </span>
          <span className="tracking-tight">ML Intern</span>
        </Link>

        <nav className="flex items-center gap-2">
          <Link
            href="/jobs"
            className="text-sm font-medium text-slate-600 hover:text-slate-900 px-3 py-1.5 rounded-lg hover:bg-slate-100 transition-colors"
          >
            My Jobs
          </Link>
          {config && (
            <span className="text-xs font-medium bg-brand-50 text-brand-700 border border-brand-100 px-2.5 py-1 rounded-full font-mono">
              {config.provider} · {config.model.split("-").slice(-2).join("-")}
            </span>
          )}
          <Link
            href="/setup"
            className="text-sm font-medium text-slate-600 hover:text-slate-900 px-3 py-1.5 rounded-lg hover:bg-slate-100 transition-colors flex items-center gap-1.5"
          >
            <span aria-hidden>⚙</span> Settings
          </Link>
        </nav>
      </div>
    </header>
  );
}
