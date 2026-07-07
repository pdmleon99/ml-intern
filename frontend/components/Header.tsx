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
    <header className="bg-white border-b border-gray-200 sticky top-0 z-50">
      <div className="max-w-6xl mx-auto px-4 h-14 flex items-center justify-between">
        <Link href="/" className="flex items-center gap-2 font-bold text-gray-900">
          <span className="text-xl">🤖</span>
          <span>ML Intern</span>
        </Link>

        <nav className="flex items-center gap-4">
          <Link href="/jobs" className="text-sm text-gray-600 hover:text-gray-900">
            My Jobs
          </Link>
          {config && (
            <span className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded-full">
              {config.provider} / {config.model.split("-").slice(-2).join("-")}
            </span>
          )}
          <Link
            href="/setup"
            className="text-sm text-gray-600 hover:text-gray-900 flex items-center gap-1"
          >
            ⚙ Settings
          </Link>
        </nav>
      </div>
    </header>
  );
}
