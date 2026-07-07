"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { Header } from "@/components/Header";
import { getJob, getReportJson, downloadReport } from "@/lib/api";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
} from "recharts";

type Tab = "overview" | "features" | "models" | "best" | "trace" | "download";

export default function ReportPage() {
  const params = useParams<{ id: string }>();
  const jobId = params.id;
  const [tab, setTab] = useState<Tab>("overview");
  const [report, setReport] = useState<any>(null);
  const [job, setJob] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([getJob(jobId), getReportJson(jobId)])
      .then(([jobData, reportData]) => {
        setJob(jobData);
        setReport(reportData);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [jobId]);

  async function handleDownloadPdf() {
    try {
      const blob = await downloadReport(jobId);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `ml_intern_report_${jobId.slice(0, 8)}.pdf`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      alert("Download failed: " + (e instanceof Error ? e.message : "Unknown error"));
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50">
        <Header />
        <div className="flex items-center justify-center min-h-[60vh]">
          <div className="w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50">
        <Header />
        <div className="max-w-4xl mx-auto px-4 py-8">
          <div className="bg-red-50 border border-red-200 rounded-xl p-6">
            <p className="text-red-700">Error loading report: {error}</p>
            <Link href="/jobs" className="text-blue-600 underline mt-2 block">← Back to jobs</Link>
          </div>
        </div>
      </div>
    );
  }

  const TABS: { key: Tab; label: string }[] = [
    { key: "overview", label: "Overview" },
    { key: "features", label: "Features" },
    { key: "models", label: "Model Comparison" },
    { key: "best", label: "Best Model" },
    { key: "trace", label: "Agent Trace" },
    { key: "download", label: "Download" },
  ];

  const SEVERITY_COLORS: Record<string, string> = {
    critical: "bg-red-100 text-red-700 border border-red-200",
    warning: "bg-yellow-100 text-yellow-700 border border-yellow-200",
    info: "bg-blue-100 text-blue-700 border border-blue-200",
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Header />
      <main className="max-w-6xl mx-auto px-4 py-8">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">ML Report</h1>
            <p className="text-sm text-gray-500 font-mono mt-1">{jobId.slice(0, 16)}...</p>
          </div>
          <Link href="/jobs" className="text-sm text-gray-500 hover:text-gray-700">
            ← Back to jobs
          </Link>
        </div>

        {/* Tabs */}
        <div className="flex gap-1 bg-gray-100 rounded-lg p-1 mb-6 overflow-x-auto">
          {TABS.map((t) => (
            <button
              key={t.key}
              onClick={() => setTab(t.key)}
              className={`px-4 py-2 text-sm font-medium rounded-md whitespace-nowrap transition-all ${
                tab === t.key ? "bg-white shadow text-blue-600" : "text-gray-600 hover:text-gray-900"
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>

        {/* Tab: Overview */}
        {tab === "overview" && (
          <div className="space-y-6">
            {/* Stats cards */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {[
                { label: "Rows", value: report?.dataset_profile?.n_rows?.toLocaleString() ?? "—" },
                { label: "Columns", value: report?.dataset_profile?.n_cols ?? "—" },
                { label: "Missing %", value: `${((report?.dataset_profile?.missing_pct ?? 0) * 100).toFixed(1)}%` },
                { label: "Problem Type", value: report?.problem_type ?? "—" },
              ].map((stat) => (
                <div key={stat.label} className="bg-white rounded-xl border border-gray-200 p-4 text-center">
                  <div className="text-2xl font-bold text-blue-600">{stat.value}</div>
                  <div className="text-sm text-gray-500 mt-1">{stat.label}</div>
                </div>
              ))}
            </div>

            {/* EDA narrative */}
            {report?.eda_narrative?.summary && (
              <div className="bg-white rounded-xl border border-gray-200 p-6">
                <h2 className="font-semibold text-gray-800 mb-3">Summary</h2>
                <p className="text-gray-700">{report.eda_narrative.summary}</p>
              </div>
            )}

            {/* Findings */}
            {report?.eda_findings?.length > 0 && (
              <div className="bg-white rounded-xl border border-gray-200 p-6">
                <h2 className="font-semibold text-gray-800 mb-4">
                  Findings ({report.eda_findings.length})
                </h2>
                <div className="space-y-2">
                  {report.eda_findings.map((f: any, i: number) => (
                    <div
                      key={i}
                      className={`flex items-start gap-3 p-3 rounded-lg text-sm ${
                        SEVERITY_COLORS[f.severity] ?? "bg-gray-50 border border-gray-200"
                      }`}
                    >
                      <span className="font-semibold capitalize min-w-16">{f.severity}</span>
                      <span className="flex-1">{f.finding}</span>
                      {f.column && f.column !== "dataset" && (
                        <span className="font-mono text-xs opacity-70">{f.column}</span>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* EDA Charts */}
            {report?.eda_charts?.length > 0 && (
              <div>
                <h2 className="font-semibold text-gray-800 mb-4">Charts</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {report.eda_charts.map((chart: any, i: number) => (
                    <div key={i} className="bg-white rounded-xl border border-gray-200 p-4">
                      <h3 className="font-medium text-gray-700 mb-2 text-sm">{chart.title}</h3>
                      <img
                        src={`data:image/png;base64,${chart.base64_png}`}
                        alt={chart.title}
                        className="w-full rounded"
                      />
                      {chart.description && (
                        <p className="text-xs text-gray-500 mt-2">{chart.description}</p>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Tab: Features */}
        {tab === "features" && (
          <div className="space-y-6">
            <div className="bg-white rounded-xl border border-gray-200 p-6">
              <h2 className="font-semibold text-gray-800 mb-2">Transformations Applied</h2>
              <p className="text-sm text-gray-500 mb-4">
                {report?.features_applied?.length ?? 0} transformations, {report?.feature_names?.length ?? 0} final features
              </p>
              <div className="overflow-x-auto">
                <table className="text-sm w-full">
                  <thead className="bg-gray-50 border-b">
                    <tr>
                      <th className="text-left p-3 font-medium text-gray-600">Transformation</th>
                      <th className="text-left p-3 font-medium text-gray-600">Type</th>
                      <th className="text-left p-3 font-medium text-gray-600">Rationale</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(report?.features_applied ?? []).map((f: any, i: number) => (
                      <tr key={i} className={i % 2 === 0 ? "bg-white" : "bg-gray-50"}>
                        <td className="p-3 font-mono text-xs">{f.name}</td>
                        <td className="p-3">
                          <span className="bg-blue-100 text-blue-700 px-2 py-0.5 rounded text-xs">
                            {f.type}
                          </span>
                        </td>
                        <td className="p-3 text-gray-600 text-xs">{f.rationale}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {report?.dropped_columns?.length > 0 && (
              <div className="bg-white rounded-xl border border-gray-200 p-6">
                <h2 className="font-semibold text-gray-800 mb-4">Dropped Columns</h2>
                <div className="space-y-2">
                  {report.dropped_columns.map((d: any, i: number) => (
                    <div key={i} className="flex items-center gap-3 p-2 bg-gray-50 rounded-lg text-sm">
                      <span className="font-mono text-xs bg-white border border-gray-200 px-2 py-0.5 rounded">
                        {d.column}
                      </span>
                      <span className="text-gray-500">{d.reason}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Tab: Model Comparison */}
        {tab === "models" && (
          <div className="space-y-6">
            {(() => {
              const completed = (report?.experiments ?? []).filter(
                (e: any) => e.status === "completed" || e.status === "completed_reduced"
              );
              const pm = completed[0]?.primary_metric ?? "score";
              const chartData = [...completed]
                .sort((a: any, b: any) => b.primary_score - a.primary_score)
                .map((e: any) => ({ name: e.model_name, score: parseFloat(e.primary_score?.toFixed(4) ?? "0") }));

              return (
                <>
                  {chartData.length > 0 && (
                    <div className="bg-white rounded-xl border border-gray-200 p-6">
                      <h2 className="font-semibold text-gray-800 mb-4">{pm} Comparison</h2>
                      <ResponsiveContainer width="100%" height={260}>
                        <BarChart data={chartData} margin={{ left: 0, right: 20 }}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                          <YAxis domain={[0, 1]} tick={{ fontSize: 12 }} />
                          <Tooltip formatter={(v: number) => v.toFixed(4)} />
                          <Bar dataKey="score" radius={[4, 4, 0, 0]}>
                            {chartData.map((_: any, index: number) => (
                              <Cell key={index} fill={index === 0 ? "#22c55e" : "#3b82f6"} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  )}

                  <div className="bg-white rounded-xl border border-gray-200 p-6">
                    <table className="text-sm w-full">
                      <thead className="bg-gray-50 border-b">
                        <tr>
                          <th className="text-left p-3 font-medium text-gray-600">Model</th>
                          <th className="text-left p-3 font-medium text-gray-600">{pm} (mean)</th>
                          <th className="text-left p-3 font-medium text-gray-600">±Std</th>
                          <th className="text-left p-3 font-medium text-gray-600">Time</th>
                          <th className="text-left p-3 font-medium text-gray-600">Status</th>
                        </tr>
                      </thead>
                      <tbody>
                        {[...(report?.experiments ?? [])]
                          .sort((a: any, b: any) => b.primary_score - a.primary_score)
                          .map((e: any, i: number) => {
                            const pm_scores = e.cv_scores?.[`test_${pm}`] ?? {};
                            const isBest = e.model_name === report?.best_model_name;
                            return (
                              <tr key={i} className={
                                isBest ? "bg-green-50" : e.is_baseline ? "bg-gray-50 italic text-gray-500"
                                : i % 2 === 0 ? "bg-white" : "bg-gray-50"
                              }>
                                <td className="p-3 font-medium flex items-center gap-1">
                                  {isBest && <span>👑</span>}
                                  {e.model_name}
                                  {e.is_baseline && <span className="text-xs text-gray-400">(naive baseline)</span>}
                                </td>
                                <td className="p-3 font-mono">{(pm_scores.mean ?? e.primary_score ?? 0).toFixed(4)}</td>
                                <td className="p-3 text-gray-400 font-mono">±{(pm_scores.std ?? 0).toFixed(4)}</td>
                                <td className="p-3 text-gray-500">{e.train_time_s?.toFixed(1) ?? "—"}s</td>
                                <td className="p-3">
                                  <span className={`text-xs px-2 py-0.5 rounded-full ${
                                    e.status === "completed" ? "bg-green-100 text-green-700" :
                                    e.status === "completed_reduced" ? "bg-yellow-100 text-yellow-700" :
                                    "bg-red-100 text-red-700"
                                  }`}>
                                    {e.status}
                                  </span>
                                </td>
                              </tr>
                            );
                          })}
                      </tbody>
                    </table>
                  </div>
                </>
              );
            })()}
          </div>
        )}

        {/* Tab: Best Model */}
        {tab === "best" && (
          <div className="space-y-6">
            <div className="bg-white rounded-xl border border-gray-200 p-6">
              <div className="flex items-center gap-3 mb-4">
                <span className="text-3xl">👑</span>
                <div>
                  <h2 className="text-xl font-bold text-gray-900">{report?.best_model_name}</h2>
                  <p className="text-sm text-gray-500">{report?.problem_type} model for "{report?.target_column}"</p>
                </div>
              </div>

              {report?.evaluation_metrics && (() => {
                const m = report.evaluation_metrics as Record<string, any>;
                const testMetrics = Object.entries(m).filter(
                  ([k]) => k.startsWith("test_") && k !== "classification_report"
                );
                // Find primary CV metric pair
                const cvMeanKey = Object.keys(m).find(k => k.startsWith("cv_") && k.endsWith("_mean"));
                const cvStdKey  = cvMeanKey ? cvMeanKey.replace("_mean", "_std") : undefined;
                const cvLabel   = cvMeanKey ? cvMeanKey.replace("cv_", "").replace("_mean", "").toUpperCase() : "";
                const cvMean    = cvMeanKey ? m[cvMeanKey] : undefined;
                const cvStd     = cvStdKey  ? m[cvStdKey]  : undefined;
                const testLabel = cvMeanKey ? `test_${cvMeanKey.replace("cv_", "").replace("_mean", "")}` : "";
                const testVal   = testLabel ? m[testLabel]  : undefined;

                return (
                  <>
                    {/* Test-set metrics grid */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
                      {testMetrics.map(([k, v]) => (
                        <div key={k} className="bg-blue-50 rounded-lg p-3 text-center">
                          <div className="text-xl font-bold text-blue-700">
                            {typeof v === "number" ? v.toFixed(4) : "—"}
                          </div>
                          <div className="text-xs text-gray-500 uppercase mt-1">
                            {k.replace("test_", "")} <span className="text-blue-400">(test)</span>
                          </div>
                        </div>
                      ))}
                    </div>

                    {/* CV vs Test comparison panel */}
                    {cvMean !== undefined && testVal !== undefined && (
                      <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 mb-4">
                        <p className="text-xs font-semibold text-amber-800 mb-3 uppercase tracking-wide">
                          Evaluation Methodology
                        </p>
                        <div className="grid grid-cols-2 gap-6 mb-3">
                          <div className="text-center">
                            <div className="text-2xl font-bold text-blue-700">{(testVal as number).toFixed(4)}</div>
                            <div className="text-xs text-gray-600 mt-1">
                              {cvLabel} <span className="font-semibold">(test set)</span>
                            </div>
                            <div className="text-xs text-green-600 mt-0.5">← real number</div>
                          </div>
                          <div className="text-center">
                            <div className="text-2xl font-bold text-amber-700">
                              {(cvMean as number).toFixed(4)}
                              {cvStd !== undefined && (
                                <span className="text-base font-normal text-amber-500"> ± {(cvStd as number).toFixed(3)}</span>
                              )}
                            </div>
                            <div className="text-xs text-gray-600 mt-1">
                              {cvLabel} <span className="font-semibold">({report.experiments?.length > 0 ? (report.experiments[0]?.cv_scores ? Object.keys(report.experiments[0].cv_scores).length > 0 ? "5" : "?" : "?") : "?"}‑fold CV)</span>
                            </div>
                            <div className="text-xs text-amber-600 mt-0.5">← training estimate</div>
                          </div>
                        </div>
                        <p className="text-xs text-amber-700 leading-relaxed">
                          ℹ️ <strong>CV score</strong> is measured on training data (optimistic).{" "}
                          <strong>Test score</strong> is measured on held-out data the model never saw during training (realistic).
                        </p>
                      </div>
                    )}
                  </>
                );
              })()}

              {report?.statistical_comparison?.vs_baseline && (() => {
                const vb = report.statistical_comparison.vs_baseline;
                const ci = report.statistical_comparison.test_set_ci;
                const CONF_STYLES: Record<string, { box: string; text: string }> = {
                  high: { box: "bg-green-50 border-green-200", text: "text-green-700" },
                  medium: { box: "bg-amber-50 border-amber-200", text: "text-amber-700" },
                  low: { box: "bg-red-50 border-red-200", text: "text-red-700" },
                };
                const style = CONF_STYLES[vb.confidence] ?? CONF_STYLES.low;
                return (
                  <div className={`${style.box} border rounded-lg p-4 mb-4`}>
                    <p className="text-xs font-semibold text-gray-700 mb-2 uppercase tracking-wide">
                      Statistical Confidence — is this actually better than guessing?
                    </p>
                    <p className="text-sm text-gray-800">
                      <span className={`font-bold ${style.text}`}>
                        {vb.lift_pct !== null ? `${(vb.lift_pct * 100).toFixed(1)}%` : "n/a"} lift
                      </span>{" "}
                      over a naive baseline —{" "}
                      <span className="font-semibold">{vb.confidence} confidence</span>{" "}
                      (P better than baseline = {vb.probability_better_than_baseline?.toFixed(2)}).
                    </p>
                    {ci?.mean !== undefined && ci?.mean !== null && (
                      <p className="text-xs text-gray-500 mt-1">
                        Test-set {report.statistical_comparison.primary_metric}: {ci.mean.toFixed(4)}{" "}
                        (95% CI: {ci.ci_low?.toFixed(4)}–{ci.ci_high?.toFixed(4)})
                      </p>
                    )}
                  </div>
                );
              })()}

              {report?.model_explanation && (
                <div className="bg-gray-50 rounded-lg p-4 text-gray-700 text-sm leading-relaxed">
                  {report.model_explanation}
                </div>
              )}
            </div>

            {report?.evaluation_charts?.length > 0 && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {report.evaluation_charts.map((chart: any, i: number) => (
                  <div key={i} className="bg-white rounded-xl border border-gray-200 p-4">
                    <h3 className="font-medium text-gray-700 mb-2 text-sm">{chart.title}</h3>
                    <img
                      src={`data:image/png;base64,${chart.base64_png}`}
                      alt={chart.title}
                      className="w-full rounded"
                    />
                  </div>
                ))}
              </div>
            )}

            {report?.recommendations?.length > 0 && (
              <div className="bg-white rounded-xl border border-gray-200 p-6">
                <h2 className="font-semibold text-gray-800 mb-4">Next Steps & Recommendations</h2>
                <ol className="space-y-2">
                  {report.recommendations.map((rec: string, i: number) => (
                    <li key={i} className="flex gap-3 text-sm text-gray-700">
                      <span className="flex-shrink-0 w-6 h-6 bg-blue-100 text-blue-700 rounded-full flex items-center justify-center text-xs font-bold">
                        {i + 1}
                      </span>
                      {rec}
                    </li>
                  ))}
                </ol>
              </div>
            )}
          </div>
        )}

        {/* Tab: Agent Trace */}
        {tab === "trace" && (
          <div className="space-y-6">
            {report?.plan && (
              <div className="bg-white rounded-xl border border-gray-200 p-6">
                <h2 className="font-semibold text-gray-800 mb-3">🧠 Planner Strategy</h2>
                <p className="text-sm text-gray-700 mb-3">{report.plan.reasoning}</p>
                <div className="flex flex-wrap gap-2 mb-3">
                  {(report.plan.model_shortlist ?? []).map((m: string) => (
                    <span key={m} className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded-full font-mono">
                      {m}
                    </span>
                  ))}
                </div>
                {report.plan.columns_to_drop?.length > 0 && (
                  <div className="text-sm text-gray-600">
                    <span className="font-medium">Dropped by planner:</span>{" "}
                    {report.plan.columns_to_drop.map((d: any) => d.column).join(", ")}
                  </div>
                )}
                {report.plan.risk_flags?.length > 0 && (
                  <div className="mt-2 text-sm text-amber-700">
                    ⚠ {report.plan.risk_flags.join(" · ")}
                  </div>
                )}
              </div>
            )}

            {report?.critic_history?.length > 0 && (
              <div className="bg-white rounded-xl border border-gray-200 p-6">
                <h2 className="font-semibold text-gray-800 mb-4">🔬 Critic Review History</h2>
                <div className="space-y-3">
                  {report.critic_history.map((c: any, i: number) => {
                    const isRetry = c.verdict?.startsWith("retry");
                    return (
                      <div key={i} className={`p-3 rounded-lg border text-sm ${
                        isRetry ? "bg-amber-50 border-amber-200" : "bg-green-50 border-green-200"
                      }`}>
                        <div className="flex items-center gap-2 mb-1">
                          <span className="font-semibold">Pass {i + 1}: {c.verdict}</span>
                          <span className="text-xs text-gray-500">({c.confidence} confidence)</span>
                          {isRetry && <span className="text-xs">↺ retry triggered</span>}
                        </div>
                        <p className="text-gray-700">{c.reasoning}</p>
                        {c.concerns?.length > 0 && (
                          <p className="text-xs text-gray-500 mt-1">Concerns: {c.concerns.join("; ")}</p>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {report?.token_usage?.length > 0 && (() => {
              const totalTokens = report.token_usage.reduce(
                (s: number, t: any) => s + (t.input_tokens ?? 0) + (t.output_tokens ?? 0), 0
              );
              const totalCost = report.token_usage.reduce((s: number, t: any) => s + (t.cost_usd ?? 0), 0);
              const totalLatency = report.token_usage.reduce((s: number, t: any) => s + (t.latency_s ?? 0), 0);
              return (
                <div className="bg-white rounded-xl border border-gray-200 p-6">
                  <h2 className="font-semibold text-gray-800 mb-4">📡 LLM Observability</h2>
                  <div className="grid grid-cols-3 gap-4 mb-4">
                    <div className="bg-gray-50 rounded-lg p-3 text-center">
                      <div className="text-xl font-bold text-gray-800">{totalTokens.toLocaleString()}</div>
                      <div className="text-xs text-gray-500">total tokens</div>
                    </div>
                    <div className="bg-gray-50 rounded-lg p-3 text-center">
                      <div className="text-xl font-bold text-gray-800">${totalCost.toFixed(4)}</div>
                      <div className="text-xs text-gray-500">est. cost</div>
                    </div>
                    <div className="bg-gray-50 rounded-lg p-3 text-center">
                      <div className="text-xl font-bold text-gray-800">{totalLatency.toFixed(1)}s</div>
                      <div className="text-xs text-gray-500">LLM latency</div>
                    </div>
                  </div>
                  <table className="text-sm w-full">
                    <thead className="bg-gray-50 border-b">
                      <tr>
                        <th className="text-left p-2 font-medium text-gray-600">Agent</th>
                        <th className="text-left p-2 font-medium text-gray-600">Model</th>
                        <th className="text-left p-2 font-medium text-gray-600">Tokens (in/out)</th>
                        <th className="text-left p-2 font-medium text-gray-600">Cost</th>
                        <th className="text-left p-2 font-medium text-gray-600">Latency</th>
                      </tr>
                    </thead>
                    <tbody>
                      {report.token_usage.map((t: any, i: number) => (
                        <tr key={i} className={i % 2 === 0 ? "bg-white" : "bg-gray-50"}>
                          <td className="p-2 capitalize">{t.agent}</td>
                          <td className="p-2 font-mono text-xs">{t.model}</td>
                          <td className="p-2 font-mono text-xs">{t.input_tokens}/{t.output_tokens}</td>
                          <td className="p-2 font-mono text-xs">${t.cost_usd?.toFixed(5)}</td>
                          <td className="p-2 font-mono text-xs">{t.latency_s?.toFixed(2)}s</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              );
            })()}

            {!report?.plan && !report?.critic_history?.length && !report?.token_usage?.length && (
              <div className="bg-white rounded-xl border border-gray-200 p-8 text-center text-gray-400">
                No agent trace data available for this run.
              </div>
            )}
          </div>
        )}

        {/* Tab: Download */}
        {tab === "download" && (
          <div className="space-y-4">
            <div className="bg-white rounded-xl border border-gray-200 p-8 text-center">
              <div className="text-5xl mb-4">📄</div>
              <h2 className="text-xl font-bold text-gray-900 mb-2">Download Report</h2>
              <p className="text-gray-500 mb-6">Get your complete analysis as a PDF report.</p>

              <div className="flex flex-col sm:flex-row gap-3 justify-center">
                <button
                  onClick={handleDownloadPdf}
                  className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-xl transition-colors"
                >
                  Download PDF Report
                </button>
                <button
                  onClick={() => {
                    const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement("a");
                    a.href = url;
                    a.download = `ml_intern_report_${jobId.slice(0, 8)}.json`;
                    a.click();
                    URL.revokeObjectURL(url);
                  }}
                  className="px-6 py-3 bg-gray-100 hover:bg-gray-200 text-gray-700 font-semibold rounded-xl transition-colors"
                >
                  Download JSON
                </button>
              </div>

              <div className="mt-6 pt-6 border-t border-gray-100">
                <p className="text-sm text-gray-500 mb-2">Shareable job ID:</p>
                <code className="bg-gray-100 px-3 py-1.5 rounded font-mono text-sm">{jobId}</code>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
