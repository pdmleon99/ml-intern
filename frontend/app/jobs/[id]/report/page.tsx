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

function Card({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return (
    <div className={`bg-white rounded-2xl border border-slate-200 shadow-soft p-6 ${className}`}>
      {children}
    </div>
  );
}

function SectionTitle({ icon, children }: { icon?: string; children: React.ReactNode }) {
  return (
    <h2 className="font-semibold text-slate-800 mb-4 flex items-center gap-2">
      {icon && <span aria-hidden>{icon}</span>}
      {children}
    </h2>
  );
}

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
      <div className="min-h-screen bg-gradient-to-b from-slate-50 to-white">
        <Header />
        <div className="flex items-center justify-center min-h-[60vh]">
          <div className="w-8 h-8 border-4 border-brand-500 border-t-transparent rounded-full animate-spin" />
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-slate-50 to-white">
        <Header />
        <div className="max-w-4xl mx-auto px-4 py-8">
          <div className="bg-rose-50 border border-rose-200 rounded-2xl p-6">
            <p className="text-rose-700">Error loading report: {error}</p>
            <Link href="/jobs" className="text-brand-600 underline mt-2 block">← Back to jobs</Link>
          </div>
        </div>
      </div>
    );
  }

  const TABS: { key: Tab; label: string; icon: string }[] = [
    { key: "overview", label: "Overview", icon: "🔍" },
    { key: "features", label: "Features", icon: "🔧" },
    { key: "models", label: "Model Comparison", icon: "📊" },
    { key: "best", label: "Best Model", icon: "👑" },
    { key: "trace", label: "Agent Trace", icon: "🧠" },
    { key: "download", label: "Download", icon: "📄" },
  ];

  const SEVERITY_STYLES: Record<string, string> = {
    critical: "bg-rose-50 text-rose-700 border border-rose-200",
    warning: "bg-amber-50 text-amber-700 border border-amber-200",
    info: "bg-sky-50 text-sky-700 border border-sky-200",
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-white">
      <Header />
      <main className="max-w-6xl mx-auto px-4 py-10">
        <div className="flex items-center justify-between mb-8">
          <div>
            <p className="text-xs font-semibold text-brand-600 uppercase tracking-wider mb-1">
              Autonomous analysis report
            </p>
            <h1 className="text-3xl font-bold text-slate-900 tracking-tight">ML Report</h1>
            <p className="text-xs text-slate-400 font-mono mt-1.5">{jobId.slice(0, 18)}…</p>
          </div>
          <Link
            href="/jobs"
            className="text-sm font-medium text-slate-500 hover:text-slate-800 transition-colors"
          >
            ← Back to jobs
          </Link>
        </div>

        {/* Tabs */}
        <div className="flex gap-1 bg-slate-100/80 rounded-xl p-1 mb-8 overflow-x-auto">
          {TABS.map((t) => (
            <button
              key={t.key}
              onClick={() => setTab(t.key)}
              className={`px-4 py-2 text-sm font-medium rounded-lg whitespace-nowrap transition-all flex items-center gap-1.5 ${
                tab === t.key
                  ? "bg-white shadow-soft text-brand-700"
                  : "text-slate-500 hover:text-slate-800"
              }`}
            >
              <span aria-hidden className="text-[13px]">{t.icon}</span>
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
                <div
                  key={stat.label}
                  className="bg-white rounded-2xl border border-slate-200 shadow-soft p-5 text-center"
                >
                  <div className="text-2xl font-bold text-brand-600 tracking-tight">{stat.value}</div>
                  <div className="text-xs text-slate-500 mt-1 uppercase tracking-wide">{stat.label}</div>
                </div>
              ))}
            </div>

            {/* EDA narrative */}
            {report?.eda_narrative?.summary && (
              <Card>
                <SectionTitle icon="📝">Summary</SectionTitle>
                <p className="text-slate-600 leading-relaxed">{report.eda_narrative.summary}</p>
              </Card>
            )}

            {/* Findings */}
            {report?.eda_findings?.length > 0 && (
              <Card>
                <SectionTitle icon="🔎">Findings ({report.eda_findings.length})</SectionTitle>
                <div className="space-y-2">
                  {report.eda_findings.map((f: any, i: number) => (
                    <div
                      key={i}
                      className={`flex items-start gap-3 p-3 rounded-xl text-sm ${
                        SEVERITY_STYLES[f.severity] ?? "bg-slate-50 border border-slate-200"
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
              </Card>
            )}

            {/* EDA Charts */}
            {report?.eda_charts?.length > 0 && (
              <div>
                <h2 className="font-semibold text-slate-800 mb-4">Charts</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {report.eda_charts.map((chart: any, i: number) => (
                    <div key={i} className="bg-white rounded-2xl border border-slate-200 shadow-soft p-4">
                      <h3 className="font-medium text-slate-700 mb-2 text-sm">{chart.title}</h3>
                      <img
                        src={`data:image/png;base64,${chart.base64_png}`}
                        alt={chart.title}
                        className="w-full rounded-lg"
                      />
                      {chart.description && (
                        <p className="text-xs text-slate-500 mt-2">{chart.description}</p>
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
            <Card>
              <SectionTitle icon="🔧">Transformations Applied</SectionTitle>
              <p className="text-sm text-slate-500 mb-4 -mt-2">
                {report?.features_applied?.length ?? 0} transformations, {report?.feature_names?.length ?? 0} final features
              </p>
              <div className="overflow-x-auto -mx-2">
                <table className="text-sm w-full">
                  <thead>
                    <tr className="border-b border-slate-200">
                      <th className="text-left px-2 py-2.5 font-medium text-slate-500 text-xs uppercase tracking-wide">Transformation</th>
                      <th className="text-left px-2 py-2.5 font-medium text-slate-500 text-xs uppercase tracking-wide">Type</th>
                      <th className="text-left px-2 py-2.5 font-medium text-slate-500 text-xs uppercase tracking-wide">Rationale</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(report?.features_applied ?? []).map((f: any, i: number) => (
                      <tr key={i} className="border-b border-slate-100 last:border-0 hover:bg-slate-50/70">
                        <td className="px-2 py-2.5 font-mono text-xs text-slate-700">{f.name}</td>
                        <td className="px-2 py-2.5">
                          <span className="bg-brand-50 text-brand-700 px-2 py-0.5 rounded-full text-xs font-medium">
                            {f.type}
                          </span>
                        </td>
                        <td className="px-2 py-2.5 text-slate-500 text-xs">{f.rationale}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>

            {report?.dropped_columns?.length > 0 && (
              <Card>
                <SectionTitle icon="🗑">Dropped Columns</SectionTitle>
                <div className="space-y-2">
                  {report.dropped_columns.map((d: any, i: number) => (
                    <div key={i} className="flex items-center gap-3 p-2.5 bg-slate-50 rounded-xl text-sm">
                      <span className="font-mono text-xs bg-white border border-slate-200 px-2 py-0.5 rounded-md text-slate-700">
                        {d.column}
                      </span>
                      <span className="text-slate-500">{d.reason}</span>
                    </div>
                  ))}
                </div>
              </Card>
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
                .map((e: any) => ({
                  name: e.model_name,
                  score: parseFloat(e.primary_score?.toFixed(4) ?? "0"),
                  isBaseline: !!e.is_baseline,
                }));
              const scores = chartData.map((d) => d.score);
              const yMin = Math.min(0, ...scores);
              const yMax = Math.max(1, ...scores);

              return (
                <>
                  {chartData.length > 0 && (
                    <Card>
                      <SectionTitle icon="📊">{pm} Comparison</SectionTitle>
                      <ResponsiveContainer width="100%" height={280}>
                        <BarChart data={chartData} margin={{ left: 0, right: 20, top: 8 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                          <XAxis dataKey="name" tick={{ fontSize: 12, fill: "#64748b" }} axisLine={{ stroke: "#e2e8f0" }} tickLine={false} />
                          <YAxis domain={[yMin, yMax]} tick={{ fontSize: 12, fill: "#64748b" }} axisLine={false} tickLine={false} />
                          <Tooltip
                            cursor={{ fill: "rgba(79, 70, 229, 0.06)" }}
                            formatter={(v: number) => v.toFixed(4)}
                            contentStyle={{
                              borderRadius: 12,
                              border: "1px solid #e2e8f0",
                              boxShadow: "0 4px 12px -2px rgb(15 23 42 / 0.08)",
                              fontSize: 13,
                            }}
                          />
                          <Bar dataKey="score" radius={[6, 6, 0, 0]} maxBarSize={56}>
                            {chartData.map((d, index) => (
                              <Cell
                                key={index}
                                fill={d.isBaseline ? "#cbd5e1" : index === 0 ? "#059669" : "#4f46e5"}
                              />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </Card>
                  )}

                  <Card className="p-0 overflow-hidden">
                    <table className="text-sm w-full">
                      <thead>
                        <tr className="bg-slate-50 border-b border-slate-200">
                          <th className="text-left px-5 py-3 font-medium text-slate-500 text-xs uppercase tracking-wide">Model</th>
                          <th className="text-left px-5 py-3 font-medium text-slate-500 text-xs uppercase tracking-wide">{pm} (mean)</th>
                          <th className="text-left px-5 py-3 font-medium text-slate-500 text-xs uppercase tracking-wide">±Std</th>
                          <th className="text-left px-5 py-3 font-medium text-slate-500 text-xs uppercase tracking-wide">Time</th>
                          <th className="text-left px-5 py-3 font-medium text-slate-500 text-xs uppercase tracking-wide">Status</th>
                        </tr>
                      </thead>
                      <tbody>
                        {[...(report?.experiments ?? [])]
                          .sort((a: any, b: any) => b.primary_score - a.primary_score)
                          .map((e: any, i: number) => {
                            const pm_scores = e.cv_scores?.[`test_${pm}`] ?? {};
                            const isBest = e.model_name === report?.best_model_name;
                            return (
                              <tr
                                key={i}
                                className={`border-b border-slate-100 last:border-0 ${
                                  isBest ? "bg-emerald-50/60" : e.is_baseline ? "text-slate-400 italic" : "hover:bg-slate-50/70"
                                }`}
                              >
                                <td className="px-5 py-3 font-medium flex items-center gap-1.5 text-slate-800">
                                  {isBest && <span aria-hidden>👑</span>}
                                  {e.model_name}
                                  {e.is_baseline && <span className="text-xs text-slate-400 font-normal">(naive baseline)</span>}
                                </td>
                                <td className="px-5 py-3 font-mono text-slate-700">{(pm_scores.mean ?? e.primary_score ?? 0).toFixed(4)}</td>
                                <td className="px-5 py-3 text-slate-400 font-mono">±{(pm_scores.std ?? 0).toFixed(4)}</td>
                                <td className="px-5 py-3 text-slate-500">{e.train_time_s?.toFixed(1) ?? "—"}s</td>
                                <td className="px-5 py-3">
                                  <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                                    e.status === "completed" ? "bg-emerald-100 text-emerald-700" :
                                    e.status === "completed_reduced" ? "bg-amber-100 text-amber-700" :
                                    "bg-rose-100 text-rose-700"
                                  }`}>
                                    {e.status}
                                  </span>
                                </td>
                              </tr>
                            );
                          })}
                      </tbody>
                    </table>
                  </Card>
                </>
              );
            })()}
          </div>
        )}

        {/* Tab: Best Model */}
        {tab === "best" && (
          <div className="space-y-6">
            <Card>
              <div className="flex items-center gap-3 mb-5">
                <span className="flex items-center justify-center w-12 h-12 rounded-2xl bg-gradient-to-br from-amber-400 to-amber-500 text-2xl shadow-soft">
                  👑
                </span>
                <div>
                  <h2 className="text-xl font-bold text-slate-900 tracking-tight">{report?.best_model_name}</h2>
                  <p className="text-sm text-slate-500">{report?.problem_type} model for &ldquo;{report?.target_column}&rdquo;</p>
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
                        <div key={k} className="bg-brand-50/60 rounded-xl p-3.5 text-center">
                          <div className="text-xl font-bold text-brand-700 tracking-tight">
                            {typeof v === "number" ? v.toFixed(4) : "—"}
                          </div>
                          <div className="text-[11px] text-slate-500 uppercase mt-1 tracking-wide">
                            {k.replace("test_", "")} <span className="text-brand-400">(test)</span>
                          </div>
                        </div>
                      ))}
                    </div>

                    {/* CV vs Test comparison panel */}
                    {cvMean !== undefined && testVal !== undefined && (
                      <div className="bg-amber-50/70 border border-amber-200/70 rounded-xl p-4 mb-4">
                        <p className="text-xs font-semibold text-amber-800 mb-3 uppercase tracking-wide">
                          Evaluation Methodology
                        </p>
                        <div className="grid grid-cols-2 gap-6 mb-3">
                          <div className="text-center">
                            <div className="text-2xl font-bold text-brand-700 tracking-tight">{(testVal as number).toFixed(4)}</div>
                            <div className="text-xs text-slate-600 mt-1">
                              {cvLabel} <span className="font-semibold">(test set)</span>
                            </div>
                            <div className="text-xs text-emerald-600 mt-0.5">← real number</div>
                          </div>
                          <div className="text-center">
                            <div className="text-2xl font-bold text-amber-700 tracking-tight">
                              {(cvMean as number).toFixed(4)}
                              {cvStd !== undefined && (
                                <span className="text-base font-normal text-amber-500"> ± {(cvStd as number).toFixed(3)}</span>
                              )}
                            </div>
                            <div className="text-xs text-slate-600 mt-1">
                              {cvLabel} <span className="font-semibold">({report.experiments?.length > 0 ? (report.experiments[0]?.cv_scores ? Object.keys(report.experiments[0].cv_scores).length > 0 ? "5" : "?" : "?") : "?"}‑fold CV)</span>
                            </div>
                            <div className="text-xs text-amber-600 mt-0.5">← training estimate</div>
                          </div>
                        </div>
                        <p className="text-xs text-amber-700 leading-relaxed">
                          <strong>CV score</strong> is measured on training data (optimistic).{" "}
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
                  high: { box: "bg-emerald-50/70 border-emerald-200/70", text: "text-emerald-700" },
                  medium: { box: "bg-amber-50/70 border-amber-200/70", text: "text-amber-700" },
                  low: { box: "bg-rose-50/70 border-rose-200/70", text: "text-rose-700" },
                };
                const style = CONF_STYLES[vb.confidence] ?? CONF_STYLES.low;
                return (
                  <div className={`${style.box} border rounded-xl p-4 mb-4`}>
                    <p className="text-xs font-semibold text-slate-600 mb-2 uppercase tracking-wide">
                      Statistical Confidence — is this actually better than guessing?
                    </p>
                    <p className="text-sm text-slate-800">
                      <span className={`font-bold ${style.text}`}>
                        {vb.lift_pct !== null ? `${(vb.lift_pct * 100).toFixed(1)}%` : "n/a"} lift
                      </span>{" "}
                      over a naive baseline —{" "}
                      <span className="font-semibold">{vb.confidence} confidence</span>{" "}
                      (P better than baseline = {vb.probability_better_than_baseline?.toFixed(2)}).
                    </p>
                    {ci?.mean !== undefined && ci?.mean !== null && (
                      <p className="text-xs text-slate-500 mt-1">
                        Test-set {report.statistical_comparison.primary_metric}: {ci.mean.toFixed(4)}{" "}
                        (95% CI: {ci.ci_low?.toFixed(4)}–{ci.ci_high?.toFixed(4)})
                      </p>
                    )}
                  </div>
                );
              })()}

              {report?.model_explanation && (
                <div className="bg-slate-50 rounded-xl p-4 text-slate-600 text-sm leading-relaxed">
                  {report.model_explanation}
                </div>
              )}
            </Card>

            {report?.evaluation_charts?.length > 0 && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {report.evaluation_charts.map((chart: any, i: number) => (
                  <div key={i} className="bg-white rounded-2xl border border-slate-200 shadow-soft p-4">
                    <h3 className="font-medium text-slate-700 mb-2 text-sm">{chart.title}</h3>
                    <img
                      src={`data:image/png;base64,${chart.base64_png}`}
                      alt={chart.title}
                      className="w-full rounded-lg"
                    />
                  </div>
                ))}
              </div>
            )}

            {report?.recommendations?.length > 0 && (
              <Card>
                <SectionTitle icon="✅">Next Steps &amp; Recommendations</SectionTitle>
                <ol className="space-y-2.5">
                  {report.recommendations.map((rec: string, i: number) => (
                    <li key={i} className="flex gap-3 text-sm text-slate-600">
                      <span className="flex-shrink-0 w-6 h-6 bg-brand-50 text-brand-700 rounded-full flex items-center justify-center text-xs font-bold">
                        {i + 1}
                      </span>
                      <span className="pt-0.5">{rec}</span>
                    </li>
                  ))}
                </ol>
              </Card>
            )}
          </div>
        )}

        {/* Tab: Agent Trace */}
        {tab === "trace" && (
          <div className="space-y-6">
            {report?.plan && (
              <Card>
                <SectionTitle icon="🧠">Planner Strategy</SectionTitle>
                <p className="text-sm text-slate-600 mb-4 leading-relaxed">{report.plan.reasoning}</p>
                <div className="flex flex-wrap gap-2 mb-3">
                  {(report.plan.model_shortlist ?? []).map((m: string) => (
                    <span key={m} className="text-xs bg-brand-50 text-brand-700 px-2.5 py-1 rounded-full font-mono">
                      {m}
                    </span>
                  ))}
                </div>
                {report.plan.columns_to_drop?.length > 0 && (
                  <div className="text-sm text-slate-500">
                    <span className="font-medium text-slate-700">Dropped by planner:</span>{" "}
                    {report.plan.columns_to_drop.map((d: any) => d.column).join(", ")}
                  </div>
                )}
                {report.plan.risk_flags?.length > 0 && (
                  <div className="mt-2 text-sm text-amber-700">
                    ⚠ {report.plan.risk_flags.join(" · ")}
                  </div>
                )}
              </Card>
            )}

            {report?.critic_history?.length > 0 && (
              <Card>
                <SectionTitle icon="🔬">Critic Review History</SectionTitle>
                <div className="space-y-3">
                  {report.critic_history.map((c: any, i: number) => {
                    const isRetry = c.verdict?.startsWith("retry");
                    return (
                      <div key={i} className={`p-4 rounded-xl border text-sm ${
                        isRetry ? "bg-amber-50/70 border-amber-200/70" : "bg-emerald-50/70 border-emerald-200/70"
                      }`}>
                        <div className="flex items-center gap-2 mb-1.5">
                          <span className="font-semibold text-slate-800">Pass {i + 1}: {c.verdict}</span>
                          <span className="text-xs text-slate-500">({c.confidence} confidence)</span>
                          {isRetry && <span className="text-xs text-amber-700">↺ retry triggered</span>}
                        </div>
                        <p className="text-slate-600 leading-relaxed">{c.reasoning}</p>
                        {c.concerns?.length > 0 && (
                          <p className="text-xs text-slate-500 mt-1.5">Concerns: {c.concerns.join("; ")}</p>
                        )}
                      </div>
                    );
                  })}
                </div>
              </Card>
            )}

            {report?.token_usage?.length > 0 && (() => {
              const totalTokens = report.token_usage.reduce(
                (s: number, t: any) => s + (t.input_tokens ?? 0) + (t.output_tokens ?? 0), 0
              );
              const totalCost = report.token_usage.reduce((s: number, t: any) => s + (t.cost_usd ?? 0), 0);
              const totalLatency = report.token_usage.reduce((s: number, t: any) => s + (t.latency_s ?? 0), 0);
              return (
                <Card>
                  <SectionTitle icon="📡">LLM Observability</SectionTitle>
                  <div className="grid grid-cols-3 gap-4 mb-5">
                    <div className="bg-slate-50 rounded-xl p-3.5 text-center">
                      <div className="text-xl font-bold text-slate-800 tracking-tight">{totalTokens.toLocaleString()}</div>
                      <div className="text-xs text-slate-500 uppercase tracking-wide mt-0.5">total tokens</div>
                    </div>
                    <div className="bg-slate-50 rounded-xl p-3.5 text-center">
                      <div className="text-xl font-bold text-slate-800 tracking-tight">${totalCost.toFixed(4)}</div>
                      <div className="text-xs text-slate-500 uppercase tracking-wide mt-0.5">est. cost</div>
                    </div>
                    <div className="bg-slate-50 rounded-xl p-3.5 text-center">
                      <div className="text-xl font-bold text-slate-800 tracking-tight">{totalLatency.toFixed(1)}s</div>
                      <div className="text-xs text-slate-500 uppercase tracking-wide mt-0.5">LLM latency</div>
                    </div>
                  </div>
                  <div className="overflow-x-auto -mx-2">
                    <table className="text-sm w-full">
                      <thead>
                        <tr className="border-b border-slate-200">
                          <th className="text-left px-2 py-2 font-medium text-slate-500 text-xs uppercase tracking-wide">Agent</th>
                          <th className="text-left px-2 py-2 font-medium text-slate-500 text-xs uppercase tracking-wide">Model</th>
                          <th className="text-left px-2 py-2 font-medium text-slate-500 text-xs uppercase tracking-wide">Tokens (in/out)</th>
                          <th className="text-left px-2 py-2 font-medium text-slate-500 text-xs uppercase tracking-wide">Cost</th>
                          <th className="text-left px-2 py-2 font-medium text-slate-500 text-xs uppercase tracking-wide">Latency</th>
                        </tr>
                      </thead>
                      <tbody>
                        {report.token_usage.map((t: any, i: number) => (
                          <tr key={i} className="border-b border-slate-100 last:border-0 hover:bg-slate-50/70">
                            <td className="px-2 py-2 capitalize text-slate-700">{t.agent}</td>
                            <td className="px-2 py-2 font-mono text-xs text-slate-500">{t.model}</td>
                            <td className="px-2 py-2 font-mono text-xs text-slate-500">{t.input_tokens}/{t.output_tokens}</td>
                            <td className="px-2 py-2 font-mono text-xs text-slate-500">${t.cost_usd?.toFixed(5)}</td>
                            <td className="px-2 py-2 font-mono text-xs text-slate-500">{t.latency_s?.toFixed(2)}s</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </Card>
              );
            })()}

            {!report?.plan && !report?.critic_history?.length && !report?.token_usage?.length && (
              <Card className="p-10 text-center text-slate-400">
                No agent trace data available for this run.
              </Card>
            )}
          </div>
        )}

        {/* Tab: Download */}
        {tab === "download" && (
          <div className="space-y-4">
            <Card className="p-10 text-center">
              <div className="text-5xl mb-4">📄</div>
              <h2 className="text-xl font-bold text-slate-900 mb-2">Download Report</h2>
              <p className="text-slate-500 mb-6">Get your complete analysis as a PDF report.</p>

              <div className="flex flex-col sm:flex-row gap-3 justify-center">
                <button
                  onClick={handleDownloadPdf}
                  className="px-6 py-3 bg-gradient-to-br from-brand-600 to-violet-600 hover:from-brand-700 hover:to-violet-700 text-white font-semibold rounded-xl transition-colors shadow-soft"
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
                  className="px-6 py-3 bg-slate-100 hover:bg-slate-200 text-slate-700 font-semibold rounded-xl transition-colors"
                >
                  Download JSON
                </button>
              </div>

              <div className="mt-6 pt-6 border-t border-slate-100">
                <p className="text-sm text-slate-500 mb-2">Shareable job ID:</p>
                <code className="bg-slate-100 px-3 py-1.5 rounded-lg font-mono text-sm text-slate-700">{jobId}</code>
              </div>
            </Card>
          </div>
        )}
      </main>
    </div>
  );
}
