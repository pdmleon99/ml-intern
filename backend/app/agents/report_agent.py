import asyncio
import json
import os

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
)
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.llm_factory import get_llm
from app.tools.pdf_tools import b64_to_image, build_cover_table, dark_table_style, get_styles

from .state import AgentState

FALLBACK_RECOMMENDATIONS = [
    "Collect more data to improve model reliability.",
    "Investigate the top features for business insight.",
    "Run hyperparameter tuning on the best model.",
    "Monitor model performance over time for drift.",
    "Consider feature interactions for potential gains.",
]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _call_llm_recommendations(llm, summary: dict, audit_lines: list) -> list:
    audit_text = "\n".join(audit_lines) if audit_lines else "(none recorded)"
    prompt = f"""You are a senior data scientist writing next-step recommendations after an automated ML pipeline has already run.

Problem: {summary.get("problem", "")}
Best model: {summary.get("best_model", "N/A")}
Metrics: {json.dumps(summary.get("metrics", {}), indent=2)}
Dataset: {summary.get("n_rows", "?")} rows, problem type: {summary.get("problem_type", "?")}

ACTIONS ALREADY TAKEN BY THE PIPELINE (DO NOT RECOMMEND THESE AGAIN):
{audit_text}

Write EXACTLY 5 recommendations for what the data scientist should do NEXT — things NOT already done above.
Focus on: model improvement, data collection, deployment, monitoring, business impact, or advanced techniques the automated pipeline cannot do.

Reply with ONLY a JSON array of 5 strings. No markdown. No numbering.
["rec 1", "rec 2", "rec 3", "rec 4", "rec 5"]"""
    response = await asyncio.to_thread(llm.invoke, prompt)
    raw = response.content.replace("```json", "").replace("```", "").strip()
    result = json.loads(raw)
    if isinstance(result, list) and len(result) >= 1:
        return result[:5]
    raise ValueError("Invalid recommendations format")


async def run_report_agent(state: AgentState) -> AgentState:
    state["current_agent"] = "report"
    state["progress_pct"] = 92
    state["messages"].append("📄 Generating report...")

    try:
        llm = get_llm(state["llm_config"])
        styles = get_styles()

        # ── 1. LLM RECOMMENDATIONS ────────────────────────────────────────
        summary_for_llm = {
            "problem": state.get("user_description", ""),
            "best_model": state.get("best_model_name", "N/A"),
            "metrics": {
                k: v for k, v in (state.get("evaluation_metrics") or {}).items()
                if k != "classification_report"
            },
            "n_rows": (state.get("profile") or {}).get("n_rows", "unknown"),
            "problem_type": state.get("problem_type", "unknown"),
        }

        # Build audit trail so LLM doesn't re-suggest completed actions
        audit_lines = []
        for t in state.get("features_applied", []):
            cols = t.get("columns_affected", [])
            cols_str = ", ".join(str(c) for c in cols[:5])
            if len(cols) > 5:
                cols_str += f" (+{len(cols)-5} more)"
            audit_lines.append(f"- DONE: {t.get('type')} on [{cols_str}] — {t.get('rationale','')[:80]}")
        for d in state.get("dropped_columns", []):
            audit_lines.append(f"- DONE: dropped '{d.get('column')}' — {d.get('reason','')[:80]}")

        try:
            recommendations = await _call_llm_recommendations(llm, summary_for_llm, audit_lines)
        except Exception as e:
            error_msg = str(e).replace(state["llm_config"].get("api_key", ""), "[REDACTED]")
            state["warnings"].append(f"LLM recommendations failed: {error_msg[:100]}")
            recommendations = FALLBACK_RECOMMENDATIONS

        # ── 2. BUILD PDF (synchronous reportlab rendering — off the event loop) ──
        pdf_path = await asyncio.to_thread(_build_pdf_sync, state, styles, recommendations)
        state["report_pdf_path"] = pdf_path

        # ── 3. REPORT JSON ────────────────────────────────────────────────
        state["report_json"] = {
            "job_id": state["job_id"],
            "dataset_profile": state.get("profile"),
            "problem_type": state.get("problem_type"),
            "target_column": state.get("target_column"),
            "eda_findings": state.get("eda_findings", []),
            "eda_charts": state.get("eda_charts", []),
            "eda_narrative": state.get("eda_narrative"),
            "plan": state.get("plan"),
            "features_applied": state.get("features_applied", []),
            "dropped_columns": state.get("dropped_columns", []),
            "feature_names": state.get("feature_names", []),
            "experiments": state.get("experiments", []),
            "baseline_score": state.get("baseline_score"),
            "critic_history": state.get("critic_history", []),
            "best_model_name": state.get("best_model_name"),
            "evaluation_metrics": state.get("evaluation_metrics"),
            "evaluation_charts": state.get("evaluation_charts", []),
            "model_explanation": state.get("model_explanation"),
            "statistical_comparison": state.get("statistical_comparison"),
            "shap_summary": state.get("shap_summary"),
            "recommendations": recommendations,
            "token_usage": state.get("token_usage", []),
            "trace_events": state.get("trace_events", []),
        }

        state["status"] = "completed"
        state["progress_pct"] = 100
        state["messages"].append("✅ Report ready — analysis complete!")

    except Exception as e:
        error_msg = str(e).replace(state["llm_config"].get("api_key", ""), "[REDACTED]")
        state["errors"].append(f"Report agent: {error_msg}")
        state["messages"].append(f"⚠ Report generation error: {error_msg[:100]}")
        state["status"] = "failed"

    return state


def _build_pdf_sync(state: AgentState, styles: dict, recommendations: list) -> str:
    os.makedirs("./data/storage", exist_ok=True)
    pdf_path = f"./data/storage/{state['job_id']}_report.pdf"
    doc = SimpleDocTemplate(
        pdf_path, pagesize=A4,
        rightMargin=0.75 * inch, leftMargin=0.75 * inch,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
    )
    story = []

    # ─ Cover page ───────────────────────────────────────────────────
    story.append(Spacer(1, 0.8 * inch))
    story.append(Paragraph("ML Intern", styles["title"]))
    story.append(Paragraph("Autonomous Data Science Report", styles["h2"]))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#3498db")))
    story.append(Spacer(1, 0.3 * inch))
    story.append(build_cover_table(state))
    story.append(PageBreak())

    # ─ EDA Section ──────────────────────────────────────────────────
    story.append(Paragraph("1. Dataset Overview", styles["h1"]))
    eda_narrative = state.get("eda_narrative") or {}
    if isinstance(eda_narrative, dict) and eda_narrative.get("summary"):
        story.append(Paragraph(eda_narrative["summary"], styles["body"]))
        if eda_narrative.get("key_findings"):
            story.append(Paragraph("Key findings:", styles["h2"]))
            for kf in eda_narrative["key_findings"]:
                story.append(Paragraph(f"• {kf}", styles["body"]))

    findings = state.get("eda_findings", [])
    if findings:
        story.append(Paragraph("Dataset Findings", styles["h2"]))
        findings_data = [["Severity", "Finding", "Column"]]
        for f in findings[:30]:
            findings_data.append([
                f.get("severity", ""),
                f.get("finding", "")[:80],
                f.get("column", "—")[:30],
            ])
        ft = Table(findings_data, colWidths=[1 * inch, 4 * inch, 1.5 * inch])
        ft.setStyle(dark_table_style())
        story.append(ft)
        story.append(Spacer(1, 0.2 * inch))

    for i, chart in enumerate(state.get("eda_charts", [])):
        try:
            story.append(Paragraph(chart.get("title", ""), styles["h2"]))
            story.append(b64_to_image(chart["base64_png"]))
            story.append(Paragraph(chart.get("description", ""), styles["body"]))
            story.append(Spacer(1, 0.2 * inch))
            if (i + 1) % 2 == 0:
                story.append(PageBreak())
        except Exception as e:
            state["warnings"].append(f"Chart render failed in PDF: {str(e)[:60]}")

    story.append(PageBreak())

    # ─ Feature Engineering ──────────────────────────────────────────
    story.append(Paragraph("2. Feature Engineering", styles["h1"]))
    story.append(Paragraph(
        f"Applied {len(state.get('features_applied', []))} transformations. "
        f"Dropped {len(state.get('dropped_columns', []))} columns. "
        f"Final feature count: {len(state.get('feature_names', []))}.",
        styles["body"],
    ))

    fe = state.get("features_applied", [])
    if fe:
        story.append(Paragraph("Transformations Applied", styles["h2"]))
        fe_data = [["Transformation", "Type", "Rationale"]]
        for f in fe[:20]:
            fe_data.append([
                (f.get("name") or "")[:30],
                (f.get("type") or "")[:20],
                (f.get("rationale") or "")[:60],
            ])
        ft = Table(fe_data, colWidths=[2 * inch, 1.5 * inch, 3 * inch])
        ft.setStyle(dark_table_style())
        story.append(ft)

    dropped = state.get("dropped_columns", [])
    if dropped:
        story.append(Paragraph("Dropped Columns", styles["h2"]))
        dr_data = [["Column", "Reason"]]
        for d in dropped[:20]:
            dr_data.append([d.get("column", "")[:30], d.get("reason", "")[:60]])
        dt = Table(dr_data, colWidths=[2 * inch, 4.5 * inch])
        dt.setStyle(dark_table_style())
        story.append(dt)

    story.append(PageBreak())

    # ─ Model Comparison ─────────────────────────────────────────────
    story.append(Paragraph("3. Model Comparison", styles["h1"]))
    completed_exp = [
        e for e in state.get("experiments", [])
        if e.get("status") in ("completed", "completed_reduced")
    ]
    completed_exp.sort(key=lambda e: e.get("primary_score", 0), reverse=True)

    if completed_exp:
        pm = completed_exp[0].get("primary_metric", "score")
        mc_data = [["Model", f"{pm} (mean)", f"{pm} (±std)", "Time (s)"]]
        for exp in completed_exp:
            cv = exp.get("cv_scores", {})
            pm_data = cv.get(f"test_{pm}", cv.get("test_r2", {}))
            mc_data.append([
                exp["model_name"],
                f"{pm_data.get('mean', 0):.4f}",
                f"±{pm_data.get('std', 0):.4f}",
                f"{exp.get('train_time_s', 0):.1f}s",
            ])
        mt = Table(mc_data, colWidths=[2.5 * inch, 1.5 * inch, 1.5 * inch, 1 * inch])
        style = dark_table_style()
        style.add("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#d5f5e3"))
        mt.setStyle(style)
        story.append(mt)

    story.append(PageBreak())

    # ─ Best Model Analysis ──────────────────────────────────────────
    best_name = state.get("best_model_name") or "N/A"
    story.append(Paragraph(f"4. Best Model: {best_name}", styles["h1"]))
    if state.get("model_explanation"):
        story.append(Paragraph(state["model_explanation"], styles["body"]))
    story.append(Spacer(1, 0.2 * inch))

    # ─ Statistical Confidence — the "is this actually better than nothing?" answer ─
    stat_comp = state.get("statistical_comparison")
    if stat_comp:
        story.append(Paragraph("Statistical Confidence", styles["h2"]))
        ci = stat_comp.get("test_set_ci") or {}
        if ci.get("mean") is not None:
            story.append(Paragraph(
                f"Test-set {stat_comp.get('primary_metric', 'score')}: {ci['mean']:.4f} "
                f"(95% CI: {ci['ci_low']:.4f}–{ci['ci_high']:.4f})",
                styles["body"],
            ))
        vs_base = stat_comp.get("vs_baseline")
        if vs_base and vs_base.get("lift") is not None:
            pct = f"{vs_base['lift_pct']:+.1%}" if vs_base.get("lift_pct") is not None else "n/a"
            story.append(Paragraph(
                f"Compared to a naive baseline: {pct} lift ({vs_base['confidence']} confidence, "
                f"P(better than baseline) = {vs_base['probability_better_than_baseline']:.2f}). "
                f"This is the honest answer to whether the model learned anything real.",
                styles["body"],
            ))
        story.append(Spacer(1, 0.15 * inch))

    critic_history = state.get("critic_history") or []
    if critic_history:
        story.append(Paragraph("Agent Self-Review (Critic)", styles["h2"]))
        for i, c in enumerate(critic_history, 1):
            story.append(Paragraph(
                f"Pass {i}: {c.get('verdict')} ({c.get('confidence')} confidence) — "
                f"{c.get('reasoning', '')[:200]}",
                styles["body"],
            ))
        story.append(Spacer(1, 0.2 * inch))

    for chart in state.get("evaluation_charts", []):
        try:
            story.append(Paragraph(chart.get("title", ""), styles["h2"]))
            story.append(b64_to_image(chart["base64_png"]))
            story.append(Paragraph(chart.get("description", ""), styles["body"]))
            story.append(Spacer(1, 0.2 * inch))
        except Exception as e:
            state["warnings"].append(f"Eval chart render failed: {str(e)[:60]}")

    story.append(PageBreak())

    # ─ Recommendations ──────────────────────────────────────────────
    story.append(Paragraph("5. Recommendations & Next Steps", styles["h1"]))
    for i, rec in enumerate(recommendations, 1):
        story.append(Paragraph(f"{i}. {rec}", styles["body"]))
        story.append(Spacer(1, 0.1 * inch))

    doc.build(story)
    return pdf_path
