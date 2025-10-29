import os
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import dotenv


try:
    import google.generativeai as genai
except Exception:
    genai = None

dotenv.load_dotenv()

# ------------------------------------------------------------
# 1. Load & merge all result CSVs
# ------------------------------------------------------------

def load_results(results_path: str | Path):
    results_path = Path(results_path)
    files = sorted(results_path.glob("*.csv"))
    datasets = {}
    for f in files:
        name = f.stem.split("_")[0]
        df = pd.read_csv(f)
        if name not in datasets:
            datasets[name] = df
        else:
            datasets[name] = pd.concat([datasets[name], df], axis=1)
    print(f"[INFO] Loaded {len(datasets)} datasets with merged scores.")
    # clean duplicates / NaNs
    for name, df in datasets.items():
        df = df.loc[:, ~df.columns.duplicated()].dropna(axis=0, how="any")
        datasets[name] = df
    return datasets


# ------------------------------------------------------------
# 2. Compute statistics, correlations, and save JSON summary
# ------------------------------------------------------------

def compute_summary(datasets: dict, results_path: Path):
    gemini_summary = {}
    for name, df in datasets.items():
        gemini_summary[name] = {
            "method_statistics": df.describe().T.to_dict(),
            "method_correlations": df.corr().to_dict(),
            "mean_scores": df.mean().to_dict(),
            "std_scores": df.std().to_dict(),
        }

    json_path = results_path / "gemini_summary.json"
    with open(json_path, "w") as f:
        json.dump(gemini_summary, f, indent=2)
    print(f"[INFO] Saved Gemini analysis summary → {json_path}")
    return gemini_summary


# ------------------------------------------------------------
# 3. Gemini Prompt & API Call
# ------------------------------------------------------------

SYSTEM_BRIEF = """
You are an expert AI Data Scientist specializing in anomaly detection for robotics time-series and multi-sensor data.
You will analyze method metrics and correlations and produce a precise, structured report.
Keep claims grounded in the provided numbers.
"""

def build_prompt(data: dict) -> str:
    reduced = {}
    for ds, obj in data.items():
        keep = {
            "mean_scores": obj.get("mean_scores", {}),
            "std_scores": obj.get("std_scores", {}),
            "method_correlations": obj.get("method_correlations", {}),
        }
        reduced[ds] = keep

    json_spec = {
        "schema_version": "1.0",
        "project_title": "Algebraic and AI-Assisted Anomaly Detection for Robotic Sensor Data",
        "per_dataset": {
            "<dataset_name>": {
                "best_detectors": [
                    {"method": "<name>", "reason": "<short reason>",
                     "supporting_numbers": {"mean": 0.0, "std": 0.0, "corr_to_others": {"PCA_Q": 0.0}}}
                ],
                "hypothesis": ["<numbered hypotheses>"],
                "numerical_patterns": ["<quantitative patterns>"],
                "notes": "<optional note>"
            }
        },
        "cross_sensor_insights": [],
        "industrial_relevance_insights": [],
        "general_final_summary": ""
    }

    instructions = {
        "task": "Analyze anomaly detector results and produce a structured JSON following the schema.",
        "requirements": [
            "Identify 'best_detectors' with supporting numbers.",
            "Generate 2–4 concise 'hypothesis' per dataset.",
            "List 2–5 'numerical_patterns' per dataset.",
            "Provide 3–6 'cross_sensor_insights'.",
            "Provide 3–6 'industrial_relevance_insights'.",
            "Finish with a concise 'general_final_summary'.",
            "Ground all insights in the numeric data provided."
        ],
        "json_contract": json_spec
    }

    prompt = {
        "system_brief": SYSTEM_BRIEF.strip(),
        "instructions": instructions,
        "data": reduced
    }
    return json.dumps(prompt, indent=2)


def call_gemini(prompt_text: str, model_name="gemini-2.0-flash",
                temperature=0.2, top_p=0.9, top_k=40,
                max_output_tokens=3500, retries=3, backoff=2.0):
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        print("[WARNING] GEMINI_API_KEY not set — skipping API call.")
        return "[ERROR] GEMINI_API_KEY missing."
    if genai is None:
        print("[ERROR] google-generativeai not installed.")
        return "[ERROR] missing SDK."

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = model.generate_content(
                prompt_text,
                generation_config=dict(
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    max_output_tokens=max_output_tokens,
                ),
            )
            if hasattr(resp, "text") and resp.text:
                return resp.text
            if hasattr(resp, "candidates") and resp.candidates:
                return resp.candidates[0].content.parts[0].text
            return str(resp)
        except Exception as e:
            last_err = e
            print(f"[WARNING] Gemini call failed ({attempt}/{retries}): {e}")
            time.sleep(backoff * attempt)
    return f"[ERROR] Gemini call failed: {last_err}"


# ------------------------------------------------------------
# 4. Robust JSON parse + Markdown renderer
# ------------------------------------------------------------

def parse_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                return None
    return None


def render_markdown(report: dict) -> str:
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# AI Analysis Report — Gemini 2.0 Flash",
        "",
        f"_Generated: {now}_",
        "",
        f"**Project:** {report.get('project_title','(unknown)')}  ",
        f"**Schema:** {report.get('schema_version','?')}",
        ""
    ]
    for ds, block in (report.get("per_dataset") or {}).items():
        block = block or {}
        lines.append(f"## Dataset: {ds}\n")
        lines.append("### Best Detectors")
        for bd in block.get("best_detectors", []):
            lines.append(f"- **{bd.get('method','?')}** — {bd.get('reason','')}")
            lines.append(f"  Numbers: `{json.dumps(bd.get('supporting_numbers',{}))}`")
        lines.append("\n### Hypotheses")
        for h in block.get("hypothesis", []):
            lines.append(f"- {h}")
        lines.append("\n### Numerical Patterns")
        for p in block.get("numerical_patterns", []):
            lines.append(f"- {p}")
        notes = (block.get("notes") or "").strip()
        if notes:
            lines.append("\n### Notes\n" + notes)
        lines.append("")

    lines.append("## Cross-Sensor Insights")
    for p in report.get("cross_sensor_insights", []):
        lines.append(f"- {p}")
    lines.append("")
    lines.append("## Industrial Relevance Insights")
    for p in report.get("industrial_relevance_insights", []):
        lines.append(f"- {p}")
    lines.append("")
    lines.append("## General Final Summary")
    lines.append(report.get("general_final_summary", "(none)"))
    lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------
# 5. Master evaluation pipeline
# ------------------------------------------------------------

def run_evaluation(results_dir="./results", ai_report_dir="./ai_report"):
    results_path = Path(results_dir)
    ai_report_path = Path(ai_report_dir)
    ai_report_path.mkdir(exist_ok=True)

    # Load + summarize
    datasets = load_results(results_path)
    summary = compute_summary(datasets, results_path)

    # Build prompt & save
    prompt = build_prompt(summary)
    prompt_path = ai_report_path / "gemini_prompt.json"
    with open(prompt_path, "w") as f:
        f.write(prompt)
    print(f"[INFO] Prompt saved → {prompt_path}")

    # Call Gemini
    raw_output = call_gemini(prompt)
    raw_txt_path = ai_report_path / "gemini_raw.txt"
    with open(raw_txt_path, "w") as f:
        f.write(raw_output)
    print(f"[INFO] Raw Gemini output saved → {raw_txt_path}")

    # Parse + save
    parsed = parse_json(raw_output) or {}
    report_json_path = ai_report_path / "report.json"
    with open(report_json_path, "w") as f:
        json.dump(parsed, f, indent=2)
    print(f"[INFO] Parsed report JSON saved → {report_json_path}")

    # Render markdown
    report_md = render_markdown(parsed)
    report_md_path = ai_report_path / "report.md"
    with open(report_md_path, "w") as f:
        f.write(report_md)
    print(f"[INFO] Markdown report saved → {report_md_path}")

    print("\n[INFO] Gemini evaluation complete.")
    print(f"- JSON → {report_json_path}")
    print(f"- Markdown → {report_md_path}")