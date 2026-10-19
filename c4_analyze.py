"""Summarize actual recorded experiments; never calls a model or fills human ratings."""
import argparse
import csv
import json
import re
import statistics
from collections import Counter
from pathlib import Path


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def run_metrics(row):
    turns = row["turns"]
    actions = [e for t in turns for e in t["trace"] if e["event"] == "action"]
    observations = [e for t in turns for e in t["trace"] if e["event"] == "observation"]
    return {
        "controller_success": len(turns) == len(row["inputs"]) and all(t["error"] is None for t in turns),
        "errors": [t["error"] for t in turns if t["error"]],
        "model_decisions": sum(t["iterations"] for t in turns),
        "tool_requests": len(actions),
        "tool_executions": sum(not e.get("cached", False) and e.get("executed", True) for e in actions),
        "suppressed_feedback_requests": sum(e["tool"] == "record_feedback" and not e.get("cached", False) and not e.get("executed", True) for e in actions),
        "failed_observations": sum(e["observation"].get("success") is not True for e in observations),
        "zero_result_searches": sum(e["tool"] == "search_products" and e["observation"].get("total_matches") == 0 for e in observations),
        "feedback_calls": sum(e["tool"] == "record_feedback" for e in actions),
        "feedback_executions": sum(e["tool"] == "record_feedback" and not e.get("cached", False) and e.get("executed", True) for e in actions),
        "words": sum(len(re.findall(r"\b[\w'-]+\b", t["output"])) for t in turns),
        "input_tokens": sum(t["usage"]["input_tokens"] for t in turns),
        "output_tokens": sum(t["usage"]["output_tokens"] for t in turns),
        "estimated_cost_usd": sum(t["known_usage_cost_usd"] or 0 for t in turns),
        "cost_complete": all(t["estimated_cost_usd"] is not None for t in turns),
        "latency_seconds": sum(t["latency_seconds"] for t in turns),
    }


def summarize(run_dir, output_dir):
    run_dir, output_dir = Path(run_dir), Path(output_dir)
    rows = read_rows(run_dir / "results.jsonl")
    measured = [{"run_index": i, "scenario_id": r["scenario_id"], "variant_id": r["variant_id"], **run_metrics(r)} for i, r in enumerate(rows)]
    by_variant = {}
    for variant in sorted({r["variant_id"] for r in rows}):
        group = [r for r in measured if r["variant_id"] == variant]
        raw = [r for r in rows if r["variant_id"] == variant]
        by_variant[variant] = {
            "jobs": len(group), "controller_successes": sum(r["controller_success"] for r in group),
            "errors": dict(Counter(e for r in group for e in r["errors"])),
            "mean_model_decisions": statistics.mean(r["model_decisions"] for r in group),
            "mean_tool_requests": statistics.mean(r["tool_requests"] for r in group),
            "mean_tool_executions": statistics.mean(r["tool_executions"] for r in group),
            "suppressed_feedback_requests": sum(r["suppressed_feedback_requests"] for r in group),
            "feedback_executions": sum(r["feedback_executions"] for r in group),
            "mean_words": statistics.mean(r["words"] for r in group),
            "mean_latency_seconds": statistics.mean(r["latency_seconds"] for r in group),
            "input_tokens": sum(r["input_tokens"] for r in group), "output_tokens": sum(r["output_tokens"] for r in group),
            "estimated_cost_usd": sum(r["estimated_cost_usd"] for r in group),
            "zero_result_searches": sum(r["zero_result_searches"] for r in group),
            "mean_expected_tool_coverage": statistics.mean(r["deterministic_metrics"]["expected_tool_coverage"] for r in raw),
            "dollar_amount_flags": sum(bool(r["deterministic_metrics"]["unsupported_usd_amounts"]) for r in raw),
        }
    comparisons = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["comparisons"]
    paired = {}
    for name, comparison in comparisons.items():
        left, right = comparison["arms"]
        pairs = []
        for scenario in sorted({r["scenario_id"] for r in rows}):
            a = next((r for r in measured if r["scenario_id"] == scenario and r["variant_id"] == left), None)
            b = next((r for r in measured if r["scenario_id"] == scenario and r["variant_id"] == right), None)
            if a is not None and b is not None:
                raw = rows[a["run_index"]]
                pairs.append({"scenario_id": scenario, "left_run": a["run_index"], "right_run": b["run_index"],
                    "comparison_focus": name in raw.get("comparison_focus", []),
                    "right_minus_left": {k: b[k] - a[k] for k in ("words", "model_decisions", "tool_requests", "estimated_cost_usd")}})
        focused = [p for p in pairs if p["comparison_focus"]]
        def means(group):
            return {k: statistics.mean(p["right_minus_left"][k] for p in group)
                    for k in ("words", "model_decisions", "tool_requests", "estimated_cost_usd")} if group else None
        paired[name] = {**comparison, "paired_scenarios": pairs, "mean_differences_all": means(pairs),
                        "focus_scenario_count": len(focused), "mean_differences_focus": means(focused)}
    summary = {"jobs_recorded": len(rows), "customer_turns": sum(len(r["turns"]) for r in rows),
        "controller_successes": sum(r["controller_success"] for r in measured),
        "errors": dict(Counter(e for r in measured for e in r["errors"])),
        "input_tokens": sum(r["input_tokens"] for r in measured), "output_tokens": sum(r["output_tokens"] for r in measured),
        "estimated_cost_usd": sum(r["estimated_cost_usd"] for r in measured),
        "all_costs_complete": all(r["cost_complete"] for r in measured),
        "variants": by_variant, "paired_comparisons": paired, "runs": measured,
        "caveats": ["One observation per scenario/variant; no significance or repeatability claim.",
                    "Controller success is not semantic correctness.",
                    "Expected-tool coverage can penalize valid alternative paths.",
                    "Dollar flags include valid computed totals/change and negative or conditional mentions; review them.",
                    "No numerical subjective ratings are assigned by this analyzer."]}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output_dir / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(measured[0]))
        writer.writeheader()
        writer.writerows(measured)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.run_dir, args.output)
    print(json.dumps({k: summary[k] for k in ("jobs_recorded", "controller_successes", "estimated_cost_usd", "variants")}, indent=2))
