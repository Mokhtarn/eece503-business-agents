"""Reproducible experiment planning and opt-in execution; never runs on import.

Outputs are actual observations, or clearly labeled mock pipeline checks. They
are not experimental conclusions. Human ratings always start empty.
"""
import argparse
import hashlib
import importlib.metadata
import json
import random
import re
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from c4_config import ModelConfig, PromptVariant, PERSONA_CONFIGS, ROOT
from c4_tools import ToolBox
from react_agent import ReactAgent, SpendingGuard

SCENARIOS_PATH = ROOT / "c4_scenarios.json"
PRIMARY_VARIANTS = ("baseline", "technical_bundle", "stepwise", "zero_shot", "few_shot", "in_message")
CONFIGURATION_SCENARIOS = ("beginner_piano", "guitar_amp", "professional", "multistep")
COMPARISONS = {
    "examples": {"arms": ["zero_shot", "few_shot"], "factor": "zero versus three worked examples"},
    "instruction_placement": {"arms": ["baseline", "in_message"], "factor": "system versus user-message operational instructions"},
    "guidance": {"arms": ["baseline", "stepwise"], "factor": "without versus with explicit stepwise guidance"},
    "persona": {"arms": ["baseline", "technical_bundle"], "factor": "Friendly versus Technical persona, including each persona's verbosity configuration"},
    "configuration": {"arms": ["temperature_low", "temperature_high"], "factor": "temperature 0.2 versus 0.8; persona, prompt and all other settings fixed"},
}


def load_scenarios():
    return json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))


def variants():
    base = ModelConfig()
    prompt = PromptVariant()
    # Reuse the baseline as the system / no-guidance / friendly comparison arm.
    # Persona evaluation deliberately compares each persona's prompt + configuration.
    return {
        "baseline": {"persona": "key", "config": base, "variant": prompt, "compare_to": None, "factor": "baseline"},
        "technical_bundle": {"persona": "string", "config": PERSONA_CONFIGS["string"], "variant": prompt, "compare_to": "baseline", "factor": "persona prompt + persona configuration (bundled comparison)"},
        "stepwise": {"persona": "key", "config": base, "variant": replace(prompt, stepwise=True), "compare_to": "baseline", "factor": "explicit stepwise guidance"},
        "zero_shot": {"persona": "key", "config": base, "variant": replace(prompt, shot_mode="zero"), "compare_to": None, "factor": "zero-shot arm (paired with few_shot)"},
        "few_shot": {"persona": "key", "config": base, "variant": replace(prompt, shot_mode="few"), "compare_to": "zero_shot", "factor": "demonstration count (zero versus three examples)"},
        "in_message": {"persona": "key", "config": base, "variant": replace(prompt, instruction_role="user"), "compare_to": "baseline", "factor": "operational instruction role"},
        "temperature_low": {"persona": "key", "config": replace(base, temperature=0.2), "variant": prompt, "compare_to": None, "factor": "low-temperature arm (paired with temperature_high)"},
        "temperature_high": {"persona": "key", "config": replace(base, temperature=0.8), "variant": prompt, "compare_to": "temperature_low", "factor": "temperature only (0.2 versus 0.8)"},
    }


def deterministic_scores(scenario, results, knowledge):
    actions = [e for r in results for e in r["trace"] if e["event"] == "action"]
    observations = [e for r in results for e in r["trace"] if e["event"] == "observation"]
    called = {e["tool"] for e in actions}
    expected = set(scenario["expected_tools"])
    text = "\n".join(r["output"] for r in results)
    terms = scenario.get("required_terms", [])
    forbidden = scenario.get("forbidden_terms", [])
    ids = set(re.findall(r"\b(?:KS|ACC)-\d{3}\b", text))
    valid_ids = set(knowledge.products) | {p["id"] for p in knowledge.accessories["items"]}
    def amounts(text):
        return {float(x.replace(",", "")) for x in re.findall(r"\$\s*([\d,]+(?:\.\d+)?)", text)}
    supported_amounts = amounts(" ".join(scenario["turns"]))
    def monetary_values(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if "usd" in key and type(value) in (int, float):
                    supported_amounts.add(float(value))
                monetary_values(value)
        elif isinstance(obj, list):
            for value in obj:
                monetary_values(value)
    for observation in observations:
        monetary_values(observation["observation"])
    # Conservative, mechanical proxies, not a claim of semantic correctness.
    return {"expected_tool_coverage": len(called & expected) / len(expected) if expected else None,
            "successful_observation_fraction": sum(e["observation"].get("success") is True for e in observations) / len(observations) if observations else None,
            "required_term_fraction": sum(t.casefold() in text.casefold() for t in terms) / len(terms) if terms else None,
            "forbidden_phrase_hits": [t for t in forbidden if t.casefold() in text.casefold()],
            "unknown_product_ids": sorted(ids - valid_ids),
            "unsupported_usd_amounts": sorted(amounts(text) - supported_amounts),
            "failure_count": sum(bool(r["error"]) for r in results),
            "semantic_correctness": None,
            "limitations": "Keyword/ID checks and expected-tool coverage are proxies; valid alternative tool paths can score lower. Human evidence review is required."}


def make_plan(seed=503, repeats=1, selected_variants=None, scenario_ids=None):
    if repeats < 1:
        raise ValueError("repeats must be positive")
    available = variants()
    # Preserve the original 72-case study; configuration tuning is a small,
    # separately recorded follow-up selected explicitly with --variants.
    selected = list(PRIMARY_VARIANTS) if selected_variants is None else selected_variants
    if set(selected) - set(available):
        raise ValueError("Unknown variant")
    scenarios = load_scenarios()
    if scenario_ids is not None:
        if set(scenario_ids) - {s["id"] for s in scenarios}:
            raise ValueError("Unknown scenario")
        scenarios = [s for s in scenarios if s["id"] in scenario_ids]
    plan = [{"scenario_id": s["id"], "variant_id": v, "repeat": repeat}
            for s in scenarios for v in selected for repeat in range(1, repeats + 1)]
    random.Random(seed).shuffle(plan)
    return plan


def manifest(seed, mode):
    files = [SCENARIOS_PATH, ROOT / "react_agent.py", ROOT / "c4_tools.py", ROOT / "c4_feedback.py", ROOT / "c4_prompts.py", ROOT / "c4_config.py", ROOT / "c4_experiments.py", *sorted((ROOT / "data").glob("*.json"))]
    return {"created_utc": datetime.now(timezone.utc).isoformat(), "seed": seed, "mode": mode,
            "comparisons": COMPARISONS,
            "versions": {x: importlib.metadata.version(x) for x in ("openai", "langgraph", "gradio", "pydantic")},
            "sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            "reproducibility_note": "Schedule/data/config are reproducible; API outputs are stochastic, model aliases can change, and no API seed is assumed."}


def run_plan(plan, output_dir, *, live=False, agent_factory=None, seed=503, budget_usd=0.25):
    if not live and agent_factory is None:
        raise ValueError("Execution requires live=True or an explicit mocked agent_factory")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    mode = "live" if live else "mock_pipeline_validation_not_experiment"
    (output_dir / "manifest.json").write_text(json.dumps(manifest(seed, mode), indent=2), encoding="utf-8")
    (output_dir / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    guard = SpendingGuard(budget_usd)
    scenarios = {s["id"]: s for s in load_scenarios()}
    choices = variants()
    rows = []
    for index, job in enumerate(plan):
        scenario = scenarios[job["scenario_id"]]
        setup = choices[job["variant_id"]]
        tools = ToolBox(log_dir=output_dir / "tool_logs" / str(index))
        agent = agent_factory(setup, tools) if agent_factory else ReactAgent(
            persona=setup["persona"], config=setup["config"], variant=setup["variant"], toolbox=tools, spending_guard=guard)
        history, results = [], []
        for message in scenario["turns"]:
            result = agent.run(message, history)
            results.append(result)
            history.extend([{"role": "user", "content": message}, {"role": "assistant", "content": result["output"]}])
            if result["error"]:
                break
        row = {**job, "mode": mode, "inputs": scenario["turns"], "persona": setup["persona"],
               "model": setup["config"].model, "configuration": asdict(setup["config"]),
               "prompt_variant": asdict(setup["variant"]), "factor": setup["factor"], "compare_to": setup["compare_to"],
               "comparison_focus": scenario.get("comparison_focus", []), "human_review_focus": scenario.get("human_review_focus", []),
               "turns": results, "deterministic_metrics": deterministic_scores(scenario, results, tools.knowledge),
               "human_judgments": {"correctness_1_to_5": None, "helpfulness_1_to_5": None, "persona_fit_1_to_5": None, "notes": None}}
        with (output_dir / "results.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        rows.append(row)
        if results[-1]["error"] in {"spending_limit", "api_failure"}:
            break
    (output_dir / "run_status.json").write_text(json.dumps({"completed_jobs": len(rows), "planned_jobs": len(plan),
        "estimated_spend_usd": guard.estimated_spend, "unsettled_reserve_usd": guard.reserved,
        "budget_usd": budget_usd, "full_plan_finished": len(rows) == len(plan)}, indent=2), encoding="utf-8")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Explicit opt-in to paid API execution")
    parser.add_argument("--output", type=Path, default=ROOT / "tmp" / "c4-experiment-run")
    parser.add_argument("--seed", type=int, default=503)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--scenarios", nargs="+")
    parser.add_argument("--budget", type=float, default=0.25)
    args = parser.parse_args()
    plan = make_plan(args.seed, args.repeats, args.variants, args.scenarios)
    if not args.live:
        print(json.dumps({"status": "plan_only_no_API_calls", "comparisons": COMPARISONS, "jobs": len(plan), "schedule": plan}, indent=2))
    else:
        print(f"Executed {len(run_plan(plan, args.output, live=True, seed=args.seed, budget_usd=args.budget))} jobs; see {args.output}")


if __name__ == "__main__":
    main()
