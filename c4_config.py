"""Verified Responses API settings. Reject unknown/unsupported settings explicitly."""
import os
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
DOCS_CHECKED = "2026-10-19"
MODEL_DOC = "https://developers.openai.com/api/docs/models/gpt-6-luna"
SETTINGS_DOC = "https://developers.openai.com/api/docs/guides/latest-model"
# Standard token rates, USD per million. Cached-input discounts deliberately ignored.
RATES = {"gpt-6-luna": (0.10, 0.50), "gpt-6.1-sol": (2.0, 10.0)}


@dataclass(frozen=True)
class ModelConfig:
    model: str = "gpt-6-luna"
    reasoning_effort: str = "none"
    verbosity: str = "medium"
    max_output_tokens: int = 900
    temperature: float | None = None
    top_p: float | None = None

    def __post_init__(self):
        # Only this documented capability profile is verified. Add a reviewed profile
        # before enabling another model, rather than guessing API compatibility.
        if self.model not in RATES:
            raise ValueError(f"Unverified model {self.model!r}; supported: {', '.join(RATES)}")
        if self.reasoning_effort not in {"none", "low", "medium", "high", "xhigh", "max"}:
            raise ValueError("Unsupported reasoning_effort")
        if self.model == "gpt-6.1-sol" and self.reasoning_effort == "none":
            raise ValueError("gpt-6.1-sol requires reasoning_effort low or higher")
        if self.verbosity not in {"low", "medium", "high"}:
            raise ValueError("verbosity must be low, medium or high")
        if type(self.max_output_tokens) is not int or not 128 <= self.max_output_tokens <= 4096:
            raise ValueError("max_output_tokens must be an integer from 128 to 4096")
        for name, value, upper in [("temperature", self.temperature, 2), ("top_p", self.top_p, 1)]:
            if value is not None:
                if type(value) not in (float, int) or not 0 <= value <= upper:
                    raise ValueError(f"{name} must be between 0 and {upper}")
                if self.reasoning_effort != "none":
                    raise ValueError(f"{name} is unsupported with reasoning_effort != none")

    @classmethod
    def from_env(cls, persona="key"):
        load_dotenv(ROOT / ".env", override=False)
        for env in ("C4_TOP_K", "C4_FREQUENCY_PENALTY", "C4_PRESENCE_PENALTY", "C4_SEED"):
            if os.getenv(env) is not None:
                raise ValueError(f"{env} is unsupported by the selected Responses API capability profile")
        defaults = PERSONA_CONFIGS[persona]
        values = asdict(defaults)
        for field, env in [("model", "OPENAI_MODEL"), ("reasoning_effort", "C4_REASONING_EFFORT"),
                           ("verbosity", "C4_VERBOSITY"), ("max_output_tokens", "C4_MAX_OUTPUT_TOKENS"),
                           ("temperature", "C4_TEMPERATURE"), ("top_p", "C4_TOP_P")]:
            if os.getenv(env):
                raw = os.environ[env]
                values[field] = int(raw) if field == "max_output_tokens" else float(raw) if field in {"temperature", "top_p"} else raw
        return cls(**values)

    def options(self):
        result = {"model": self.model, "reasoning": {"effort": self.reasoning_effort},
                  "text": {"verbosity": self.verbosity}, "max_output_tokens": self.max_output_tokens}
        for name in ("temperature", "top_p"):
            if getattr(self, name) is not None:
                result[name] = getattr(self, name)
        return result


# Match tone using response length, with the same low-cost effort and token ceiling.
# Temperature is left unset so persona facts do not depend on sampling noise.
PERSONA_CONFIGS = {"key": ModelConfig(verbosity="medium"), "string": ModelConfig(verbosity="low")}


def estimate_cost(model, input_tokens, output_tokens, cache_write_tokens=0):
    rates = RATES.get(model)
    return None if rates is None else ((input_tokens + 0.25 * cache_write_tokens) * rates[0] + output_tokens * rates[1]) / 1_000_000


@dataclass(frozen=True)
class PromptVariant:
    stepwise: bool = False
    shot_mode: str = "worked"
    instruction_role: str = "system"

    def __post_init__(self):
        if type(self.stepwise) is not bool:
            raise ValueError("stepwise must be a boolean")
        if self.shot_mode not in {"worked", "zero", "few"}:
            raise ValueError("shot_mode must be worked, zero or few")
        if self.instruction_role not in {"system", "user"}:
            raise ValueError("instruction_role must be system or user")
