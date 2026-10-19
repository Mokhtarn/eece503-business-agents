"""C4: manual LangGraph ReAct controller; no prebuilt agent or tool executors.

Only sanitized public outcomes leave run(). Protocol items (including opaque
encrypted reasoning continuation) exist transiently, never in public traces.
"""
import inspect
import json
import os
import time
from dataclasses import asdict
from typing import TypedDict
from openai import OpenAI
from langgraph.graph import StateGraph, START, END
from business_agent import normalise_history, _text, MAX_MESSAGE_CHARS
from c4_config import ModelConfig, PromptVariant, estimate_cost, ROOT
from c4_prompts import PERSONAS, build_input
from c4_tools import ToolBox, TOOL_SCHEMAS
from c4_feedback import feedback_decision


class AgentState(TypedDict):
    customer_message: str
    history: list
    protocol: list
    calls: list
    validated: dict
    observation: dict
    completed: dict
    trace: list
    steps: int
    answer: str
    error: str | None
    usage: dict


class BudgetExceeded(RuntimeError):
    pass


class SpendingGuard:
    """Software estimate only, not billing enforcement. No API retries.

    Reserve a conservative byte-based input bound plus the full output cap before
    every request. Failed/usage-missing requests retain their reservation.
    """
    def __init__(self, limit_usd=0.25):
        self.limit_usd = limit_usd
        self.estimated_spend = 0.0
        self.reserved = 0.0

    def reserve(self, config, request):
        token_bound = len(json.dumps(request, ensure_ascii=False).encode("utf-8")) + 4096
        cost = estimate_cost(config.model, token_bound, config.max_output_tokens, cache_write_tokens=token_bound)
        if cost is None or self.estimated_spend + self.reserved + cost > self.limit_usd:
            raise BudgetExceeded("Smoke-test software spending estimate reached its limit")
        self.reserved += cost
        return cost

    def settle(self, reservation, cost):
        if cost is not None:
            self.reserved -= reservation
            self.estimated_spend += cost


def sdk_supported_options():
    from openai.resources.responses import Responses
    return set(inspect.signature(Responses.create).parameters)


class ReactAgent:
    def __init__(self, persona="key", config=None, variant=None, toolbox=None, client=None,
                 max_iterations=8, spending_guard=None):
        if persona not in PERSONAS:
            raise ValueError("persona must be key or string")
        if type(max_iterations) is not int or not 1 <= max_iterations <= 16:
            raise ValueError("max_iterations must be from 1 to 16")
        self.persona = persona
        self.config = config if config is not None else ModelConfig.from_env(persona)
        self.variant = variant if variant is not None else PromptVariant()
        self.tools = toolbox if toolbox is not None else ToolBox()
        self.client = client
        self.max_iterations = max_iterations
        self.spending_guard = spending_guard
        unsupported = set(self.config.options()) - sdk_supported_options()
        if unsupported:
            raise ValueError(f"Installed OpenAI SDK does not support: {sorted(unsupported)}")
        graph = StateGraph(AgentState)
        for name in ("decide", "validate", "act", "observe", "final"):
            graph.add_node(name, getattr(self, "_" + name))
        graph.add_edge(START, "decide")
        graph.add_conditional_edges("decide", self._decision_edge, {"validate": "validate", "final": "final"})
        graph.add_conditional_edges("validate", lambda s: "final" if s["error"] else "act", {"final": "final", "act": "act"})
        graph.add_edge("act", "observe")
        graph.add_conditional_edges("observe", lambda s: "final" if s["error"] else "decide", {"final": "final", "decide": "decide"})
        graph.add_edge("final", END)
        self.graph = graph.compile()

    def _client(self):
        if self.client is None:
            # Also load privately when callers supply an explicit ModelConfig.
            from dotenv import load_dotenv
            load_dotenv(ROOT / ".env", override=False)
            key = os.getenv("OPENAI_API_KEY", "").strip()
            if not key or key in {"replace_with_your_api_key", "your_private_key_here", "your_api_key_here"}:
                raise ValueError("Missing API key")
            self.client = OpenAI(api_key=key, timeout=40, max_retries=0)
        return self.client

    @staticmethod
    def _decision_edge(state):
        return "final" if state["error"] or not state["calls"] else "validate"

    def _decide(self, state):
        if state["steps"] >= self.max_iterations:
            return {"error": "max_iterations", "calls": []}
        step = state["steps"] + 1
        request = {**self.config.options(), "input": state["protocol"], "tools": TOOL_SCHEMAS,
                   "tool_choice": "auto", "parallel_tool_calls": False, "store": False,
                   "include": ["reasoning.encrypted_content"]}
        reservation = 0
        try:
            client = self._client()
            if self.spending_guard:
                reservation = self.spending_guard.reserve(self.config, request)
            response = client.responses.create(**request)
            usage = dict(state["usage"])
            reported = response.usage
            cost = None
            if reported is not None:
                for name in ("input_tokens", "output_tokens", "total_tokens"):
                    usage[name] += getattr(reported, name)
                cache_writes = getattr(reported.input_tokens_details, "cache_write_tokens", 0) or 0
                usage["cache_write_tokens"] += cache_writes
                usage["requests_with_usage"] += 1
                cost = estimate_cost(self.config.model, reported.input_tokens, reported.output_tokens, cache_writes)
            else:
                usage["complete"] = False
            if self.spending_guard:
                self.spending_guard.settle(reservation, cost)
            if response.status != "completed":
                return {"steps": step, "usage": usage, "error": "incomplete_response", "calls": []}
            calls = [{"call_id": item.call_id, "name": item.name, "arguments": item.arguments}
                     for item in response.output if item.type == "function_call"]
            # Opaque continuation is sent only back to the API, never displayed/saved.
            protocol = state["protocol"] + [item.model_dump(exclude_none=True) for item in response.output]
            answer = response.output_text.strip() if not calls else ""
            trace = state["trace"] + [{"step": step, "event": "decision", "decision": "tool" if calls else "final",
                                        "tools": [c["name"] for c in calls], "tool_requests": calls}]
            return {"steps": step, "usage": usage, "protocol": protocol, "calls": calls,
                    "answer": answer, "trace": trace, "error": None if calls or answer else "empty_response"}
        except BudgetExceeded:
            return {"steps": state["steps"], "error": "spending_limit", "calls": []}
        except Exception as exc:
            # No raw exception text: SDK errors can contain credentials or customer data.
            usage = {**state["usage"], "complete": False}
            return {"steps": step, "usage": usage, "error": "api_failure", "calls": [],
                    "trace": state["trace"] + [{"step": step, "event": "error", "error": "api_failure",
                                               "error_type": type(exc).__name__, "http_status": getattr(exc, "status_code", None)}]}

    def _validate(self, state):
        try:
            if len(state["calls"]) != 1:
                raise ValueError("Only sequential single-tool decisions are supported")
            call = state["calls"][0]
            if not isinstance(call["call_id"], str) or not call["call_id"]:
                raise ValueError("Missing call id")
            args = self.tools.validate(call["name"], call["arguments"])
            return {"validated": args}
        except (ValueError, TypeError):
            return {"error": "invalid_tool_request", "trace": state["trace"] + [
                {"step": state["steps"], "event": "error", "error": "invalid_tool_request"}]}

    def _act(self, state):
        call = state["calls"][0]
        signature = call["name"] + ":" + json.dumps(state["validated"], sort_keys=True)
        completed = dict(state["completed"])
        cached = signature in completed
        guard = None
        executed = not cached
        args = state["validated"]
        if not cached:
            try:
                if call["name"] == "record_feedback":
                    guard = feedback_decision(args["question"], state["customer_message"], state["history"], state["trace"], self.tools.knowledge)
                    if guard["save"]:
                        args = {"question": guard["question"]}
                    else:
                        executed = False
                        completed[signature] = {"success": True, "saved": False, "status": "skipped",
                                                "reason_code": guard["reason_code"], "message": guard["message"]}
                if signature not in completed:
                    completed[signature] = self.tools.execute(call["name"], args)
            except Exception:
                completed[signature] = {"success": False, "error": "Local tool execution failed"}
        return {"completed": completed, "observation": completed[signature], "trace": state["trace"] + [
            {"step": state["steps"], "event": "action", "tool": call["name"],
             "arguments": state["validated"], "executed_arguments": args if executed else None,
             "cached": cached, "executed": executed, "guard": guard}]}

    def _observe(self, state):
        call, observation = state["calls"][0], state["observation"]
        return {"protocol": state["protocol"] + [{"type": "function_call_output", "call_id": call["call_id"],
                                                   "output": json.dumps(observation, ensure_ascii=False)}],
                "trace": state["trace"] + [{"step": state["steps"], "event": "observation", "tool": call["name"],
                                            "observation": observation}],
                "error": None if observation.get("success") is True else "tool_failure"}

    def _final(self, state):
        answer = state["answer"]
        if state["error"]:
            messages = {"max_iterations": "I reached the lookup limit before completing your request. Try a narrower question.",
                        "api_failure": "I couldn't reach the model. Please check the API key, model access, network or quota and try again.",
                        "invalid_tool_request": "I couldn't validate a requested lookup, so I stopped safely. Please rephrase your request.",
                        "tool_failure": "A shop lookup or local save failed, so I stopped. Please try again later.",
                        "spending_limit": "The software spending estimate reached its limit; no further API request was made."}
            answer = messages.get(state["error"], "The model did not finish a usable answer. Please try a shorter request.")
            if any(e.get("event") == "action" and e.get("tool", "").startswith("record_") for e in state["trace"]):
                answer += " A local record may already have been saved; check local logs before resubmitting."
        return {"answer": answer, "trace": state["trace"] + [{"step": state["steps"], "event": "final", "text": answer,
                                                               "error": state["error"]}]}

    def run(self, message, history=()):
        start = time.monotonic()
        message = _text(message, "message", MAX_MESSAGE_CHARS)
        history = normalise_history(history)
        protocol = build_input(self.persona, self.variant, self.tools.knowledge, history, message)
        initial = {"customer_message": message, "history": history, "protocol": protocol, "calls": [], "validated": {}, "observation": {}, "completed": {},
                   "trace": [], "steps": 0, "answer": "", "error": None,
                   "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cache_write_tokens": 0, "requests_with_usage": 0, "complete": True}}
        state = self.graph.invoke(initial, {"recursion_limit": self.max_iterations * 5 + 5})
        usage = state["usage"]
        estimate = estimate_cost(self.config.model, usage["input_tokens"], usage["output_tokens"], usage["cache_write_tokens"])
        return {"output": state["answer"], "trace": state["trace"], "iterations": state["steps"],
                "persona": self.persona, "model": self.config.model, "configuration": asdict(self.config),
                "prompt_variant": asdict(self.variant), "usage": usage,
                "estimated_cost_usd": estimate if usage["complete"] else None,
                "known_usage_cost_usd": estimate, "cost_note": "Standard token estimate; cached discounts ignored. Not billing enforcement.",
                "error": state["error"], "latency_seconds": time.monotonic() - start}

    def chat(self, message, history=()):
        try:
            return self.run(message, history)["output"]
        except (ValueError, TypeError):
            return "Please enter a nonempty message of at most 2,000 characters and check C4 configuration."
        except Exception:
            return "I couldn't finish this request. Please check the local configuration and try again."
