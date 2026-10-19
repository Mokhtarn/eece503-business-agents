"""Six allowlisted tools backed by unchanged C3 evidence and JSONL conventions."""
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4
from pydantic import BaseModel, ConfigDict, Field, StrictStr, StrictInt, StrictFloat, ValidationError, field_validator
from business_agent import KnowledgeBase, ROOT, WRITE_LOCK, _normalise


class Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class SearchArgs(Arguments):
    query: StrictStr | None = Field(default=None, max_length=200, description="Product name, brand, model or catalogue IDs. Multiple named models are alternatives.")
    instrument_type: Literal["Piano", "Guitar", "Amplifier"] | None = None
    subtype: StrictStr | None = None
    max_price_usd: StrictFloat | None = Field(default=None, ge=0, le=1_000_000)
    specifications: StrictStr | None = Field(default=None, max_length=200)
    use_case: StrictStr | None = Field(default=None, max_length=200)
    limit: StrictInt = Field(default=6, ge=1, le=12)
    offset: StrictInt = Field(default=0, ge=0, le=52)


class ProductArgs(Arguments):
    product_ids: list[StrictStr] = Field(min_length=1, max_length=12)

    @field_validator("product_ids")
    @classmethod
    def ids(cls, values):
        if any(not re.fullmatch(r"(?:KS|ACC)-\d{3}", x) for x in values):
            raise ValueError("Use catalogue IDs such as KS-001 or ACC-001")
        return list(dict.fromkeys(values))


class PolicyArgs(Arguments):
    topic: Literal["returns", "warranties", "pickup", "delivery", "services", "orders", "hours", "showroom", "all"]


class LeadArgs(Arguments):
    email: StrictStr = Field(min_length=3, max_length=254)
    name: StrictStr = Field(min_length=1, max_length=200)
    message: StrictStr = Field(min_length=1, max_length=2000)

    @field_validator("email")
    @classmethod
    def email_format(cls, value):
        if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", value.strip()):
            raise ValueError("email must have a valid name@domain format")
        return value.strip()

    @field_validator("name", "message")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("Must not be blank")
        return value.strip()


class FeedbackArgs(Arguments):
    question: StrictStr = Field(min_length=1, max_length=2000)

    @field_validator("question")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("Must not be blank")
        return value.strip()


MODELS = {"search_products": SearchArgs, "get_product_details": ProductArgs,
          "check_inventory": ProductArgs, "get_store_policy": PolicyArgs,
          "record_customer_interest": LeadArgs, "record_feedback": FeedbackArgs}
DESCRIPTIONS = {
    "search_products": "Search 52 core models. Put names, brands, model codes or IDs in query; specifications contains feature keywords only. Named models are alternatives. Category/subtype/budget and all feature words filter. Use case filters/ranks suitability keywords. Paginate with offset/limit; null means no filter.",
    "get_product_details": "Get reliable full product or accessory records by catalogue ID. Preserve spec notes and missing facts.",
    "check_inventory": "Get simulated USD price, quantity, availability and special-order status by catalogue ID. Not a reservation.",
    "get_store_policy": "Get approved returns, warranties, pickup, delivery, services, orders, hours or showroom policies.",
    "record_customer_interest": "Save customer-supplied name, email and requested follow-up notes locally. Does not email, book or reserve.",
    "record_feedback": "Save explicit feedback or a specific unanswered question locally for review.",
}


def tool_schemas():
    schemas = []
    for name, model in MODELS.items():
        schema = model.model_json_schema()
        if name == "search_products":
            catalogue = json.loads((ROOT / "data/product_catalogue.json").read_text(encoding="utf-8"))
            schema["properties"]["subtype"]["anyOf"][0]["enum"] = sorted({p["subtype"] for p in catalogue["products"]})
        # OpenAI strict schemas require every property, with nullable optional filters.
        schema["required"] = list(schema["properties"])
        for prop in schema["properties"].values():
            prop.pop("default", None)
        schemas.append({"type": "function", "name": name, "description": DESCRIPTIONS[name],
                        "strict": True, "parameters": schema})
    return schemas


TOOL_SCHEMAS = tool_schemas()


class ToolBox:
    def __init__(self, knowledge=None, log_dir=None):
        self.knowledge = knowledge if knowledge is not None else KnowledgeBase()
        # Per-instance sinks avoid changing C3's global AGENT_LOG_DIR in parallel tests.
        self.log_dir = Path(log_dir) if log_dir is not None else Path(os.getenv("C4_LOG_DIR") or os.getenv("AGENT_LOG_DIR") or ROOT / "logs")
        self.products = {**self.knowledge.products, **{p["id"]: p for p in self.knowledge.accessories["items"]}}

    def validate(self, name, arguments):
        if name not in MODELS:
            raise ValueError("Unknown tool")
        if isinstance(arguments, str):
            if len(arguments) > 10000:
                raise ValueError("Tool arguments too long")
            arguments = json.loads(arguments)
        model = MODELS[name].model_validate(arguments)
        if isinstance(model, SearchArgs) and model.subtype is not None:
            if model.subtype not in {p["subtype"] for p in self.knowledge.products.values()}:
                raise ValueError("Unknown subtype; use the catalogue subtype exactly or null")
        if isinstance(model, ProductArgs) and any(pid not in self.products for pid in model.product_ids):
            raise ValueError("Unknown catalogue ID")
        return model.model_dump()

    def execute(self, name, arguments):
        try:
            args = self.validate(name, arguments)
            return getattr(self, name)(**args)
        except (ValueError, TypeError):
            return {"success": False, "error": "Invalid tool name or arguments; check the documented schema and catalogue IDs."}
        except Exception:
            return {"success": False, "error": "The local tool failed; no successful result was confirmed."}

    def search_products(self, query=None, instrument_type=None, subtype=None, max_price_usd=None,
                        specifications=None, use_case=None, limit=6, offset=0):
        words = _normalise(specifications or "").split()
        normalized_query = _normalise(query or "")
        named_ids = set(self.knowledge._named_products(normalized_query))
        # Recover old/misplaced name requests without dropping actual feature filters.
        legacy_ids = set(self.knowledge._named_products(_normalise(specifications or "")))
        recovered_names = bool(legacy_ids)
        if legacy_ids:
            identity_words = {w for pid in legacy_ids for w in re.findall(r"[a-z]+|\d+", _normalise(
                " ".join(str(self.products[pid][k]) for k in ("id", "brand", "model"))))}
            words = [w for w in words if not set(re.findall(r"[a-z]+|\d+", w)) <= identity_words and w not in {"and", "or", "vs", "versus"}]
            if query and not named_ids:
                named_ids = {p["id"] for p in self.knowledge.products.values() if all(
                    w in _normalise(" ".join(str(p[k]) for k in ("id", "brand", "model"))).split()
                    for w in normalized_query.split())} & legacy_ids
            else:
                named_ids = named_ids & legacy_ids if named_ids else legacy_ids
        def name_matches(p):
            if named_ids or legacy_ids:
                return p["id"] in named_ids
            if not normalized_query:
                return True
            identity = _normalise(" ".join(str(p[k]) for k in ("id", "brand", "model")))
            return all(w in identity.split() for w in normalized_query.split())
        name_candidates = [p for p in self.knowledge.products.values() if name_matches(p)]
        use_words = set(_normalise(use_case or "").split()) - {"a", "an", "the", "for", "and", "or", "with", "to", "of", "my", "i", "want", "need"}
        if use_words & {"beginner", "beginners", "first"}:
            use_words |= {"entry", "beginners", "first", "lessons"}
        matches = []
        for p in name_candidates:
            if instrument_type and p["category"] != instrument_type:
                continue
            if subtype and p["subtype"] != subtype:
                continue
            if max_price_usd is not None and p["price_usd"] > max_price_usd:
                continue
            if not all(w in _normalise(p["key_specs"]) for w in words):
                continue
            score = len(use_words & set(_normalise(p["best_for"] + " " + p["tier"]).split()))
            if use_words and score == 0:
                continue
            matches.append((score, p))
        matches.sort(key=lambda item: (-item[0], item[1]["price_usd"], item[1]["id"]))
        fields = ("id", "brand", "model", "category", "subtype", "tier", "key_specs", "best_for", "spec_note")
        return {"success": True, "total_matches": len(matches), "offset": offset,
                "products": [{**{k: p[k] for k in fields}, "use_case_keyword_matches": score} for score, p in matches[offset:offset + limit]],
                "catalogue_name_matches": [{k: p[k] for k in ("id", "brand", "model")} for p in name_candidates] if query or recovered_names else [],
                "recovered_name_filter": recovered_names,
                "lookup_note": "A zero filtered result is not proof a named model is absent. Check catalogue_name_matches; keep budget/features explicit when broadening a query.",
                "filter_note": "Specifications require all literal keywords; use case requires at least one suitability/tier keyword (with beginner aliases) and ranks matches. Call details and inventory for selected IDs.",
                "as_of": self.knowledge.catalogue["as_of"]}

    def get_product_details(self, product_ids):
        return {"success": True, "products": [self.products[pid].copy() for pid in product_ids]}

    def check_inventory(self, product_ids):
        fields = ("id", "brand", "model", "name", "price_usd", "currency", "stock_quantity", "availability_status",
                  "fulfillment_type", "price_type", "price_note", "stock_note", "inventory_last_updated")
        return {"success": True, "simulated": True, "currency": "USD", "as_of": self.knowledge.catalogue["as_of"],
                "items": [{**{k: self.products[pid][k] for k in fields if k in self.products[pid]},
                           "special_order": self.products[pid].get("fulfillment_type") == "staff_quote_special_order"} for pid in product_ids],
                "special_order_policy": self.knowledge.policies["orders"]["special_order"]}

    def get_store_policy(self, topic):
        return {"success": True, "effective_date": self.knowledge.policies["effective_date"],
                "topic": topic, "policy": self.knowledge.policies if topic == "all" else self.knowledge.policies[topic]}

    def _save(self, filename, fields):
        record = {"id": str(uuid4()), "timestamp_utc": datetime.now(timezone.utc).isoformat(), **fields}
        try:
            with WRITE_LOCK:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                with (self.log_dir / filename).open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
        except OSError:
            return {"success": False, "error": "Local log could not be written. Please try again later."}
        logging.getLogger(__name__).info("Tool saved %s record_id=%s", filename, record["id"])
        return {"success": True, "record_id": record["id"], "file": filename}

    def record_customer_interest(self, email, name, message):
        return self._save("customer_interest.jsonl", {"email": email, "name": name, "message": message})

    def record_feedback(self, question):
        return self._save("feedback.jsonl", {"question": question})
