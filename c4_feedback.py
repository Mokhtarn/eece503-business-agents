"""Context checks before a C4 feedback write. No extra model/API request.

Customer input, supplied feedback and unanswered shop facts are different cases.
These conservative checks cover known source-grounding failures; they do not
claim to be a general semantic correctness classifier. Accepted notes retain the
customer's actual request rather than a model-invented follow-up question.
"""
import re
from business_agent import _normalise

TOPICS = {
    "returns": r"\b(return|returns|refund|refunds)\b",
    "warranties": r"\b(warranty|warranties|coverage)\b",
    "pickup": r"\b(pickup|pick up|collect|collection)\b",
    "delivery": r"\b(deliver|delivered|delivery|shipping|moving)\b",
    "services": r"\b(setup|restringing|restring|service|services|appointment|appointments)\b",
    "orders": r"\b(order|orders|payment|payments|reserve|reservation)\b",
    "hours": r"\b(hours|opening|open|closed)\b",
}


def topics(text):
    return {name for name, pattern in TOPICS.items() if re.search(pattern, text)}


def feedback_decision(question, message, history, trace, knowledge):
    """Return save/skip plus a public reason code and faithful source text."""
    source = message.strip()
    # A short request to log an earlier question can use the last actual user turn.
    if re.search(r"\b(log|record|save)\b.{0,15}\b(that|it|question)\b", source, re.I) and len(source.split()) < 10:
        prior = [m["content"] for m in history if m.get("role") == "user"]
        if prior:
            source = prior[-1]
    original, proposed = _normalise(source), _normalise(question)

    def skip(code, message):
        return {"save": False, "reason_code": code, "message": message}

    # Actual supplied feedback is allowed even when mixed with a buying question.
    supplied = re.search(r"\b(?:feedback|complaint)\s*:\s*(\S.+)", source, re.I)
    complaint = re.search(r"\b(i am|i m|i was|we are|we were|i feel)\b.{0,15}\b(unhappy|dissatisfied|disappointed)\b|\bi (want|would like) to complain\b", original)
    if supplied or complaint:
        return {"save": True, "question": source, "reason_code": "supplied_customer_feedback"}
    if re.search(r"\bfeedback\b", original) and re.search(r"\b(can|may|how|want to|like to)\b", original):
        return skip("missing_feedback_contents", "Ask for the feedback contents; none have been supplied to save.")

    lead_request = re.search(r"\b(follow up|demo|save|interest|contact)\b", original)
    missing_contact = re.search(r"\b(email|name|contact details)\b", proposed)
    shop_contact = re.search(r"\b(shop|store|your)\b.{0,15}\b(email|phone|address)\b", original)
    if lead_request and missing_contact and not shop_contact:
        return skip("missing_customer_input", "Missing customer contact details are not an unanswered shop fact. Ask the customer; do not log feedback.")
    if re.search(r"\b(tablet|device)\b", original) and (
        re.search(r"\b(unspecified|not specified|not provided)\b", original)
        or re.search(r"\b(what|which)\b.{0,35}\b(tablet model|operating system|model and os)\b", proposed)
    ):
        return skip("missing_customer_input", "Ask for the customer's device model/OS before a specific compatibility inquiry. Do not log missing customer input.")

    observations = [e["observation"] for e in trace if e.get("event") == "observation" and e["observation"].get("success") is True]
    covered = set()
    for observation in observations:
        if "topic" in observation and "policy" in observation:
            covered.update(TOPICS if observation["topic"] == "all" else [observation["topic"]])
    wanted, suggested = topics(original), topics(proposed)
    if suggested - wanted:
        return skip("unasked_question", "The proposed note introduces a topic the customer did not request. Answer the actual request without creating this entry.")

    # General policy questions are answered by the retrieved section. Specific
    # absent terms explicitly requested by the customer may still be logged.
    relevant = suggested & covered
    specific_unknown = bool(
        ("warranties" in relevant and knowledge._named_products(original)
         and re.search(r"\b(duration|years|months|how long|warranty period for)\b", original))
        or re.search(r"\b(installment|financing|rental|rentals)\b", original)
        or re.search(r"\b(pickup|pick up)\b.{0,25}\b(documents|identification|procedure|stairs|charges)\b", original)
        or re.search(r"\b(documents|identification|procedure|stairs|charges)\b.{0,25}\b(pickup|pick up)\b", original)
        or re.search(r"\b(appointment|appointments)\b.{0,25}\b(availability|available|slots|when)\b", original)
        or re.search(r"\b(availability|available|slots|when)\b.{0,25}\b(appointment|appointments)\b", original)
        or ("delivery" in relevant and re.search(r"\b(deliver|delivery|shipping)\b.{0,15}\bto\b", original)
            and not re.search(r"\b(beirut|mount lebanon)\b", original))
    )
    if relevant and not specific_unknown:
        return skip("answered_by_policy", "The retrieved policy answers the customer's general policy request. State its conditions; do not invent an extra question to log.")

    inventory_read = any("items" in o and "special_order_policy" in o for o in observations)
    inventory_question = re.search(r"\b(price|pricing|stock|availability|quote|charges|taxes|accessories)\b", proposed)
    if inventory_question:
        explicit_extra = re.search(r"\b(tax|taxes|accessories|real world|live inventory|all in|including delivery|final quote|binding quote|exact price|final price)\b", original)
        known_ids = knowledge._named_products(original)
        if known_ids and not inventory_read and not explicit_extra and re.search(r"\b(price|cost|stock|availability)\b", original):
            return skip("catalogue_lookup_available", "This is answerable from the catalogue. Use check_inventory for " + ", ".join(known_ids) + "; do not log it as unanswered.")
        if inventory_read and not explicit_extra:
            return skip("answered_by_inventory", "Inventory already supplies the requested simulated figures. Give staff-confirmation caveats without logging an unasked real-world or extra-charge inquiry.")

    # A candidate must be about the actual request; preserve that source verbatim.
    stop = {"the", "a", "an", "is", "are", "for", "to", "of", "and", "or", "what", "does", "do", "can", "you", "your", "customer", "question", "please", "shop", "store"}
    shared = (set(original.split()) & set(proposed.split())) - stop
    if not shared:
        return skip("unasked_question", "The proposed feedback cannot be tied to the customer's actual words. Do not save it.")
    return {"save": True, "question": source, "reason_code": "unanswered_customer_request"}
