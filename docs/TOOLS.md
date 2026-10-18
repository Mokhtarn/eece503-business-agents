# Tool schemas

The agent has two local functions, defined in `business_agent.py` and passed to
OpenAI through `TOOL_SCHEMAS`. Each schema specifies the function name,
description and required string parameters, with `strict: true` and
`additionalProperties: false`.

| Function | Required parameters | Effect |
| --- | --- | --- |
| `record_customer_interest` | `email`, `name`, `message` | Save customer contact details and interest to `logs/customer_interest.jsonl` |
| `record_feedback` | `question` | Save feedback or an unanswered question to `logs/feedback.jsonl` |

Successful output:

```json
{"success": true, "record_id": "generated UUID", "file": "feedback.jsonl"}
```

Failed output:

```json
{"success": false, "error": "Reason for failure"}
```

Arguments must match the schema and contain nonempty strings. Email must have a
`name@domain.tld` format. Limits are 254 characters for email, 200 for name and
2,000 for notes or a question. The dispatcher returns validation or write errors
to the model so it can explain the failure.

Every saved record includes an ID and UTC timestamp. Records append across app
restarts. `AGENT_LOG_DIR` can override the default `logs/` directory, which is
excluded from Git. The terminal shows saved record IDs without contact details.

The model requests a tool, Python executes it, and the result is returned with
the original `tool_call_id`. Identical calls execute once per turn. The chat loop
allows two tool rounds before requesting a final answer with tools disabled.

The agent collects missing name, email and interest before saving a lead. It
asks for feedback contents before logging a request to leave feedback. A
successful result means a local record was saved; neither tool sends email,
reserves stock, books an appointment or places an order.
