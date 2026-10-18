# Keys & Strings - EECE 503 C3

A customer-support chatbot for a fictional piano and guitar shop in Beirut,
built with Python, OpenAI and Gradio. It answers business and product questions,
collects customer interest, and logs feedback or unanswered questions.

[Demo video](demo/MokhtarNasser_keys%26strings_demo.mp4)

The catalogue contains 52 instrument and amplifier models and 14 generic
accessories. Store prices and stock are simulated for this project. Special-order
prices require a staff quote. The chatbot saves local records; it cannot send
email, take payment, reserve stock or confirm appointments.

## Setup

Run these commands from the repository root in PowerShell. Python 3.11 or newer
is recommended; the project was tested with Python 3.14.3.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

If activation is blocked, use `.\.venv\Scripts\python.exe` instead of `python`.
On macOS/Linux, use `python3 -m venv .venv` and `source .venv/bin/activate`.

Set your credentials in the local `.env` file:

```dotenv
OPENAI_API_KEY=your_private_key_here
OPENAI_MODEL=gpt-6-luna
```

Existing environment variables take precedence over `.env`. Restart the app after
changing them. Keep API keys and customer records private; `.env` and `logs/` are
excluded from Git. Messages and selected business evidence are sent to OpenAI.
Use fictional contact details in demonstrations.

## Run

```powershell
python app.py
```

Open **http://127.0.0.1:7860** and stop the server with **Ctrl+C**. If the port is
occupied, set `$env:GRADIO_SERVER_PORT="7861"` before starting the app.
A configured API key and model access are needed for chat. API requests incur
usage charges; quota or connection failures appear as an error message.

To use the notebook, open `business_agent.ipynb` in Jupyter or VS Code, select the
`.venv` interpreter and run the cells in order from the repository root. It shares
the implementation in `business_agent.py` and `app.py`. Its final cell launches
Gradio; use `demo.close()` before launching it again. Set `C3_LAUNCH_UI=0` to run
the notebook without starting a server.

## How it works

- **Persona and instructions:** separate blocks define Key's voice and operating rules.
- **Router:** one compact LLM call classifies each message as a general question,
  lead collection or feedback. It uses strict JSON output, a 64-token output cap,
  and up to four recent messages of 400 characters each.
- **Retrieval:** local search selects product records, including model-code aliases
  and accessory singular/plural names. Every answer includes a short business
  overview, product categories and opening hours, with relevant PDF and policy excerpts.
- **Tools:** `record_customer_interest(email, name, message)` and
  `record_feedback(question)` append records to `logs/customer_interest.jsonl`
  and `logs/feedback.jsonl`. See [tool schemas and behavior](docs/TOOLS.md).
- **History:** up to 16 recent messages are retained, capped at 12,000 characters
  total and 1,000 per historical message. New messages are limited to 2,000 characters.

A normal turn makes two API requests: classification and answering. Tool results
require additional answering requests. Answers are capped at 1,024 output tokens,
with at most two tool rounds and no automatic API retries. The app uses
`reasoning_effort="none"` and low verbosity with its default model.
Full transcripts are not saved. Set `AGENT_LOG_DIR` to change the record directory.

## Files

| File | Purpose |
| --- | --- |
| `business_agent.ipynb` | Notebook explaining the agent and launching the chatbot |
| `business_agent.py` | Persona, instructions, router, retrieval, tools and chat loop |
| `app.py`, `assets/` | Gradio interface, stylesheet and avatar |
| `me/about_business.pdf`, `me/business_summary.txt` | Business profile and summary |
| `data/` | Canonical catalogue, accessories, store information and policies |
| `docs/TOOLS.md` | Tool schemas, outputs and logging behavior |
| `demo/` | Functionality video |
