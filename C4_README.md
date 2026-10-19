# Keys & Strings: C4

Main submission: [react_agent.ipynb](react_agent.ipynb).

The notebook contains the use case, manual LangGraph implementation, persona
prompts and configurations, operational instructions, experiment design,
measured comparisons and five reflections. Recorded inputs, outputs, tool traces,
usage and failures for the 72 primary cases and eight temperature-comparison cases are in
[docs/C4_RESULTS.json](docs/C4_RESULTS.json).

## Setup

From the repository root, using Python 3.11 or newer:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-c4.txt
```

Keep the API key in the private `.env` file:

```dotenv
OPENAI_API_KEY=your_private_key_here
OPENAI_MODEL=gpt-6-luna
```

Existing environment values take precedence. The project was tested with Python
3.14.3. Select the same Python environment when opening the notebook. Run its
cells from the repository root; they read local data and saved results without
making API requests.

## Launch

```powershell
python react_app.py
```

The app opens at http://127.0.0.1:7861. Select Key or String and optionally show
the latest tool trace. Clear conversation resets short-term memory. Traces can
contain supplied contact details. Use fictional contacts for demonstrations.
The original C3 app remains available with `python app.py`.

## Implementation

- `react_agent.py`: manual decide, validate, act, observe and final nodes with
  conditional edges. Native tool calls are parsed and executed by local dispatch.
- `c4_config.py`, `c4_prompts.py`: validated model settings, two persona prompts,
  shared factual constraints and separate operational instructions.
- `c4_tools.py`, `c4_feedback.py`: six tools using the existing C3 catalogue and
  policies, plus contextual checks before feedback writes.
- `react_app.py`: separate Gradio interface with bounded session memory.
- `c4_scenarios.json`, `c4_experiments.py`, `c4_analyze.py`: scenarios, controlled
  comparisons, seeded execution and offline aggregation.

The default permits eight model decisions per turn and one tool call per
decision. Argument, tool and API errors stop safely. Duplicate calls within a
turn reuse the observation. Only actions, observations and final answers are
logged; hidden reasoning is excluded. C3 code, data and its submission tag are
preserved.

Key uses medium verbosity and String uses low verbosity. Both default to Luna,
reasoning effort `none` and 900 output tokens per request. Optional environment
settings are `C4_REASONING_EFFORT`, `C4_VERBOSITY`, `C4_MAX_OUTPUT_TOKENS`,
`C4_TEMPERATURE` and `C4_TOP_P`. Sampling controls require effort `none`.
Unknown models and unsupported settings are rejected. See the notebook for
capability restrictions and source documentation.

## Experiments

The six configurations compare personas, reasoning guidance, examples and
instruction placement across twelve scenarios. The recorded run completed
70 of 72 cases without controller errors. Two premature lead saves failed
argument validation. All failures remain in the evidence. Estimated API cost
was $0.099704. The baseline completed all twelve scenarios at the lowest cost
among the setups that completed all twelve, so it is the final default.

A separate controlled comparison uses temperatures 0.2 and 0.8 on four scenarios,
holding Key's persona, prompt, model, verbosity and token cap fixed. Its inputs,
outputs and measurements are stored under `configuration_study` in the same
results file and discussed in the notebook. Both temperatures completed all four
scenarios; 0.8 produced shorter answers and cost slightly less. The original
sampling setting was retained because temperature tuning did not improve
completion or reduce model requests.

```powershell
# Print the schedule without API calls.
python c4_experiments.py

# Optional paid reproduction; choose a new output directory.
python c4_experiments.py --live --repeats 1 --seed 503 --budget 0.25 --output tmp/new-study
python c4_analyze.py tmp/new-study --output tmp/new-analysis

# Reproduce only the eight-case configuration comparison.
python c4_experiments.py --live --variants temperature_low temperature_high --scenarios beginner_piano guitar_amp professional multistep --seed 503 --budget 0.05 --output tmp/new-temperature
python c4_analyze.py tmp/new-temperature --output tmp/new-temperature-analysis
```

The budget is a software estimate, not billing enforcement. Experiment logs are
isolated; normal customer records go to the ignored `logs/` directory.
Prices and inventory are simulated. The tools save local records; they do not
send messages, book appointments, reserve stock or process orders.

Development tests, mocks and validation scripts stay local and are not required
to run the submitted notebook or application.
