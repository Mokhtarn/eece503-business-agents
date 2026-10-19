"""Separate C4 Gradio UI using C3's visual assets, with per-session chat history."""
import functools
import os
from dataclasses import replace
from pathlib import Path
import gradio as gr
from app import DESCRIPTION, HEAD, ASSETS, AVATAR, _theme, _sidebar_html, SIDEBAR_JS
from c4_config import ModelConfig
from c4_tools import ToolBox
from react_agent import ReactAgent


def respond(message, history, persona, show_trace, agent_factory=None):
    history = list(history or [])
    try:
        agent = agent_factory(persona) if agent_factory else ReactAgent(persona=persona)
        result = agent.run(message, history)
    except Exception:
        result = {"output": "I couldn't complete that request. Check the message (1-2,000 characters), model settings and API connection.",
                  "trace": [], "iterations": 0, "error": "request_failure"}
    updated = history + [{"role": "user", "content": message}, {"role": "assistant", "content": result["output"]}]
    # Keep UI memory bounded as well as the API context. No shared server transcript.
    updated = updated[-16:]
    trace = {"steps": result["iterations"], "events": result["trace"], "error": result["error"]} if show_trace else None
    return "", updated, trace


def create_demo(agent_factory=None):
    tools = ToolBox()
    with gr.Blocks(analytics_enabled=False, title="Keys & Strings | C4 ReAct") as demo:
        gr.HTML(DESCRIPTION)
        gr.Markdown("**C4 - Key & String** - Choose an advisor. Chat uses OpenAI; prices and stock are simulations. Use fictional contact details.")
        with gr.Row():
            persona = gr.Dropdown(choices=[("Key - Friendly Music Advisor", "key"), ("String - Technical Instrument Specialist", "string")], value="key", label="Advisor")
            show_trace = gr.Checkbox(label="Show observable tool trace", value=False)
        chat = gr.Chatbot(show_label=False, elem_id="ks-chat", avatar_images=(None, str(AVATAR)), buttons=["copy"], feedback_options=None)
        message = gr.Textbox(label="Your message", placeholder="Ask about instruments, stock, policies or follow-up...", max_lines=6)
        with gr.Row():
            send = gr.Button("Send", variant="primary")
            clear = gr.Button("Clear conversation")
        with gr.Accordion("Observable actions and observations", open=False):
            gr.Markdown("Tools, arguments and returned evidence only. Contact details can appear here when tracing a lead save.")
            trace = gr.JSON(label="Latest interaction")
        callback = functools.partial(respond, agent_factory=agent_factory)
        for event in (message.submit, send.click):
            event(callback, [message, chat, persona, show_trace], [message, chat, trace], concurrency_limit=1)
        clear.click(lambda: ("", [], None), outputs=[message, chat, trace])
        show_trace.change(lambda: None, outputs=trace)
        with gr.Sidebar(open=True, width=300, elem_id="ks-sidebar"):
            gr.HTML(_sidebar_html(tools.knowledge), js_on_load=SIDEBAR_JS)
    demo.launch = functools.partial(demo.launch, theme=_theme(), css_paths=[ASSETS / "app.css"], head=HEAD, favicon_path=str(AVATAR))
    return demo


if __name__ == "__main__":
    create_demo().launch(server_name="127.0.0.1", server_port=int(os.getenv("C4_GRADIO_PORT", "7861")), share=False)
