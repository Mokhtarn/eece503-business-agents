"""Run the local Gradio demonstration with: python app.py."""

import functools
import html
import logging
import os

from dotenv import load_dotenv

from business_agent import BusinessAgent, ROOT

ASSETS = ROOT / "assets"
AVATAR = ASSETS / "key_avatar.svg"

# (card label, message sent when the card is clicked). The first four cover all three
# routes and are the ones shown on phones.
EXAMPLES = [
    ("Beginner pianos under $800", "I'm a beginner. Which digital pianos cost under $800?"),
    ("Book a P-145 demo", "I'd like a Yamaha P-145 demo. Can you record my interest?"),
    ("Hours and returns", "What are your opening hours and return conditions?"),
    ("Shop phone number", "What is your shop's phone number?"),
    ("First electric guitar + amp", "I want my first electric guitar and a small amp. What would you suggest?"),
    ("Leave feedback", "Some feedback: I'd love to see more left-handed guitars in the showroom."),
]

# Markdown/HTML rendered by ChatInterface above the chat. Classes are styled in assets/app.css.
DESCRIPTION = """<div class="ks-hero">
<div class="ks-hero-body">
<p class="ks-eyebrow">Piano &amp; Guitar Shop &middot; Beirut</p>
<h1 class="ks-title">Keys <span>&amp;</span> Strings</h1>
<p class="ks-lede">Chat with <strong>Key</strong>, our shop assistant, about instruments, prices, accessories and store policies.</p>
</div>
<p class="ks-disclaimer">A fictional shop for a class demo: prices and inventory are educational simulations.
Chat messages go to OpenAI. If you want follow-up, Key can save a name and email locally for staff review &mdash; please use fictional contact details.</p>
<div class="ks-keys" aria-hidden="true"></div>
</div>"""

PLACEHOLDER = """<div class="ks-welcome">
<div class="ks-welcome-mark" aria-hidden="true"></div>
<p class="ks-welcome-title">Hi, I'm Key.</p>
<p>Ask me to compare instruments, check a price, or save a follow-up request for our team. Try one of these:</p>
</div>"""

HEAD = """<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&display=swap">"""

# Runs in the browser so "open now" uses Beirut time and stays current without server calls.
SIDEBAR_JS = """
const fmt = new Intl.DateTimeFormat('en-US', {timeZone: 'Asia/Beirut', weekday: 'long',
                                              hour: '2-digit', minute: '2-digit', hourCycle: 'h23'});
const order = ['sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday'];
const minutes = (hhmm) => { const [h, m] = hhmm.split(':').map(Number); return h * 60 + m; };
function tick() {
  const parts = Object.fromEntries(fmt.formatToParts(new Date()).map(p => [p.type, p.value]));
  const today = parts.weekday.toLowerCase();
  const now = Number(parts.hour) * 60 + Number(parts.minute);
  const rows = [...element.querySelectorAll('[data-day]')];
  rows.forEach(row => row.classList.toggle('is-today', row.dataset.day === today));
  const span = (day) => { const row = rows.find(r => r.dataset.day === day); return row && row.dataset.hours.includes('-') ? row.dataset.hours.split('-') : null; };
  const status = element.querySelector('.ks-status');
  const show = (state, detail) => {
    status.className = `ks-status is-${state.toLowerCase()}`;
    status.innerHTML = `<strong>${state} now</strong><span>${detail}</span>`;
  };
  const hours = span(today);
  if (hours && now >= minutes(hours[0]) && now < minutes(hours[1])) {
    return show('Open', `Until ${hours[1]} today`);
  }
  for (let step = 0; step < 7; step++) {
    const day = order[(order.indexOf(today) + step) % 7];
    const next = span(day);
    if (next && (step > 0 || now < minutes(next[0]))) {
      const when = step === 0 ? 'today' : step === 1 ? 'tomorrow' : 'on ' + day[0].toUpperCase() + day.slice(1);
      return show('Closed', `Opens ${when} at ${next[0]}`);
    }
  }
  show('Closed', 'Check back soon');
}
tick();
setInterval(tick, 60000);
// The sidebar starts open for desktops; on phones it would cover the chat.
if (window.matchMedia('(max-width: 767px)').matches) {
  document.querySelector('#ks-sidebar.open .toggle-button')?.click();
}
"""


def _theme():
    import gradio as gr

    brass = gr.themes.Color(
        c50="#fbf6ec", c100="#f5e9d0", c200="#ebd3a3", c300="#dfb972", c400="#d2a14f",
        c500="#b8873b", c600="#9a6d2e", c700="#7b5526", c800="#5e4121", c900="#45301a",
        c950="#2a1d10", name="brass",
    )
    # Ivory, ebony and brass: a piano's keys and hardware. Finer details live in assets/app.css.
    return gr.themes.Base(
        primary_hue=brass, secondary_hue=brass, neutral_hue=gr.themes.colors.stone,
        radius_size=gr.themes.sizes.radius_lg,
        font=(gr.themes.GoogleFont("DM Sans"), "ui-sans-serif", "system-ui", "sans-serif"),
    ).set(
        body_background_fill="#f6f1e7", body_background_fill_dark="#12100e",
        body_text_color="#231d18", body_text_color_dark="#efe7da",
        body_text_color_subdued="#7a6d5f", body_text_color_subdued_dark="#a39582",
        background_fill_primary="#fffdf8", background_fill_primary_dark="#1b1714",
        background_fill_secondary="#f6efe2", background_fill_secondary_dark="#231e1a",
        block_background_fill="#fffdf8", block_background_fill_dark="#1b1714",
        border_color_primary="#e6dbc8", border_color_primary_dark="#352d26",
        block_border_color="#e6dbc8", block_border_color_dark="#352d26",
        input_background_fill="#fffdf8", input_background_fill_dark="#1b1714",
        color_accent="#b8873b", color_accent_soft="#f5e9d0", color_accent_soft_dark="#3a2c1a",
        link_text_color="#9a6d2e", link_text_color_dark="#dfb972",
        button_primary_background_fill="#1c1714", button_primary_background_fill_dark="#d2a14f",
        button_primary_background_fill_hover="#3a2f27", button_primary_background_fill_hover_dark="#dfb972",
        button_primary_text_color="#fbf7ef", button_primary_text_color_dark="#1c1714",
    )


def _sidebar_html(knowledge):
    """Static shop facts from the same JSON files the agent answers from."""
    store, products = knowledge.store, knowledge.products.values()
    esc = html.escape
    counts = {category: sum(p["category"] == category for p in products)
              for category in ("Piano", "Guitar", "Amplifier")}
    in_stock = sum(p["stock_quantity"] > 0 for p in products)
    hours = "".join(
        f'<li data-day="{day}" data-hours="{esc(value)}"><span>{day[:3].title()}</span><span>{esc(value)}</span></li>'
        for day, value in store["opening_hours"].items()
    )
    team = "".join(
        f'<li><span class="ks-initials">{esc("".join(word[0] for word in person["name"].split()[:2]))}</span>'
        f'<span><strong>{esc(person["name"])}</strong><small>{esc(person["role"])}</small></span></li>'
        for person in store["team"]
    )
    return f"""<div class="ks-side">
<section>
<p class="ks-side-label">Showroom hours <small>Beirut time</small></p>
<p class="ks-status"><strong>Mon&ndash;Sat</strong><span>10:00&ndash;18:00</span></p>
<ul class="ks-hours">{hours}</ul>
</section>
<section>
<p class="ks-side-label">In the catalogue</p>
<div class="ks-stats">
<div><strong>{counts["Piano"]}</strong><span>pianos</span></div>
<div><strong>{counts["Guitar"]}</strong><span>guitars</span></div>
<div><strong>{counts["Amplifier"]}</strong><span>amps</span></div>
</div>
<p class="ks-fine">{in_stock} models in simulated stock, others by special order. Catalogue as of {esc(knowledge.catalogue["as_of"])}.</p>
</section>
<section>
<p class="ks-side-label">What Key can do</p>
<ul class="ks-abilities">
<li><span class="ks-dot" data-route="general"></span><span><strong>Answer questions</strong><small>From our catalogue, policies and shop profile.</small></span></li>
<li><span class="ks-dot" data-route="lead"></span><span><strong>Save a follow-up request</strong><small>Name, email and interest, saved locally for staff. No emails are sent.</small></span></li>
<li><span class="ks-dot" data-route="feedback"></span><span><strong>Log feedback</strong><small>Plus any question it can't answer, so the team can fill the gap.</small></span></li>
</ul>
</section>
<section>
<p class="ks-side-label">The team</p>
<ul class="ks-team">{team}</ul>
</section>
</div>"""


def create_demo(agent=None):
    """Construct the UI without launching a server or requiring an API key."""
    import gradio as gr

    load_dotenv(ROOT / ".env", override=False)
    assistant = agent if agent is not None else BusinessAgent()
    demo = gr.ChatInterface(
        fn=assistant.chat,
        description=DESCRIPTION,
        chatbot=gr.Chatbot(
            show_label=False, scale=1, layout="bubble", elem_id="ks-chat",
            avatar_images=(None, str(AVATAR)), placeholder=PLACEHOLDER,
            buttons=["copy"], feedback_options=None,
        ),
        textbox=gr.Textbox(
            placeholder="Ask about pianos, guitars, prices or store policies...",
            show_label=False, scale=7, autofocus=True, submit_btn=True, max_lines=6,
            elem_id="ks-input",
        ),
        examples=[prompt for _, prompt in EXAMPLES],
        example_labels=[label for label, _ in EXAMPLES],
        cache_examples=False,
        save_history=False,
        analytics_enabled=False,
    )
    # No title argument, so ChatInterface skips its default <h1>; the hero above replaces it.
    demo.title = "Keys & Strings | Shop Assistant"
    with demo:
        with gr.Sidebar(open=True, width=300, elem_id="ks-sidebar"):
            gr.HTML(_sidebar_html(assistant.knowledge), js_on_load=SIDEBAR_JS)
    # Apply the same styling when launched from this script or the notebook.
    demo.launch = functools.partial(
        demo.launch, theme=_theme(), css_paths=[ASSETS / "app.css"], head=HEAD,
        favicon_path=str(AVATAR),
    )
    return demo


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # Local demo only; no public share link is created.
    create_demo().launch(server_name="127.0.0.1", server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")), share=False)
