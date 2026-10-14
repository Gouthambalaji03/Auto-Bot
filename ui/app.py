"""
ui/app.py — Gradio Application Builder with Aligned Modern Layout & Recent Chats Sidebar

Layout Specifications Replicated from Architecture:
1. Top Sticky Header with Brand Title, Live Engine Badge, and User Controls.
2. Left Aligned Sidebar featuring 'Recent Chats' history and quick capabilities.
3. Central Chat Canvas with rich markdown responses, EMI cards, and feedback controls.
4. Bottom Floating Capsule Input Bar with language indicator and disclaimer note.
5. Interactive Action Chips below input area for one-click prompts.
"""

import os
import re
import uuid
from collections.abc import AsyncIterator
from typing import Optional, Tuple

import gradio as gr

from agents.automotive_agent import stream_chat_with_autobot
from db.auth import db_create_user, db_authenticate_user
from db.queries import (
    db_create_conversation,
    db_save_chat_message,
    db_get_user_conversations,
    db_get_conversation_messages,
    db_delete_conversation,
)


# ─────────────────────────────────────────────
# Helper Functions: Car Recommendation Card Formatter
# ─────────────────────────────────────────────

def format_car_recommendation_cards(text: str) -> str:
    """Converts structured vehicle recommendation sections into modern HTML car cards."""
    if not text or "<div class=\"car-card\"" in text or "<div class='car-card'" in text:
        return text

    pattern = re.compile(
        r"###\s+(?:(?:\d+\.|\*|\-)\s+)?([^\n·\(\)]+?)(?:\s+[·\-–]\s+([^\n\(\)]+?))?(?:\s*\(([^\n\)]+?)\))?\n\n?"
        r"((?:[\*\-]\s+[^\n]+\n?)+)"
        r"(?:\n*>+\s*(?:💡\s*)?\*\*Best For:\*\*\s*([^\n]+))?",
        re.MULTILINE
    )

    def replace_with_card(match):
        name = match.group(1).strip()
        segment = (match.group(2) or "").strip()
        price_header = (match.group(3) or "").strip()
        specs_raw = match.group(4).strip()
        best_for = (match.group(5) or "").strip()

        spec_items = []
        features = []
        price = price_header

        for line in specs_raw.split("\n"):
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"^[\*\-]\s*", "", line)

            if ("Price:" in line or "💰" in line) and not price:
                p_match = re.search(r"Price:\*\*\s*(.+)$", line, re.IGNORECASE)
                if p_match:
                    price = p_match.group(1).strip()
                continue
            elif "Price:" in line or "💰" in line:
                continue

            if "Key Features:" in line or "Features:" in line or "🌟" in line:
                f_match = re.search(r"Features:\*\*\s*(.+)$", line, re.IGNORECASE)
                if f_match:
                    raw_feats = f_match.group(1).split(",")
                    for f in raw_feats:
                        f_clean = f.strip()
                        if f_clean:
                            features.append(f_clean)
                continue

            spec_match = re.search(r"(?:([^\*:]+?)\s*)?\*\*([^\*:]+?):\*\*\s*(.+)$", line)
            if spec_match:
                icon_or_prefix = (spec_match.group(1) or "").strip()
                label = spec_match.group(2).strip()
                val = spec_match.group(3).strip()
                full_label = f"{icon_or_prefix} {label}".strip() if icon_or_prefix else label
                spec_items.append((full_label, val))
            else:
                spec_items.append(("", line))

        card_html = ['\n<div class="car-card">']
        card_html.append('  <div class="car-card-header">')
        card_html.append('    <div class="car-title-group">')
        card_html.append(f'      <h3 class="car-name">{name}</h3>')
        if segment:
            card_html.append(f'      <span class="car-segment-badge">{segment}</span>')
        card_html.append('    </div>')
        if price:
            card_html.append(f'    <div class="car-price-badge">{price}</div>')
        card_html.append('  </div>')

        if spec_items:
            card_html.append('  <div class="car-specs-grid">')
            for lbl, val in spec_items:
                card_html.append('    <div class="spec-card">')
                if lbl:
                    card_html.append(f'      <span class="spec-label">{lbl}</span>')
                card_html.append(f'      <span class="spec-value">{val}</span>')
                card_html.append('    </div>')
            card_html.append('  </div>')

        if features:
            card_html.append('  <div class="car-features-section">')
            card_html.append('    <span class="features-title">🌟 Key Features</span>')
            card_html.append('    <div class="feature-pills">')
            for f in features:
                card_html.append(f'      <span class="feature-pill">{f}</span>')
            card_html.append('    </div>')
            card_html.append('  </div>')

        if best_for:
            card_html.append(f'  <div class="car-best-for">💡 <strong>Best For:</strong> {best_for}</div>')

        card_html.append('</div>\n')
        return "\n".join(card_html)

    return pattern.sub(replace_with_card, text)


# ─────────────────────────────────────────────
# Helper Functions: Auth & History UI Handlers
# ─────────────────────────────────────────────

def get_history_choices(user_state: Optional[dict]) -> list[tuple[str, str]]:
    """Fetches past conversations for logged-in user and returns choices for Dropdown/Radio."""
    if not user_state or "id" not in user_state:
        return []
    try:
        convs = db_get_user_conversations(user_state["id"])
        choices = []
        for c in convs:
            title = c.get("title", "Untitled Chat")
            count = c.get("message_count", 0)
            label = f"💬 {title} ({count})"
            choices.append((label, c["session_id"]))
        return choices
    except Exception as e:
        print(f"Error fetching history choices: {e}")
        return []


def on_login(username_or_email: str, password: str):
    """Handles User Login and navigates to Main Chat Application."""
    success, msg, user_dict = db_authenticate_user(username_or_email, password)
    if success and user_dict:
        choices = get_history_choices(user_dict)
        radio_update = gr.Radio(choices=choices, value=None, interactive=True)
        status_html = f"<div style='color:#065f46;font-weight:600;font-size:13px;padding:9px 14px;background:#ecfdf5;border:1px solid #a7f3d0;border-radius:10px;margin-top:10px;'>✅ {msg}</div>"
        user_badge = f"<div class='user-pill'><span class='user-pill-dot is-online'>●</span> <strong>{user_dict['username']}</strong></div>"
        return (
            user_dict,                                          # user_state
            status_html,                                        # auth_status
            gr.Column(visible=False),                           # auth_view (hide)
            gr.Column(visible=True),                            # main_view (show)
            radio_update,                                       # history_radio
            user_badge,                                         # user_badge_md
        )
    else:
        status_html = f"<div style='color:#b91c1c;font-weight:600;font-size:13px;padding:9px 14px;background:#fef2f2;border:1px solid #fecaca;border-radius:10px;margin-top:10px;'>❌ {msg}</div>"
        return (
            None,                                               # user_state
            status_html,                                        # auth_status
            gr.Column(visible=True),                            # auth_view (keep visible)
            gr.Column(visible=False),                           # main_view (keep hidden)
            gr.Radio(choices=[], value=None),                   # history_radio
            "<div class='user-pill'><span class='user-pill-dot'>●</span> <strong>Guest User</strong></div>",
        )


def on_signup(username: str, email: str, password: str):
    """Handles User Sign Up and navigates to Main Chat Application."""
    success, msg, user_dict = db_create_user(username, email, password)
    if success and user_dict:
        choices = get_history_choices(user_dict)
        radio_update = gr.Radio(choices=choices, value=None, interactive=True)
        status_html = f"<div style='color:#065f46;font-weight:600;font-size:13px;padding:9px 14px;background:#ecfdf5;border:1px solid #a7f3d0;border-radius:10px;margin-top:10px;'>✅ {msg}</div>"
        user_badge = f"<div class='user-pill'><span class='user-pill-dot is-online'>●</span> <strong>{user_dict['username']}</strong></div>"
        return (
            user_dict,
            status_html,
            gr.Column(visible=False),                           # auth_view (hide)
            gr.Column(visible=True),                            # main_view (show)
            radio_update,
            user_badge,
        )
    else:
        status_html = f"<div style='color:#b91c1c;font-weight:600;font-size:13px;padding:9px 14px;background:#fef2f2;border:1px solid #fecaca;border-radius:10px;margin-top:10px;'>❌ {msg}</div>"
        return (
            None,
            status_html,
            gr.Column(visible=True),
            gr.Column(visible=False),
            gr.Radio(choices=[], value=None),
            "<div class='user-pill'><span class='user-pill-dot'>●</span> <strong>Guest User</strong></div>",
        )


def on_guest():
    """Allows exploring app as Guest User."""
    return (
        None,                                                   # user_state
        "",                                                     # auth_status
        gr.Column(visible=False),                               # auth_view (hide)
        gr.Column(visible=True),                                # main_view (show)
        gr.Radio(choices=[], value=None, interactive=False),    # history_radio
        "<div class='user-pill'><span class='user-pill-dot'>●</span> <strong>Guest User</strong></div>",
    )


def on_logout():
    """Handles Sign Out and returns to Auth Landing View."""
    return (
        None,                                                   # user_state = None
        None,                                                   # active_session_id = None
        [],                                                     # chatbot = []
        "<div style='color:#475569;font-size:13px;padding:9px 14px;background:#f1f5f9;border:1px solid #e2e8f0;border-radius:10px;margin-top:10px;'>Signed out successfully.</div>", # auth_status
        gr.Column(visible=True),                                # auth_view (show)
        gr.Column(visible=False),                               # main_view (hide)
        gr.Radio(choices=[], value=None, interactive=False),    # history_radio
        "<div class='user-pill'><span class='user-pill-dot'>●</span> <strong>Guest User</strong></div>",
        "", "", "", "", ""                                      # clear inputs
    )


def show_login_form():
    """Switch the authentication form without relying on Gradio Tabs internals."""
    return (
        gr.Column(visible=True),
        gr.Column(visible=False),
        gr.Button(elem_classes=["auth-tab", "is-active"]),
        gr.Button(elem_classes=["auth-tab"]),
    )
def show_signup_form():
    """Switch the authentication form without relying on Gradio Tabs internals."""
    return (
        gr.Column(visible=False),
        gr.Column(visible=True),
        gr.Button(elem_classes=["auth-tab"]),
        gr.Button(elem_classes=["auth-tab", "is-active"]),
    )


def on_new_chat(user_state: Optional[dict]) -> Tuple[None, list, str]:
    """Clears chatbot UI and starts a new conversation session."""
    return None, [], ""


def on_select_history(session_id: str, user_state: Optional[dict]) -> Tuple[str, list, str]:
    """Loads selected past conversation from database into chatbot UI."""
    if not session_id or not user_state or "id" not in user_state:
        return "", [], ""
    try:
        raw_msgs = db_get_conversation_messages(session_id, user_state["id"])
        formatted_history = []
        for m in raw_msgs:
            content = m["content"]
            if m["role"] == "assistant":
                content = format_car_recommendation_cards(content)
                content = content.replace("Gemini 2.5 Flash", "Gemini 3.6 Flash")
            formatted_history.append({"role": m["role"], "content": content})
        return session_id, formatted_history, ""
    except Exception as e:
        return "", [], f"<div class='history-status-alert is-error'>Error loading chat: {e}</div>"


def on_delete_history(session_id: str, user_state: Optional[dict]) -> Tuple[None, list, gr.Radio, str]:
    """Deletes selected conversation from database and updates history UI."""
    if not session_id or not user_state or "id" not in user_state:
        return None, [], gr.Radio(choices=[], value=None), "<div class='history-status-alert is-warning'>Select a chat above to delete</div>"
    try:
        db_delete_conversation(session_id, user_state["id"])
        choices = get_history_choices(user_state)
        return None, [], gr.Radio(choices=choices, value=None), "<div class='history-status-alert is-success'>Chat deleted successfully</div>"
    except Exception as e:
        return session_id, [], gr.Radio(choices=[], value=None), f"<div class='history-status-alert is-error'>Error deleting: {e}</div>"


# ─────────────────────────────────────────────
# Async Main Chat Handler
# ─────────────────────────────────────────────

async def chat(
    user_message: str,
    history: list,
    user_state: Optional[dict] = None,
    active_session_id: Optional[str] = None,
) -> AsyncIterator[Tuple[list, str, Optional[str], gr.Radio]]:
    """Stream natural-language Markdown into Gradio and persist only completed turns."""
    if not user_message.strip():
        choices = get_history_choices(user_state)
        yield history, "", active_session_id, gr.Radio(choices=choices, value=active_session_id)
        return

    session_id = active_session_id
    if not session_id:
        session_id = str(uuid.uuid4())

    try:
        if user_state and "id" in user_state:
            title = user_message[:40] + ("..." if len(user_message) > 40 else "")
            db_create_conversation(user_state["id"], session_id, title)
            db_save_chat_message(user_state["id"], session_id, "user", user_message)

        stream_history = list(history)
        stream_history.append({"role": "user", "content": user_message})
        stream_history.append({"role": "assistant", "content": ""})
        choices = get_history_choices(user_state)
        selected_val = session_id if any(c[1] == session_id for c in choices) else None
        radio = gr.Radio(choices=choices, value=selected_val)
        output_markdown = ""
        intent_labels: tuple[str, ...] = ()
        elapsed = 0.0
        async for update in stream_chat_with_autobot(
            user_message,
            history,
            user_id=user_state.get("id") if user_state else None,
            session_id=session_id,
        ):
            output_markdown = update.content
            if update.complete:
                intent_labels = update.intents
                elapsed = update.elapsed_seconds
            else:
                stream_history[-1] = {"role": "assistant", "content": output_markdown}
                yield stream_history, "", session_id, radio

        # Convert car recommendations into modern card view
        output_markdown = format_car_recommendation_cards(output_markdown)

        # Dynamic model display badge
        model_name = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
        model_display = f"⚡ {model_name.replace('-', ' ').title()} Automotive Agent"
        formatted = output_markdown + f"\n\n---\n`{model_display}` &nbsp;·&nbsp; `⏱️ {elapsed}s`"
        stream_history[-1] = {"role": "assistant", "content": formatted}

        if user_state and "id" in user_state and session_id:
            db_intent = ",".join(intent_labels) if intent_labels else "unclassified"
            db_save_chat_message(user_state["id"], session_id, "assistant", formatted, intent=db_intent)

        choices = get_history_choices(user_state)
        selected_val = session_id if any(c[1] == session_id for c in choices) else None
        yield stream_history, "", session_id, gr.Radio(choices=choices, value=selected_val)

    except Exception as e:
        print(f"[UI] [ERROR] Chat request failed: {type(e).__name__}: {e}", flush=True)
        error_msg = f"**Error:** {str(e)}\n\nPlease check your API key and try again."
        if "stream_history" in locals():
            stream_history[-1] = {"role": "assistant", "content": error_msg}
        else:
            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": error_msg})
            stream_history = history
        choices = get_history_choices(user_state)
        yield stream_history, "", active_session_id, gr.Radio(choices=choices)


# ─────────────────────────────────────────────
# Modern Aligned Dark Theme CSS
# ─────────────────────────────────────────────

PREMIUM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body, .gradio-container {
    font-family: 'Inter', -apple-system, sans-serif !important;
    background: #f8fafc !important;
    color: #0f172a !important;
    min-height: 100vh;
}

.gradio-container { padding: 0 !important; max-width: 100% !important; }
footer { display: none !important; }

/* ── Authentication: scoped, content-sized two-panel layout ── */
#auth-view {
    min-height: 100vh;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: clamp(24px, 5vh, 48px) 20px !important;
    background: #f8fafc !important;
    color: #0f172a !important;
}

#auth-card {
    width: min(980px, 100%) !important;
    margin: 0 auto !important;
    padding: 0 !important;
    overflow: hidden !important;
    background: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 22px !important;
    box-shadow: 0 18px 44px rgba(15, 23, 42, 0.08) !important;
}

#auth-card > .wrap { padding: 0 !important; }

.auth-layout {
    display: grid !important;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) !important;
    align-items: stretch !important;
    width: 100% !important;
    gap: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}

.auth-layout > .auth-form-pane,
.auth-layout > .auth-promo-pane {
    width: 100% !important;
    max-width: 100% !important;
    min-width: 0 !important;
}

#auth-card .auth-form-pane {
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
    align-items: center !important;
    padding: 42px 44px !important;
    background: #ffffff !important;
    width: 100% !important;
}
#auth-card .auth-form-pane > .wrap,
#auth-card .auth-form-inner > .wrap {
    width: 100% !important;
    min-height: 0 !important;
}
#auth-card .auth-form-inner {
    width: 100% !important;
    max-width: 420px !important;
    display: flex !important;
    flex-direction: column !important;
    flex: 0 0 auto !important;
}

#auth-card .auth-form-heading { margin: 0 0 26px !important; text-align: left; }
#auth-card .auth-mark {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 38px;
    height: 38px;
    margin-bottom: 12px;
    border-radius: 10px;
    background: rgba(37, 99, 235, 0.08);
    color: #2563eb;
    font-size: 20px;
    border: 1px solid rgba(37, 99, 235, 0.15);
}
#auth-card .auth-brand {
    font-family: 'Inter', -apple-system, sans-serif;
    font-size: 26px;
    font-weight: 700;
    letter-spacing: -0.5px;
    line-height: 1.2;
    color: #0f172a;
    margin-bottom: 5px;
}
#auth-card .auth-form-heading p { color: #64748b; font-size: 14px; line-height: 1.5; }

/* Explicit switch buttons avoid Gradio Tabs' generated tab-nav sizing rules. */
#auth-card .auth-tab-switch {
    display: flex !important;
    flex-direction: row !important;
    gap: 12px !important;
    width: 100% !important;
    margin: 0 0 20px !important;
}
#auth-card .auth-tab-switch > * {
    flex: 1 1 0% !important;
    width: 50% !important;
    min-width: 0 !important;
}
#auth-card button.auth-tab {
    width: 100% !important;
    min-height: 42px !important;
    height: 42px !important;
    margin: 0 !important;
    padding: 0 14px !important;
    background: #f8fafc !important;
    border: 1px solid #dbe3ee !important;
    border-radius: 10px !important;
    color: #64748b !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    box-shadow: none !important;
}
#auth-card button.auth-tab.is-active {
    background: #eff6ff !important;
    border-color: #2563eb !important;
    color: #1d4ed8 !important;
    font-weight: 600 !important;
}

/* Form fields, status, and guest action all share auth-form-inner's width. */
#auth-card .auth-panel { width: 100% !important; min-height: 0 !important; }
#auth-card .auth-panel > .wrap { padding: 0 !important; min-height: 0 !important; }
#auth-card .auth-field-label {
    display: block;
    margin: 0 0 6px;
    color: #334155;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.01em;
}

#auth-card .auth-input {
    display: block !important;
    margin: 0 0 14px !important;
    padding: 0 !important;
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    width: 100% !important;
}
#auth-card .auth-input:last-of-type { margin-bottom: 0 !important; }

#auth-card .auth-input .input-container {
    display: flex !important;
    align-items: center !important;
    width: 100% !important;
    min-height: 46px !important;
    margin-top: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 10px !important;
    box-shadow: none !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
}
#auth-card .auth-input .input-container:focus-within {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15) !important;
}
#auth-card .auth-input input {
    display: block !important;
    width: 100% !important;
    height: 44px !important;
    min-height: 44px !important;
    margin: 0 !important;
    padding: 0 14px !important;
    background: transparent !important;
    border: 0 !important;
    border-radius: 0 !important;
    color: #0f172a !important;
    font-size: 14px !important;
    line-height: 44px !important;
    box-shadow: none !important;
}
#auth-card .auth-input input::placeholder { color: #94a3b8 !important; }
#auth-card .auth-input input:focus { outline: 0 !important; box-shadow: none !important; }

#auth-card .gr-textbox,
#auth-card [data-testid="textbox"] {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
}

/* Primary CTA Button */
#auth-card .auth-primary,
#auth-card button.auth-primary {
    width: 100% !important;
    min-height: 46px !important;
    height: 46px !important;
    margin-top: 20px !important;
    border: none !important;
    border-radius: 10px !important;
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    color: #ffffff !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 6px rgba(37, 99, 235, 0.25) !important;
    transition: all 0.15s ease !important;
}
#auth-card .auth-primary:hover {
    background: linear-gradient(135deg, #1d4ed8, #1e40af) !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35) !important;
    transform: translateY(-1px) !important;
}

/* Status Notification Component (Hidden when empty) */
#auth-card #auth-status { display: none; min-height: 0 !important; margin: 0 !important; padding: 0 !important; }
#auth-card #auth-status:not(:empty) { display: block !important; }
#auth-card #auth-status > div { margin: 12px 0 0 !important; padding: 10px 14px !important; border-radius: 8px !important; font-size: 13px !important; font-weight: 500 !important; }

/* Guest Section (Secondary CTA inside Form Pane) */
#auth-card .guest-divider {
    margin: 24px 0 10px;
    padding-top: 16px;
    border-top: 1px solid #e2e8f0;
    text-align: center;
    width: 100%;
}
#auth-card .guest-divider span { color: #64748b !important; font-size: 12px; }
#auth-card .guest-button,
#auth-card button.guest-button {
    width: 100% !important;
    margin: 0 !important;
    min-height: 46px !important;
    height: 46px !important;
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 10px !important;
    color: #334155 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    box-shadow: none !important;
    transition: background 0.15s ease, border-color 0.15s ease !important;
}
.guest-button:hover { background: #f8fafc !important; border-color: #94a3b8 !important; }

/* The promo pane is a grid sibling that naturally matches the form pane's height. */
#auth-card .auth-promo-pane {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    overflow: hidden !important;
    padding: 44px !important;
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%) !important;
    box-sizing: border-box !important;
}
#auth-card .auth-promo-pane > .wrap { width: 100% !important; }
#auth-card .auth-promo-copy {
    max-width: 320px;
    text-align: center;
}
#auth-card .auth-promo-copy .auth-mark {
    margin-bottom: 20px;
    background: rgba(255, 255, 255, 0.15) !important;
    color: #60a5fa !important;
    box-shadow: none;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
}
#auth-card .auth-promo-copy h2 {
    color: #ffffff !important;
    font-size: 32px !important;
    font-weight: 600 !important;
    letter-spacing: -0.5px !important;
    line-height: 1.2 !important;
    margin-bottom: 12px !important;
}
#auth-card .auth-promo-copy p {
    margin-top: 0 !important;
    color: #94a3b8 !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
}

@media (max-width: 768px) {
    #auth-view { padding: 24px 16px !important; }
    #auth-card { border-radius: 16px !important; width: 100% !important; }
    .auth-layout { grid-template-columns: minmax(0, 1fr) !important; }
    .auth-layout > .auth-form-pane, .auth-layout > .auth-promo-pane { width: 100% !important; }
    #auth-card .auth-form-pane { padding: 32px 20px !important; }
    #auth-card .auth-promo-pane { padding: 32px 20px !important; }
    #auth-card .auth-promo-copy { max-width: 360px; }
    #auth-card .auth-promo-copy .auth-mark { display: none !important; }
    #auth-card .auth-promo-copy h2 { font-size: 24px !important; }
}

/* ── Main Chat App Shell: Modern Slate & Automotive Cobalt Theme ── */
html, body, #root, .gradio-container {
    width: 100% !important;
    min-width: 0 !important;
    height: 100vh !important;
    max-height: 100vh !important;
    overflow: hidden !important;
    margin: 0 !important;
    padding: 0 !important;
    background: #f8fafc !important;
    color: #0f172a !important;
}

.gradio-container, .gradio-container > .main, #main-view > .wrap {
    width: 100% !important;
    max-width: none !important;
    height: 100vh !important;
    max-height: 100vh !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
    background: #f8fafc !important;
}

#main-view {
    --color-accent: #2563eb;
    --color-accent-soft: rgba(37, 99, 235, 0.1);
    height: 100vh !important;
    max-height: 100vh !important;
    overflow: hidden !important;
    background: #f8fafc !important;
    color: #0f172a !important;
    display: flex !important;
    flex-direction: column !important;
}

#main-view > .wrap {
    padding: 0 !important;
    display: flex !important;
    flex-direction: column !important;
    height: 100vh !important;
    max-height: 100vh !important;
    overflow: hidden !important;
}

/* Header Navigation Bar */
.top-navbar {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    height: 56px !important;
    min-height: 56px !important;
    max-height: 56px !important;
    padding: 0 24px !important;
    background: #ffffff !important;
    border-bottom: 1px solid #e2e8f0 !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03) !important;
    flex-shrink: 0 !important;
    z-index: 10 !important;
}

.top-navbar > .wrap {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    width: 100% !important;
    padding: 0 !important;
}

.header-user-controls {
    display: flex !important;
    align-items: center !important;
    gap: 12px !important;
    margin: 0 !important;
}

.user-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 500;
    color: #334155;
}
.user-pill-dot {
    color: #94a3b8;
    font-size: 10px;
}
.user-pill-dot.is-online {
    color: #10b981;
}
.btn-logout,
button.btn-logout {
    min-height: 36px !important;
    max-height: 36px !important;
    height: 36px !important;
    padding: 0 16px !important;
    font-size: 12.5px !important;
    font-weight: 600 !important;
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    border: none !important;
    color: #ffffff !important;
    border-radius: 10px !important;
    cursor: pointer !important;
    box-shadow: 0 2px 6px rgba(37, 99, 235, 0.25) !important;
    transition: all 0.15s ease !important;
}
.btn-logout:hover {
    background: linear-gradient(135deg, #1d4ed8, #1e40af) !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35) !important;
    transform: translateY(-1px) !important;
}

/* App Workspace (Split View) */
.app-workspace {
    flex: 1 1 0% !important;
    height: calc(100vh - 56px) !important;
    max-height: calc(100vh - 56px) !important;
    gap: 0 !important;
    background: #f8fafc !important;
    display: flex !important;
    overflow: hidden !important;
    min-height: 0 !important;
}

.app-workspace > .wrap {
    display: flex !important;
    width: 100% !important;
    height: 100% !important;
    padding: 0 !important;
    overflow: hidden !important;
}

/* Sidebar Panel */
.sidebar-panel {
    width: 280px !important;
    min-width: 280px !important;
    max-width: 280px !important;
    height: 100% !important;
    max-height: 100% !important;
    min-height: 0 !important;
    padding: 18px 16px !important;
    background: #ffffff !important;
    border: 0 !important;
    border-right: 1px solid #e2e8f0 !important;
    border-radius: 0 !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: space-between !important;
    overflow-y: auto !important;
    flex-shrink: 0 !important;
}

.sidebar-top-group {
    display: flex !important;
    flex-direction: column !important;
    gap: 8px !important;
    flex-grow: 1 !important;
    min-height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}

.sidebar-bottom-group {
    margin-top: auto !important;
    padding-top: 14px !important;
    border-top: 1px solid #f1f5f9 !important;
    flex-grow: 0 !important;
    flex-shrink: 0 !important;
}

/* New Chat Button */
.btn-new-chat,
button.btn-new-chat {
    min-height: 40px !important;
    max-height: 40px !important;
    height: 40px !important;
    padding: 0 16px !important;
    font-size: 13.5px !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    box-shadow: 0 1px 3px rgba(37, 99, 235, 0.2) !important;
    text-align: center !important;
    background: #2563eb !important;
    border: none !important;
    color: #ffffff !important;
    margin-bottom: 8px !important;
    width: 100% !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
}

.btn-new-chat:hover {
    background: #1d4ed8 !important;
    box-shadow: 0 4px 10px rgba(37, 99, 235, 0.3) !important;
    transform: translateY(-1px) !important;
}

.sidebar-title {
    font-size: 11.5px !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    color: #94a3b8 !important;
    margin: 8px 0 6px 2px !important;
}

/* History List */
.history-select {
    background: transparent !important;
    border: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    max-height: 180px !important;
    overflow-y: auto !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 4px !important;
}

.history-select fieldset {
    border: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 4px !important;
}

.history-select input[type="radio"] {
    display: none !important;
    appearance: none !important;
    opacity: 0 !important;
    width: 0 !important;
    height: 0 !important;
}

.history-select label {
    display: flex !important;
    align-items: center !important;
    width: 100% !important;
    min-height: 34px !important;
    max-height: 34px !important;
    padding: 6px 10px !important;
    border-radius: 8px !important;
    font-size: 12.5px !important;
    font-weight: 500 !important;
    color: #334155 !important;
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    cursor: pointer !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    transition: all 0.15s ease !important;
    margin: 0 !important;
}

.history-select label span {
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    width: 100% !important;
    display: block !important;
}

.history-select label:hover {
    background: #eff6ff !important;
    border-color: #bfdbfe !important;
    color: #1d4ed8 !important;
}

.history-select label.selected,
.history-select label:has(input:checked) {
    background: #eff6ff !important;
    border-color: #2563eb !important;
    color: #1d4ed8 !important;
    font-weight: 600 !important;
}

.empty-history-box {
    padding: 10px 8px;
    font-size: 11.5px;
    color: #94a3b8;
    text-align: center;
    border: 1px dashed #e2e8f0;
    border-radius: 8px;
    background: #f8fafc;
    margin: 2px 0 6px 0;
}

/* Clean Delete Button */
.btn-delete-chat,
button.btn-delete-chat {
    min-height: 30px !important;
    max-height: 30px !important;
    height: 30px !important;
    padding: 0 10px !important;
    font-size: 11.5px !important;
    font-weight: 500 !important;
    background: transparent !important;
    border: 1px dashed #cbd5e1 !important;
    color: #64748b !important;
    border-radius: 6px !important;
    margin-top: 2px !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
}
.btn-delete-chat:hover {
    background: #fef2f2 !important;
    border-color: #fca5a5 !important;
    color: #ef4444 !important;
}

/* History status alert badge */
.history-status-alert {
    padding: 6px 10px !important;
    border-radius: 8px !important;
    font-size: 11.5px !important;
    font-weight: 500 !important;
    text-align: center !important;
    margin-top: 6px !important;
    line-height: 1.35 !important;
}
.history-status-alert.is-warning {
    background: #fffbeb !important;
    border: 1px solid #fde68a !important;
    color: #b45309 !important;
}
.history-status-alert.is-success {
    background: #ecfdf5 !important;
    border: 1px solid #a7f3d0 !important;
    color: #065f46 !important;
}
.history-status-alert.is-error {
    background: #fef2f2 !important;
    border: 1px solid #fecaca !important;
    color: #b91c1c !important;
}

/* Interactive Prompt Starter Chips */
.feature-list {
    margin-top: 14px;
    padding-top: 12px;
    border-top: 1px solid #f1f5f9;
}

.prompt-chip,
button.prompt-chip {
    width: 100% !important;
    min-height: 32px !important;
    height: auto !important;
    padding: 6px 10px !important;
    font-size: 11.5px !important;
    font-weight: 500 !important;
    color: #475569 !important;
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    margin-bottom: 5px !important;
    text-align: left !important;
    cursor: pointer !important;
    line-height: 1.35 !important;
    transition: all 0.15s ease !important;
    box-shadow: none !important;
}

.prompt-chip:hover {
    background: #eff6ff !important;
    border-color: #93c5fd !important;
    color: #1d4ed8 !important;
    transform: translateX(2px) !important;
}

.sync-note {
    color: #94a3b8 !important;
    font-size: 11px;
    line-height: 1.35;
    text-align: center;
}

/* ── Main Chat Area (Flex Viewport Locked) ── */
.main-chat-area {
    flex: 1 1 0% !important;
    height: 100% !important;
    max-height: 100% !important;
    min-height: 0 !important;
    padding: 16px 28px 12px 28px !important;
    background: #f8fafc !important;
    display: flex !important;
    flex-direction: column !important;
    overflow: hidden !important;
}

.main-chat-area > .wrap {
    max-width: 1040px !important;
    width: 100% !important;
    height: 100% !important;
    max-height: 100% !important;
    min-height: 0 !important;
    margin: 0 auto !important;
    padding: 0 !important;
    display: flex !important;
    flex-direction: column !important;
    overflow: hidden !important;
}

.chat-shell {
    height: 100% !important;
    max-height: 100% !important;
    min-height: 0 !important;
    background: transparent !important;
    border: 0 !important;
    display: flex !important;
    flex-direction: column !important;
    overflow: hidden !important;
    gap: 8px !important;
}

.chat-shell > .wrap {
    height: 100% !important;
    max-height: 100% !important;
    min-height: 0 !important;
    display: flex !important;
    flex-direction: column !important;
    padding: 0 !important;
    overflow: hidden !important;
}

/* Chatbot Outer Container (Flex fills remaining vertical space) */
.chatbot-main {
    flex: 1 1 auto !important;
    flex-grow: 1 !important;
    flex-shrink: 1 !important;
    min-height: 0 !important;
    max-height: none !important;
    height: 100% !important;
    overflow: hidden !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 16px !important;
    background: #ffffff !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02), 0 4px 16px rgba(15, 23, 42, 0.04) !important;
    margin-bottom: 2px !important;
}

.chatbot-main,
.chatbot-main .panel,
.chatbot-main .message-row,
.chatbot-main .message-wrap,
.chatbot-main [data-testid="chatbot"],
.chatbot-main > .wrap {
    background: #ffffff !important;
}

.chatbot-main > .wrap {
    padding: 20px 24px 60px 24px !important;
    border: 0 !important;
    box-shadow: none !important;
    display: flex !important;
    flex-direction: column !important;
    overflow-y: auto !important;
    height: 100% !important;
    max-height: 100% !important;
    min-height: 0 !important;
    scroll-padding-bottom: 60px !important;
}

/* Ensure the last message always has ample clearance from bottom edge */
.chatbot-main .message-row:last-child {
    margin-bottom: 36px !important;
    padding-bottom: 16px !important;
}

/* Suppress ugly Gradio autoscroll indicator and message action icons */
.chatbot-main button.scroll-down,
.chatbot-main .scroll-down,
.chatbot-main .scroll-hide,
.chatbot-main button[aria-label*="down"],
.chatbot-main button[aria-label*="Scroll"],
.chatbot-main button[aria-label*="Copy"],
.chatbot-main button[title*="Copy"],
.chatbot-main button[aria-label*="Share"],
.chatbot-main button[title*="Share"],
.chatbot-main .message-buttons,
.chatbot-main .message-actions,
.chatbot-main button.icon-button,
.chatbot-main [data-testid="copy-button"],
.chatbot-main [data-testid="share-button"],
.chatbot-main [aria-label*="delete"] {
    display: none !important;
    opacity: 0 !important;
    visibility: hidden !important;
    pointer-events: none !important;
    width: 0 !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* Welcome empty state */
.chat-empty-state {
    text-align: center;
    padding: clamp(30px, 10vh, 100px) 20px 20px 20px;
    max-width: 580px;
    margin: 0 auto;
}
.chat-empty-icon {
    font-size: 32px;
    color: #2563eb;
    margin-bottom: 12px;
}
.chat-empty-title {
    font-size: 26px !important;
    font-weight: 700 !important;
    color: #0f172a !important;
    letter-spacing: -0.5px !important;
    line-height: 1.25 !important;
    margin-bottom: 8px !important;
}
.chat-empty-subtitle {
    font-size: 14px !important;
    color: #64748b !important;
    line-height: 1.5 !important;
}

/* Message Bubbles */
.chatbot-main .message-wrap {
    display: flex !important;
    flex-direction: column !important;
    gap: 12px !important;
    width: 100% !important;
}

.chatbot-main .message-row {
    display: flex !important;
    flex-direction: column !important;
    width: 100% !important;
    clear: both !important;
    margin-bottom: 14px !important;
}

/* User Message: CarDekho Warm Cream Pill Bubble with Dark Slate Text */
.message.user,
.message.user *,
.message.user p,
.message.user span,
.message.user div,
.message.user strong,
.message.user em {
    color: #1e293b !important;
}

.message.user {
    display: block !important;
    background: #fdf8f4 !important;
    border: 1px solid #fed7aa !important;
    border-radius: 16px 16px 4px 16px !important;
    padding: 10px 18px !important;
    width: fit-content !important;
    max-width: min(640px, 80%) !important;
    margin-left: auto !important;
    margin-right: 0 !important;
    margin-bottom: 12px !important;
    font-size: 14.5px !important;
    font-weight: 500 !important;
    line-height: 1.5 !important;
    box-sizing: border-box !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
    clear: both !important;
}

/* Bot Message: Structured Automotive Presentation with Clean Alignment */
.message.bot {
    display: block !important;
    background: transparent !important;
    border: 0 !important;
    color: #0f172a !important;
    padding: 8px 16px 20px 16px !important;
    width: 100% !important;
    max-width: 920px !important;
    margin-right: auto !important;
    margin-left: 0 !important;
    font-size: 14.5px !important;
    line-height: 1.7 !important;
    clear: both !important;
}

.message.bot p {
    margin-bottom: 12px !important;
    line-height: 1.65 !important;
}

.message.bot h3 {
    font-size: 16px !important;
    font-weight: 700 !important;
    color: #0f172a !important;
    margin: 20px 0 10px 0 !important;
    padding-bottom: 6px !important;
    border-bottom: 1px solid #f1f5f9 !important;
    display: flex !important;
    align-items: center !important;
    gap: 6px !important;
}

.message.bot h4 {
    font-size: 15px !important;
    font-weight: 600 !important;
    color: #1e293b !important;
    margin: 16px 0 8px 0 !important;
}

/* Clear List Indentation & Hierarchical Bullet Spacing */
.message.bot ul,
.message.bot ol {
    padding-left: 26px !important;
    margin: 10px 0 16px 0 !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 6px !important;
}

.message.bot li {
    color: #1e293b !important;
    font-size: 14px !important;
    line-height: 1.65 !important;
    margin-bottom: 4px !important;
}

.message.bot li > ul,
.message.bot li > ol {
    padding-left: 20px !important;
    margin: 6px 0 8px 0 !important;
}

.message.bot hr {
    border: none !important;
    border-top: 1px solid #e2e8f0 !important;
    margin: 18px 0 !important;
    width: 100% !important;
}

/* Automotive Specs & Maintenance Markdown Tables */
.message.bot table {
    width: 100% !important;
    border-collapse: separate !important;
    border-spacing: 0 !important;
    margin: 12px 0 18px 0 !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    background: #ffffff !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02) !important;
}

.message.bot th {
    background: #f1f5f9 !important;
    color: #334155 !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 10px 14px !important;
    text-align: left !important;
    border-bottom: 1px solid #e2e8f0 !important;
}

.message.bot td {
    padding: 10px 14px !important;
    border-bottom: 1px solid #f1f5f9 !important;
    color: #1e293b !important;
    font-size: 13.5px !important;
    line-height: 1.5 !important;
}

.message.bot tr:nth-child(even) td {
    background: #f8fafc !important;
}

.message.bot tr:last-child td {
    border-bottom: none !important;
}

.message.bot code {
    background: #eff6ff !important;
    color: #1d4ed8 !important;
    padding: 2px 7px !important;
    border-radius: 6px !important;
    font-size: 12.5px !important;
    border: 1px solid #dbeafe !important;
    font-family: 'JetBrains Mono', monospace !important;
}

.message.bot blockquote {
    border-left: 3px solid #2563eb !important;
    background: #f8fafc !important;
    padding: 10px 14px !important;
    margin: 14px 0 !important;
    border-radius: 0 8px 8px 0 !important;
    color: #334155 !important;
}

/* ─────────────────────────────────────────────
   Car Recommendation Card View
   ───────────────────────────────────────────── */
.car-card {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 14px !important;
    padding: 18px 22px !important;
    margin: 16px 0 20px 0 !important;
    box-shadow: 0 3px 12px rgba(15, 23, 42, 0.05) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    overflow: hidden !important;
}

.car-card:hover {
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.08) !important;
    border-color: #cbd5e1 !important;
}

.car-card-header {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    flex-wrap: wrap !important;
    gap: 10px !important;
    padding-bottom: 12px !important;
    border-bottom: 1px solid #f1f5f9 !important;
    margin-bottom: 14px !important;
}

.car-title-group {
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
    flex-wrap: wrap !important;
}

.car-name {
    font-size: 17.5px !important;
    font-weight: 700 !important;
    color: #0f172a !important;
    margin: 0 !important;
    padding: 0 !important;
    border: none !important;
    letter-spacing: -0.2px !important;
}

.car-segment-badge {
    background: #eff6ff !important;
    color: #2563eb !important;
    border: 1px solid #bfdbfe !important;
    padding: 3px 10px !important;
    border-radius: 9999px !important;
    font-size: 11.5px !important;
    font-weight: 600 !important;
}

.car-price-badge {
    background: #f0fdf4 !important;
    color: #166534 !important;
    border: 1px solid #bbf7d0 !important;
    padding: 4px 12px !important;
    border-radius: 8px !important;
    font-size: 13.5px !important;
    font-weight: 700 !important;
    letter-spacing: -0.1px !important;
}

.car-specs-grid {
    display: grid !important;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)) !important;
    gap: 10px !important;
    margin: 12px 0 16px 0 !important;
}

.spec-card {
    background: #f8fafc !important;
    border: 1px solid #f1f5f9 !important;
    border-radius: 10px !important;
    padding: 10px 12px !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 3px !important;
}

.spec-label {
    font-size: 11px !important;
    font-weight: 600 !important;
    color: #64748b !important;
    display: flex !important;
    align-items: center !important;
    gap: 4px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.3px !important;
}

.spec-value {
    font-size: 13.5px !important;
    font-weight: 600 !important;
    color: #1e293b !important;
    line-height: 1.35 !important;
}

.car-features-section {
    margin: 12px 0 !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 7px !important;
}

.features-title {
    font-size: 12px !important;
    font-weight: 600 !important;
    color: #475569 !important;
}

.feature-pills {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 6px !important;
}

.feature-pill {
    background: #f1f5f9 !important;
    color: #334155 !important;
    border: 1px solid #e2e8f0 !important;
    padding: 3px 9px !important;
    border-radius: 6px !important;
    font-size: 12px !important;
    font-weight: 500 !important;
}

.car-best-for {
    background: #fefce8 !important;
    border-left: 3px solid #f59e0b !important;
    border-radius: 0 8px 8px 0 !important;
    padding: 10px 14px !important;
    margin-top: 12px !important;
    color: #854d0e !important;
    font-size: 13px !important;
    line-height: 1.5 !important;
}

/* Docked Floating Input Bar (Sleek Elevated Capsule, No Collision) */
.floating-input-bar {
    flex: 0 0 auto !important;
    flex-grow: 0 !important;
    flex-shrink: 0 !important;
    height: auto !important;
    min-height: 52px !important;
    max-height: 58px !important;
    width: 100% !important;
    background: #ffffff !important;
    border: 1px solid #d1d5db !important;
    border-radius: 14px !important;
    padding: 4px 6px 4px 14px !important;
    box-shadow: 0 4px 16px -2px rgba(15, 23, 42, 0.08) !important;
    margin: 10px auto 0 !important;
    display: flex !important;
    align-items: center !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
}
.floating-input-bar:focus-within {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15), 0 4px 16px -2px rgba(15, 23, 42, 0.1) !important;
}

.floating-input-bar > .wrap {
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    width: 100% !important;
    height: auto !important;
    min-height: 0 !important;
    flex-grow: 0 !important;
    padding: 0 !important;
    gap: 8px !important;
}

/* Eliminate Inner Double-Border: Strip all inner borders and backgrounds from Gradio's textbox elements */
.floating-input-bar .gradio-textbox,
.floating-input-bar [data-testid="textbox"],
.floating-input-bar .block,
.floating-input-bar .wrap,
.floating-input-bar .input-container,
.floating-input-bar .container,
.floating-input-bar fieldset,
.floating-input-bar label {
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    background: transparent !important;
    padding: 0 !important;
    margin: 0 !important;
}

.floating-input-bar .input-container:focus-within {
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
}

.floating-input-bar textarea {
    background: transparent !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    color: #0f172a !important;
    font-size: 14px !important;
    font-family: 'Inter', sans-serif !important;
    padding: 8px 10px !important;
    resize: none !important;
    caret-color: #2563eb !important;
    line-height: 1.4 !important;
}
.floating-input-bar textarea:focus {
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
}
.floating-input-bar textarea::placeholder { color: #94a3b8 !important; }

.send-button,
button.send-button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 13.5px !important;
    padding: 0 20px !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
    min-height: 40px !important;
    max-height: 40px !important;
    height: 40px !important;
    box-shadow: 0 2px 6px rgba(37, 99, 235, 0.25) !important;
    flex-shrink: 0 !important;
}
.send-button:hover {
    background: linear-gradient(135deg, #1d4ed8, #1e40af) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35) !important;
}

/* Disclaimer text */
.disclaimer-text {
    flex-shrink: 0 !important;
    text-align: center;
    font-size: 11.5px;
    color: #9ca3af;
    margin: 6px 0 0 0;
    line-height: 1.3;
    font-style: italic;
}

@media (max-width: 900px) {
    .app-workspace { flex-direction: column !important; }
    .sidebar-panel {
        width: 100% !important;
        max-width: 100% !important;
        height: auto !important;
        max-height: 220px !important;
        border-right: 0 !important;
        border-bottom: 1px solid #e2e8f0 !important;
    }
    .main-chat-area { padding: 12px 14px !important; }
    .message.user,
    .message.bot { max-width: 94% !important; }
}

/* Scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #94a3b8; }
"""


# ─────────────────────────────────────────────
# Build UI
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# Force-light theme
# ─────────────────────────────────────────────
FORCED_LIGHT_THEME = gr.themes.Default().set(
    body_background_fill="#f8fafc",
    body_background_fill_dark="#f8fafc",
    background_fill_primary="#ffffff",
    background_fill_primary_dark="#ffffff",
    background_fill_secondary="#f1f5f9",
    background_fill_secondary_dark="#f1f5f9",
    block_background_fill="#ffffff",
    block_background_fill_dark="#ffffff",
    body_text_color="#0f172a",
    body_text_color_dark="#0f172a",
    block_label_text_color="#0f172a",
    block_label_text_color_dark="#0f172a",
    input_background_fill="#ffffff",
    input_background_fill_dark="#ffffff",
    border_color_primary="#e2e8f0",
    border_color_primary_dark="#e2e8f0",
)

# Belt-and-suspenders: also strip any 'dark' class Gradio adds to
# <body>/<html> at load time, in case a future Gradio version still
# toggles a class-based dark mode alongside the theme variables.
AUTO_ROUTER_JS = """
() => {
    // 1. Force light theme
    document.body.classList.remove('dark');
    document.documentElement.classList.remove('dark');

    // 2. Extract Route from Query Param (if redirected from F5 refresh / direct URL)
    const getTargetRoute = () => {
        const urlParams = new URLSearchParams(window.location.search);
        const routeParam = urlParams.get('route');
        if (routeParam) {
            history.replaceState({}, '', routeParam);
            return routeParam;
        }
        return window.location.pathname;
    };

    // 3. State-Driven URL Route Sync
    const syncUrlWithState = () => {
        const authView = document.getElementById('auth-view');
        const mainView = document.getElementById('main-view');

        const isMainVisible = mainView && (!mainView.classList.contains('hidden') && getComputedStyle(mainView).display !== 'none');
        const isAuthVisible = authView && (!authView.classList.contains('hidden') && getComputedStyle(authView).display !== 'none');

        if (isMainVisible) {
            const checkedRadio = document.querySelector('.history-select input[type="radio"]:checked');
            let targetPath = '/chat';
            if (checkedRadio && checkedRadio.value) {
                targetPath = '/chat/' + checkedRadio.value;
            }
            if (window.location.pathname !== targetPath) {
                if (checkedRadio && checkedRadio.value) {
                    history.pushState({}, '', targetPath);
                } else if (window.location.pathname === '/login' || window.location.pathname === '/signup' || window.location.pathname === '/') {
                    history.replaceState({}, '', '/chat');
                }
            }
        } else if (isAuthVisible) {
            const signupForm = document.querySelector('.auth-panel:nth-of-type(2)');
            const isSignup = signupForm && (!signupForm.classList.contains('hidden') && getComputedStyle(signupForm).display !== 'none');
            const targetPath = isSignup ? '/signup' : '/login';
            if (window.location.pathname !== targetPath && (window.location.pathname.startsWith('/chat') || window.location.pathname === '/')) {
                history.replaceState({}, '', targetPath);
            }
        }
    };

    // Initial checks and continuous state sync
    setTimeout(syncUrlWithState, 200);
    setTimeout(syncUrlWithState, 600);
    setInterval(syncUrlWithState, 500);

    // Watch for DOM visibility changes on auth-view and main-view
    const observer = new MutationObserver(syncUrlWithState);
    setTimeout(() => {
        const authView = document.getElementById('auth-view');
        const mainView = document.getElementById('main-view');
        if (authView) observer.observe(authView, { attributes: true, attributeFilter: ['style', 'class'], subtree: true });
        if (mainView) observer.observe(mainView, { attributes: true, attributeFilter: ['style', 'class'], subtree: true });
    }, 400);

    // 4. Initial URL Route Sync on Load for direct links / redirects
    const handleRoute = () => {
        const route = getTargetRoute();
        if (route === '/signup') {
            const btns = document.querySelectorAll('.auth-tab');
            if (btns.length >= 2 && !btns[1].classList.contains('is-active')) {
                btns[1].click();
            }
        } else if (route === '/login') {
            const btns = document.querySelectorAll('.auth-tab');
            if (btns.length >= 1 && !btns[0].classList.contains('is-active')) {
                btns[0].click();
            }
        } else if (route.startsWith('/chat/')) {
            const sessionId = route.replace('/chat/', '');
            if (sessionId && sessionId !== 'new') {
                setTimeout(() => {
                    const radioInputs = document.querySelectorAll('.history-select input[type="radio"]');
                    for (const inp of radioInputs) {
                        if (inp.value === sessionId) {
                            inp.click();
                            break;
                        }
                    }
                }, 500);
            }
        }
    };
    setTimeout(handleRoute, 250);

    // 5. Radio Item Click Sync
    document.addEventListener('click', (e) => {
        const historyContainer = e.target.closest('.history-select');
        if (historyContainer) {
            const labelOrItem = e.target.closest('label') || e.target.closest('.wrap > *');
            if (labelOrItem) {
                const radio = labelOrItem.querySelector('input[type="radio"]') || (e.target.tagName === 'INPUT' ? e.target : null);
                if (radio && radio.value) {
                    history.pushState({}, '', '/chat/' + radio.value);
                }
            }
            setTimeout(() => {
                const checked = document.querySelector('.history-select input[type="radio"]:checked');
                if (checked && checked.value) {
                    history.pushState({}, '', '/chat/' + checked.value);
                }
            }, 100);
        }
    }, true);

    document.addEventListener('change', (e) => {
        if (e.target && e.target.closest('.history-select')) {
            const checked = document.querySelector('.history-select input[type="radio"]:checked');
            if (checked && checked.value) {
                history.pushState({}, '', '/chat/' + checked.value);
            }
        }
    });

    // 6. New Chat & Delete Chat Buttons URL Sync
    document.addEventListener('click', (e) => {
        if (e.target && e.target.closest('.btn-new-chat')) {
            history.pushState({}, '', '/chat/new');
        }
        if (e.target && e.target.closest('.btn-delete-chat')) {
            history.pushState({}, '', '/chat');
        }
    });

    // 7. Browser Autofill Dispatcher for Gradio Inputs
    document.addEventListener('click', (e) => {
        if (e.target && (e.target.innerText === 'Sign In' || e.target.innerText === 'Create Account')) {
            const inputs = document.querySelectorAll('.auth-panel input');
            inputs.forEach(inp => {
                inp.dispatchEvent(new Event('input', { bubbles: true }));
                inp.dispatchEvent(new Event('change', { bubbles: true }));
            });
        }
    }, true);

    // 8. Browser Back/Forward Navigation
    window.addEventListener('popstate', handleRoute);
}
"""


def build_ui():
    with gr.Blocks(
        title="AutoBot — AI Automobile Assistant",
    ) as demo:

        # Session States
        user_state = gr.State(value=None)
        active_session_id = gr.State(value=None)

        # ═════════════════════════════════════════════════════════════
        # 1. AUTHENTICATION LANDING VIEW (Visible FIRST on page load)
        # ═════════════════════════════════════════════════════════════
        with gr.Column(visible=True, elem_id="auth-view") as auth_view:
            with gr.Column(elem_id="auth-card"):
                with gr.Row(elem_classes="auth-layout"):
                    with gr.Column(scale=1, min_width=360, elem_classes="auth-form-pane"):
                        with gr.Column(elem_classes="auth-form-inner"):
                            gr.HTML("""
                            <div class="auth-form-heading">
                                <div class="auth-mark">✦</div>
                                <div class="auth-brand">AutoBot</div>
                                <p>Your AI Automobile Assistant.</p>
                            </div>
                            """)

                            # Explicit buttons + visible panels are more stable than styling
                            # Gradio Tabs' generated tab-nav/tabitem wrappers.
                            with gr.Row(elem_classes="auth-tab-switch"):
                                login_tab_btn = gr.Button("Sign In", elem_classes=["auth-tab", "is-active"])
                                signup_tab_btn = gr.Button("Sign Up", elem_classes=["auth-tab"])

                            with gr.Column(visible=True, elem_classes="auth-panel") as login_form:
                                gr.HTML("<label class='auth-field-label'>Username or Email</label>")
                                login_id_input = gr.Textbox(show_label=False, placeholder="alex or alex@example.com", lines=1, elem_classes="auth-input")
                                gr.HTML("<label class='auth-field-label'>Password</label>")
                                login_pw_input = gr.Textbox(show_label=False, type="password", placeholder="••••••••", lines=1, elem_classes="auth-input")
                                login_btn = gr.Button("Sign In", variant="primary", size="lg", elem_classes="auth-primary")

                            with gr.Column(visible=False, elem_classes="auth-panel") as signup_form:
                                gr.HTML("<label class='auth-field-label'>Username</label>")
                                signup_name_input = gr.Textbox(show_label=False, placeholder="alex", lines=1, elem_classes="auth-input")
                                gr.HTML("<label class='auth-field-label'>Email</label>")
                                signup_email_input = gr.Textbox(show_label=False, placeholder="alex@example.com", lines=1, elem_classes="auth-input")
                                gr.HTML("<label class='auth-field-label'>Password</label>")
                                signup_pw_input = gr.Textbox(show_label=False, type="password", placeholder="At least 6 characters", lines=1, elem_classes="auth-input")
                                signup_btn = gr.Button("Create Account", variant="primary", size="lg", elem_classes="auth-primary")

                            auth_status = gr.HTML("", elem_id="auth-status")

                            gr.HTML("""
                            <div class="guest-divider">
                                <span>Use without saving history</span>
                            </div>
                            """)
                            guest_btn = gr.Button("Continue as Guest", variant="secondary", size="sm", elem_classes="guest-button")

                    with gr.Column(scale=1, min_width=360, elem_classes="auth-promo-pane"):
                        gr.HTML("""
                        <div class="auth-promo-copy">
                            <div class="auth-mark">✦</div>
                            <h2>AutoBot</h2>
                            <p>Sign in or create an account to save your chat sessions and diagnostic reports.</p>
                        </div>
                        """)

        # ═════════════════════════════════════════════════════════════
        # 2. MAIN APPLICATION VIEW (Replicated Layout Architecture)
        # ═════════════════════════════════════════════════════════════
        with gr.Column(visible=False, elem_id="main-view") as main_view:

            # Top Sticky Header Navigation Bar
            with gr.Row(elem_classes="top-navbar"):
                gr.HTML("""
                <div style="display:flex;align-items:center;gap:12px;">
                    <span style="font-size:18px;font-weight:700;color:#0f172a;letter-spacing:-0.4px;display:flex;align-items:center;gap:7px;">
                        <span style="color:#2563eb;font-size:20px;">✦</span> AutoBot
                    </span>
                    <span style="font-size:12px;color:#64748b;font-weight:500;padding-left:12px;border-left:1px solid #e2e8f0;">Automotive AI Intelligence</span>
                </div>
                """)
                with gr.Row(elem_classes="header-user-controls"):
                    user_badge_md = gr.HTML("<div class='user-pill'><span class='user-pill-dot'>●</span> <strong>Guest User</strong></div>")
                    logout_btn = gr.Button("Sign Out", size="sm", elem_classes="btn-logout")

            with gr.Row(elem_classes="app-workspace"):

                # ── Left Aligned Sidebar Panel (Recent Chats & Prompt Starters) ──
                with gr.Column(scale=1, min_width=240, elem_classes="sidebar-panel"):
                    with gr.Column(elem_classes="sidebar-top-group"):
                        new_chat_btn = gr.Button("+ New Chat", size="sm", elem_classes="btn-new-chat")
                        
                        gr.HTML("<div class='sidebar-title'>🕒 Recent Chats</div>")
                        
                        history_radio = gr.Radio(
                            label="",
                            choices=[],
                            value=None,
                            interactive=True,
                            show_label=False,
                            elem_classes="history-select",
                            container=False,
                        )
                        empty_history_box = gr.HTML("<div class='empty-history-box'>No saved chats yet</div>")
                        delete_history_btn = gr.Button("🗑️ Delete Selected", size="sm", elem_classes="btn-delete-chat")
                        history_status = gr.HTML("")

                        gr.HTML("""
                        <div class="feature-list">
                            <div class="sidebar-title">Prompt Starters</div>
                        </div>
                        """)
                        prompt_btn_1 = gr.Button("🚗 Best SUVs under ₹15L", size="sm", elem_classes="prompt-chip")
                        prompt_btn_2 = gr.Button("🔧 Squeaking brake check", size="sm", elem_classes="prompt-chip")
                        prompt_btn_3 = gr.Button("📅 30,000 km maintenance", size="sm", elem_classes="prompt-chip")
                        prompt_btn_4 = gr.Button("💰 EMI for ₹10L car loan", size="sm", elem_classes="prompt-chip")

                    # Bottom User Profile Footer Card
                    with gr.Column(elem_classes="sidebar-bottom-group"):
                        gr.HTML("<div class='sync-note'>Conversations sync to PostgreSQL when signed in.</div>")

                # ── Main Central Chat Area ──
                with gr.Column(scale=4, elem_classes="main-chat-area"):
                    with gr.Column(elem_classes="chat-shell"):
                        chatbot = gr.Chatbot(
                            value=[],
                            elem_classes="chatbot-main",
                            show_label=False,
                            layout="panel",
                            container=False,
                            placeholder="""
<div class="chat-empty-state">
    <div class="chat-empty-icon">✦</div>
    <h2 class="chat-empty-title">Welcome to AutoBot</h2>
    <p class="chat-empty-subtitle">Your intelligent automobile companion for car buying, diagnostics, maintenance, and finance.</p>
</div>
""",
                        )

                        with gr.Row(elem_classes="floating-input-bar"):
                            msg_input = gr.Textbox(
                                placeholder="Ask Anything about Cars",
                                show_label=False,
                                lines=1,
                                max_lines=4,
                                container=False,
                                scale=8,
                            )
                            send_btn = gr.Button("Send", variant="primary", elem_classes="send-button", scale=1)

                        gr.HTML("<div class='disclaimer-text'><i>Responses may be inaccurate. Be sure to verify important details</i></div>")

        # ── Event Wire Up ───────────────────────────
        login_tab_btn.click(
            fn=show_login_form,
            inputs=[],
            outputs=[login_form, signup_form, login_tab_btn, signup_tab_btn],
            js="() => { history.pushState({}, '', '/login'); }",
        )

        signup_tab_btn.click(
            fn=show_signup_form,
            inputs=[],
            outputs=[login_form, signup_form, login_tab_btn, signup_tab_btn],
            js="() => { history.pushState({}, '', '/signup'); }",
        )

        login_btn.click(
            fn=on_login,
            inputs=[login_id_input, login_pw_input],
            outputs=[user_state, auth_status, auth_view, main_view, history_radio, user_badge_md],
        )
        login_pw_input.submit(
            fn=on_login,
            inputs=[login_id_input, login_pw_input],
            outputs=[user_state, auth_status, auth_view, main_view, history_radio, user_badge_md],
        )

        signup_btn.click(
            fn=on_signup,
            inputs=[signup_name_input, signup_email_input, signup_pw_input],
            outputs=[user_state, auth_status, auth_view, main_view, history_radio, user_badge_md],
        )
        signup_pw_input.submit(
            fn=on_signup,
            inputs=[signup_name_input, signup_email_input, signup_pw_input],
            outputs=[user_state, auth_status, auth_view, main_view, history_radio, user_badge_md],
        )

        guest_btn.click(
            fn=on_guest,
            inputs=[],
            outputs=[user_state, auth_status, auth_view, main_view, history_radio, user_badge_md],
            js="() => { history.pushState({}, '', '/chat'); }",
        )

        logout_btn.click(
            fn=on_logout,
            inputs=[],
            outputs=[
                user_state, active_session_id, chatbot, auth_status, auth_view, main_view,
                history_radio, user_badge_md, login_id_input, login_pw_input,
                signup_name_input, signup_email_input, signup_pw_input
            ],
            js="() => { history.pushState({}, '', '/login'); }",
        )

        new_chat_btn.click(
            fn=on_new_chat,
            inputs=[user_state],
            outputs=[active_session_id, chatbot, history_status],
        )

        history_radio.change(
            fn=on_select_history,
            inputs=[history_radio, user_state],
            outputs=[active_session_id, chatbot, history_status],
        )

        delete_history_btn.click(
            fn=on_delete_history,
            inputs=[history_radio, user_state],
            outputs=[active_session_id, chatbot, history_radio, history_status],
        )

        prompt_btn_1.click(
            fn=lambda: "Give me the latest verified SUV options under 15 Lakhs in India with price and key specs",
            outputs=[msg_input],
        )
        prompt_btn_2.click(
            fn=lambda: "Why are my car brakes making a squeaking noise when stopping? Is it safe to drive?",
            outputs=[msg_input],
        )
        prompt_btn_3.click(
            fn=lambda: "What is the recommended 30,000 km milestone service checklist and estimated cost?",
            outputs=[msg_input],
        )
        prompt_btn_4.click(
            fn=lambda: "Calculate monthly EMI for a 10 Lakh car loan at 9% interest for 5 years with 2 Lakh down payment",
            outputs=[msg_input],
        )

        send_btn.click(
            fn=chat,
            inputs=[msg_input, chatbot, user_state, active_session_id],
            outputs=[chatbot, msg_input, active_session_id, history_radio],
        )
        msg_input.submit(
            fn=chat,
            inputs=[msg_input, chatbot, user_state, active_session_id],
            outputs=[chatbot, msg_input, active_session_id, history_radio],
        )
    return demo
