"""Visual identity: the night-shift fraud room of a UK bank.

Deep ink-navy surfaces, condensed signage type (Barlow), and colour that always means
something: amber = risk rising, red = alert, teal = genuine / approved. Nothing is
coloured for decoration.

Every page draws with the helpers below, so the look stays consistent.
"""

from __future__ import annotations

import html
import math

import altair as alt
import streamlit as st

INK = "#0E1726"  # page background
PANEL = "#152238"  # raised surfaces
LINE = "#24344F"  # dividers, grid
TEXT = "#E6EDF7"
MUTED = "#8A9BB4"
CALM = "#3A5378"  # an ordinary payment
AMBER = "#F2B33D"  # risk rising
RED = "#FF4D5E"  # alert
TEAL = "#3CCFB4"  # genuine / approved

FONT = "Barlow"
DISPLAY = "Barlow Condensed"

SCORE_FLOOR = 1e-6  # lowest fraud score drawn on log scales

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600&family=Barlow+Condensed:wght@500;600;700&display=swap');

.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp button,
.stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stCaptionContainer"] {{
    font-family: '{FONT}', system-ui, sans-serif;
}}
.stApp h1, .stApp h2, .stApp h3 {{
    font-family: '{DISPLAY}', '{FONT}', sans-serif;
    letter-spacing: -0.01em;
    color: {TEXT};
}}
.stApp h1 {{ font-size: 3.1rem; font-weight: 600; line-height: 1.02; padding-bottom: 0.2rem; }}
.stApp h2 {{ font-size: 1.75rem; font-weight: 600; }}
.stApp h3 {{ font-size: 1.35rem; font-weight: 600; }}
[data-testid="stMainBlockContainer"] {{ padding-top: 2.4rem; max-width: 1320px; }}
[data-testid="stSidebar"] {{ background: #0A1220; border-right: 1px solid {LINE}; }}
[data-testid="stAppDeployButton"] {{ display: none; }}
[data-testid="stMetricValue"] {{ font-family: '{DISPLAY}', sans-serif; }}

.fl-lede {{ color: {MUTED}; font-size: 1.08rem; max-width: 68ch; line-height: 1.55; margin: 0 0 1.2rem; }}

.fl-brand {{ font-family: '{DISPLAY}', sans-serif; font-size: 2rem; font-weight: 700; color: {TEXT};
             line-height: 1; margin: 0.2rem 0 0.4rem; }}
.fl-brand-mark {{ display: inline-block; width: 0.55rem; height: 1.45rem; background: {RED};
                  margin-right: 0.5rem; vertical-align: -0.12rem; border-radius: 2px; }}
.fl-side-note {{ color: {MUTED}; font-size: 0.92rem; line-height: 1.5; }}

/* Instrument panel: one strip divided by hairlines, not a row of cards */
.fl-panel {{ display: grid; grid-template-columns: repeat(var(--n), 1fr); border-top: 1px solid {LINE};
             border-bottom: 1px solid {LINE}; margin: 0.4rem 0 1.4rem; }}
.fl-kpi {{ padding: 0.9rem 1.1rem 0.8rem; border-left: 1px solid {LINE}; }}
.fl-kpi:first-child {{ border-left: none; padding-left: 0.1rem; }}
.fl-kpi-value {{ font-family: '{DISPLAY}', sans-serif; font-size: 2.5rem; font-weight: 600;
                 line-height: 1; color: {TEXT}; font-variant-numeric: tabular-nums; }}
.fl-kpi-label {{ color: {MUTED}; font-size: 0.92rem; margin-top: 0.35rem; }}
.fl-kpi--alert .fl-kpi-value {{ color: {RED}; }}
.fl-kpi--ok .fl-kpi-value {{ color: {TEAL}; }}
.fl-kpi--warn .fl-kpi-value {{ color: {AMBER}; }}
@media (max-width: 760px) {{
  .fl-panel {{ grid-template-columns: repeat(2, 1fr); }}
  .fl-kpi:nth-child(odd) {{ border-left: none; padding-left: 0.1rem; }}
}}

/* Reasons: how much of the suspicion each signal carries */
.fl-reason {{ margin: 0.55rem 0; }}
.fl-reason-text {{ color: {TEXT}; font-size: 0.97rem; line-height: 1.35; }}
.fl-reason-text b {{ color: {AMBER}; font-weight: 600; margin-right: 0.35rem; }}
.fl-reason-bar {{ height: 4px; background: {LINE}; border-radius: 2px; margin-top: 0.3rem; }}
.fl-reason-bar span {{ display: block; height: 100%; background: {AMBER}; border-radius: 2px; }}
.fl-empty {{ color: {MUTED}; font-size: 0.95rem; }}

/* Alert queue entries */
.fl-alert {{ background: {PANEL}; border-left: 4px solid {RED}; border-radius: 4px;
             padding: 0.85rem 1rem 0.7rem; margin-bottom: 0.75rem; }}
.fl-alert--false {{ border-left-color: {MUTED}; }}
.fl-alert-head {{ display: flex; justify-content: space-between; gap: 0.8rem; align-items: baseline; }}
.fl-amount {{ font-family: '{DISPLAY}', sans-serif; font-size: 1.6rem; font-weight: 600; color: {TEXT}; }}
.fl-truth {{ font-size: 0.85rem; font-weight: 600; color: {RED}; text-align: right; }}
.fl-alert--false .fl-truth {{ color: {MUTED}; }}
.fl-meta {{ color: {MUTED}; font-size: 0.92rem; margin: 0.1rem 0 0.35rem; }}

/* Decision block */
.fl-decision {{ border-radius: 6px; padding: 1.1rem 1.25rem; margin-bottom: 1rem; border: 1px solid; }}
.fl-decision-title {{ font-family: '{DISPLAY}', sans-serif; font-size: 2.2rem; font-weight: 700; line-height: 1; }}
.fl-decision-sub {{ color: {TEXT}; margin-top: 0.45rem; font-size: 1rem; line-height: 1.45; }}
.fl-decision--alert {{ background: rgba(255, 77, 94, 0.10); border-color: rgba(255, 77, 94, 0.55); }}
.fl-decision--alert .fl-decision-title {{ color: {RED}; }}
.fl-decision--ok {{ background: rgba(60, 207, 180, 0.08); border-color: rgba(60, 207, 180, 0.45); }}
.fl-decision--ok .fl-decision-title {{ color: {TEAL}; }}

/* Risk meter: log scale from an ordinary payment to near-certain fraud */
.fl-meter {{ margin: 0.4rem 0 1.3rem; }}
.fl-meter-track {{ position: relative; height: 10px; border-radius: 5px;
                   background: linear-gradient(90deg, {CALM} 0%, {CALM} 45%, {AMBER} 72%, {RED} 100%); }}
.fl-meter-threshold {{ position: absolute; top: -6px; width: 2px; height: 22px; background: {TEXT}; }}
.fl-meter-dot {{ position: absolute; top: -5px; width: 20px; height: 20px; margin-left: -10px;
                 border-radius: 50%; background: {TEXT}; border: 4px solid {INK}; box-shadow: 0 0 0 2px {TEXT}; }}
.fl-meter-scale {{ display: flex; justify-content: space-between; color: {MUTED}; font-size: 0.85rem; margin-top: 0.55rem; }}

/* Attack steps */
.fl-step {{ display: grid; grid-template-columns: 4.5rem 1fr 7rem 8.5rem; gap: 0.8rem; align-items: center;
            border-left: 4px solid {CALM}; background: {PANEL}; border-radius: 4px;
            padding: 0.6rem 0.9rem; margin: 0.5rem 0 0.2rem; }}
.fl-step--alert {{ border-left-color: {RED}; }}
.fl-step--blocked {{ border-left-color: {AMBER}; opacity: 0.75; }}
.fl-step-n {{ font-family: '{DISPLAY}', sans-serif; font-size: 1.4rem; font-weight: 600; color: {TEXT}; }}
.fl-step-time {{ color: {MUTED}; font-size: 0.85rem; }}
.fl-step-what {{ color: {TEXT}; line-height: 1.35; }}
.fl-step-what span {{ color: {MUTED}; }}
.fl-step-risk {{ font-family: '{DISPLAY}', sans-serif; font-size: 1.3rem; color: {TEXT}; }}
.fl-status {{ font-weight: 600; font-size: 0.92rem; }}
.fl-status--alert {{ color: {RED}; }}
.fl-status--blocked {{ color: {AMBER}; }}
.fl-status--passed {{ color: {MUTED}; }}
@media (max-width: 760px) {{ .fl-step {{ grid-template-columns: 3.5rem 1fr; }} }}
</style>
"""


def inject() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def _e(text) -> str:
    return html.escape(str(text))


def page_header(title: str, lede: str) -> None:
    st.title(title)
    st.markdown(f'<p class="fl-lede">{lede}</p>', unsafe_allow_html=True)


def kpi_panel(items: list[tuple[str, str, str | None]]) -> None:
    """One instrument strip. items = (label, value, tone) with tone in alert / ok / warn / None."""
    cells = "".join(
        f'<div class="fl-kpi{" fl-kpi--" + tone if tone else ""}">'
        f'<div class="fl-kpi-value">{_e(value)}</div><div class="fl-kpi-label">{_e(label)}</div></div>'
        for label, value, tone in items
    )
    st.markdown(
        f'<div class="fl-panel" style="--n:{len(items)}">{cells}</div>', unsafe_allow_html=True
    )


def reasons_html(reasons: list[dict], empty_text: str = "Nothing unusual stands out.") -> str:
    if not reasons:
        return f'<div class="fl-empty">{_e(empty_text)}</div>'
    return "".join(
        f'<div class="fl-reason"><div class="fl-reason-text"><b>{r["share"]:.0%}</b>{_e(r["text"])}</div>'
        f'<div class="fl-reason-bar"><span style="width:{min(max(r["share"], 0), 1) * 100:.1f}%"></span></div></div>'
        for r in reasons
    )


def decision(alert: bool, title: str, sub: str) -> None:
    tone = "alert" if alert else "ok"
    st.markdown(
        f'<div class="fl-decision fl-decision--{tone}"><div class="fl-decision-title">{_e(title)}</div>'
        f'<div class="fl-decision-sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def _log_position(score: float) -> float:
    low = math.log10(SCORE_FLOOR)
    return min(max((math.log10(max(score, SCORE_FLOOR)) - low) / -low, 0.0), 1.0) * 100


def risk_meter(score: float, threshold: float) -> None:
    """Where this payment sits between 'ordinary' and 'near-certain fraud', on a log scale."""
    st.markdown(
        f'<div class="fl-meter"><div class="fl-meter-track">'
        f'<div class="fl-meter-threshold" style="left:{_log_position(threshold):.1f}%"></div>'
        f'<div class="fl-meter-dot" style="left:{_log_position(score):.1f}%"></div></div>'
        f'<div class="fl-meter-scale"><span>Ordinary payment</span><span>Alert line</span>'
        f"<span>Near-certain fraud</span></div></div>",
        unsafe_allow_html=True,
    )


def style_chart(chart: alt.TopLevelMixin) -> alt.TopLevelMixin:
    """Dark, quiet axes so the data carries the colour."""
    return (
        chart.configure(background="transparent", font=FONT)
        .configure_view(stroke=None)
        .configure_axis(
            labelColor=MUTED,
            titleColor=MUTED,
            gridColor=LINE,
            domainColor=LINE,
            tickColor=LINE,
            labelFontSize=12,
            titleFontSize=12,
            titleFontWeight=500,
        )
        .configure_legend(labelColor=TEXT, titleColor=MUTED, labelFontSize=12)
    )


def show_chart(chart: alt.TopLevelMixin) -> None:
    st.altair_chart(style_chart(chart), width="stretch", theme=None)
