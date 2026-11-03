"""FraudLens analyst app (Streamlit).

Run it with ``fraudlens-app`` (after ``pip install -e .``), or ``streamlit run streamlit_app.py``.
"""

from __future__ import annotations

import streamlit as st

from fraudlens.app import theme
from fraudlens.app.state import get_world
from fraudlens.app.views import live, performance, rings, simulator, test_payment

GITHUB_URL = "https://github.com/beggumuzun/fraudlens"


def run() -> None:
    st.set_page_config(page_title="FraudLens", page_icon=":material/shield:", layout="wide")

    pages = [
        st.Page(
            live.render,
            title="Live monitor",
            icon=":material/monitoring:",
            url_path="live",
            default=True,
        ),
        st.Page(
            test_payment.render,
            title="Test a payment",
            icon=":material/credit_card:",
            url_path="test-a-payment",
        ),
        st.Page(
            simulator.render,
            title="Attack simulator",
            icon=":material/bug_report:",
            url_path="attack-simulator",
        ),
        st.Page(
            rings.render,
            title="Fraud rings",
            icon=":material/hub:",
            url_path="fraud-rings",
        ),
        st.Page(
            performance.render,
            title="Model performance",
            icon=":material/insights:",
            url_path="model-performance",
        ),
    ]
    page = st.navigation(pages)
    theme.inject()

    with st.sidebar:
        st.markdown(
            '<div class="fl-brand"><span class="fl-brand-mark"></span>FraudLens</div>'
            '<p class="fl-side-note">Real-time, explainable card fraud detection. Every payment '
            "is scored the moment it happens, and every alert says why.</p>"
            '<p class="fl-side-note">All customers and payments are synthetic, made by the '
            "project's own data engine. No real card data is used.</p>",
            unsafe_allow_html=True,
        )
        st.markdown(f"[Source code on GitHub]({GITHUB_URL})")

    get_world()  # build (or fetch the cached) demo world before any page draws
    page.run()


if __name__ == "__main__":
    run()
