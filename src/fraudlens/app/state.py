"""The single shared demo world, built once per server and cached."""

from __future__ import annotations

import os

import streamlit as st

from fraudlens.app.world import APP_CONFIG, DemoWorld, build_demo_world
from fraudlens.data.generator import GeneratorConfig

# A smaller world for automated tests: same code paths, a few seconds to build.
SMALL_CONFIG = GeneratorConfig(n_customers=400, n_days=30, seed=11, fraud_customer_share=0.08)


@st.cache_resource(
    show_spinner=(
        "Building the demo world: generating ~190,000 card payments and training the "
        "fraud model. This happens once, on first launch, and takes about 30 seconds..."
    )
)
def get_world() -> DemoWorld:
    small = os.environ.get("FRAUDLENS_DEMO_SIZE") == "small"
    return build_demo_world(SMALL_CONFIG if small else APP_CONFIG)
