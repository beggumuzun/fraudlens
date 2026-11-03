"""Entry point for Streamlit Community Cloud (and ``streamlit run streamlit_app.py``).

The cloud installs requirements.txt but not this package itself, so the source folder is
put on the import path here; locally, ``pip install -e .`` makes this a no-op.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fraudlens.app.main import run  # noqa: E402

run()
