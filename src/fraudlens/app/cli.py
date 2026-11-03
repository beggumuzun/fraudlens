"""``fraudlens-app``: start the Streamlit app and open it in the browser."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    app = Path(__file__).with_name("main.py")
    sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(app), *sys.argv[1:]]))


if __name__ == "__main__":
    main()
