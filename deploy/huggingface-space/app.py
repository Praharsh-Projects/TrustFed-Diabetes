from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fl_diabetes.dashboard_app import create_dashboard_app


def _env_path(name: str, default: Path | None = None) -> str | None:
    value = os.getenv(name)
    if value:
        return value
    return str(default) if default is not None else None


RESULTS_DIR = _env_path("RESULTS_DIR", ROOT / "results" / "full_cdc_polished_summary")
VISUAL_RESULTS_DIR = _env_path("VISUAL_RESULTS_DIR")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8050"))

app = create_dashboard_app(results_dir=RESULTS_DIR, visual_results_dir=VISUAL_RESULTS_DIR)
server = app.server


if __name__ == "__main__":
    try:
        from waitress import serve
    except ModuleNotFoundError:
        app.run(host=HOST, port=PORT, debug=False)
    else:
        serve(server, host=HOST, port=PORT)
