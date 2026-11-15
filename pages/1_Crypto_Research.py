"""Independent crypto lab page; the existing equity analyzer remains the main page."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parents[1] / "crypto_lab" / "dashboard.py"), run_name="__main__")
