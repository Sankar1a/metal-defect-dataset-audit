"""Launcher that works with `python -I` (isolated mode drops the script directory from sys.path).

  python -I dsqc_run.py scan --spec specs/gc10_raw.yaml -o OUT
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsqc.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
