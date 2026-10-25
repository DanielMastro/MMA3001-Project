"""
run_tests.py - run every pytest test and save the report to Docs/test_report.txt.

    python Scripts/run_tests.py        (run from the repo folder)

Where this fits in the project
------------------------------
The tests in Scripts/Tests/ check the pipeline against a known-answer motor,
the finite-difference convergence rates, the four flux models and the Zenodo
converter. Running pytest on its own only prints the results to the screen;
this script also saves them, so the report in Docs/ always matches the code
it was run on. run_all.py calls it as its last step, so one command produces
every result and the test report.

It runs exactly the same command as typing `python -m pytest -v` in the repo
folder, using the same Python interpreter that is running this script.
"""

from __future__ import annotations

__docformat__ = "numpy"   # docstring style, read by pdoc

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent            # the repo folder
REPORT = ROOT / "Docs" / "test_report.txt"


def write_test_report(report_path=REPORT):
    """Run `pytest -v` from the repository folder and save everything it prints.

    pytest runs as a separate process, so it behaves exactly as it does in
    the terminal.

    Parameters
    ----------
    report_path : str or pathlib.Path, optional
        Where to save the report. Default Docs/test_report.txt.

    Returns
    -------
    exit_code : int
        pytest's exit code: 0 means every test passed; anything else means at
        least one failed, and the report says which and why.
    """
    # Run pytest as a separate process so it behaves exactly as it does in the
    # terminal. PYTHONIOENCODING makes it write UTF-8 even on Windows, where
    # captured output would otherwise use the older cp1252 encoding.
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run([sys.executable, "-m", "pytest", "-v"], cwd=ROOT, env=env,
                            capture_output=True, text=True, encoding="utf-8", errors="replace")

    # Save the full output (stderr as well, in case pytest itself fails to start)
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(result.stdout + result.stderr, encoding="utf-8")

    # Echo pytest's final summary line, e.g. "22 passed in 16.4s"
    lines = result.stdout.strip().splitlines()
    print(f"  {lines[-1].strip('= ') if lines else 'pytest produced no output'}")
    print(f"  saved {report_path.relative_to(ROOT)}")
    return result.returncode


if __name__ == "__main__":
    sys.exit(write_test_report())
