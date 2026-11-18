"""Load the real evaluation JSON reports for the Dataset & Evaluation page.

Every report in this project is a **custom** evaluation run.  No official
Ultralytics ``val`` output is stored, so the loader exposes that fact instead of
implying the numbers came from generalised model validation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ROOT_DIR / "evaluation" / "results"


def _read(name: str) -> dict[str, Any] | None:
    path = RESULTS_DIR / name
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _application_report() -> tuple[dict[str, Any] | None, str | None]:
    """Prefer the post-fix 960 px report, then 960, then the 640 px run."""
    for name in (
        "application_evaluation_report_960_dupfix.json",
        "application_evaluation_report_960.json",
        "application_evaluation_report.json",
    ):
        report = _read(name)
        if report is not None:
            return report, name
    return None, None


def load_evaluation() -> dict[str, Any]:
    ppe = _read("evaluation_report.json")
    person = _read("person_evaluation_report.json")
    application, application_name = _application_report()

    class_rows = []
    if ppe and isinstance(ppe.get("class_results"), dict):
        for name, values in ppe["class_results"].items():
            if not isinstance(values, dict):
                continue
            class_rows.append(
                {
                    "name": name,
                    "ground_truth": values.get("ground_truth", 0),
                    "precision": values.get("precision", 0.0),
                    "recall": values.get("recall", 0.0),
                    "map50": values.get("mAP50", 0.0),
                    "map50_95": values.get("mAP50_95", 0.0),
                }
            )
        class_rows.sort(key=lambda row: row["name"])

    return {
        "ppe": ppe,
        "person": person,
        "application": application,
        "application_file": application_name,
        "class_rows": class_rows,
        # The project contains no Ultralytics `val` artifact.  Keep this
        # explicit so the template never presents custom numbers as official.
        "official_available": False,
        "results_dir": str(RESULTS_DIR),
    }
