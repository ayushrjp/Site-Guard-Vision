"""Run existing application tests with isolated SQLite and preview cleanup."""
from __future__ import annotations

import hashlib
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "reports" / "ppe_evaluation" / "application_validation"
TESTS = ROOT / "tests"
RENDERED = ROOT / "static" / "rendered"
DEFAULT_DB = ROOT / "data" / "siteguard.db"


def sha256(path):
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    previous_db_hash = sha256(DEFAULT_DB)
    previous_db_env = os.environ.get("SITEGUARD_DB_PATH")
    before_rendered = {p.name for p in RENDERED.iterdir() if p.is_file()}
    buffer = io.StringIO()
    with tempfile.TemporaryDirectory(prefix="siteguard-task6-db-") as tmp:
        os.environ["SITEGUARD_DB_PATH"] = str(Path(tmp) / "isolated-test.db")
        try:
            suite = unittest.defaultTestLoader.discover(str(TESTS), pattern="test_*.py")
            with redirect_stdout(buffer), redirect_stderr(buffer):
                result = unittest.TextTestRunner(stream=buffer, verbosity=2).run(suite)
            text = buffer.getvalue()
        finally:
            # Only remove uniquely named previews produced during this suite.
            for path in RENDERED.iterdir():
                if (
                    path.is_file()
                    and path.name not in before_rendered
                    and path.name.startswith("det_")
                ):
                    path.unlink(missing_ok=True)
            if previous_db_env is None:
                os.environ.pop("SITEGUARD_DB_PATH", None)
            else:
                os.environ["SITEGUARD_DB_PATH"] = previous_db_env
    db_hash_after = sha256(DEFAULT_DB)
    (OUT / "regression_suite.log").write_text(text, encoding="utf-8")
    status = {
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "successful": result.wasSuccessful(),
        "default_database_unchanged": previous_db_hash == db_hash_after,
        "default_database_sha256_before": previous_db_hash,
        "default_database_sha256_after": db_hash_after,
        "test_names": [
            test.id() for test, _reason in result.skipped
        ],
        "failure_details": [
            {"test": test.id(), "traceback": trace}
            for test, trace in result.failures
        ],
        "error_details": [
            {"test": test.id(), "traceback": trace}
            for test, trace in result.errors
        ],
    }
    md = [
        "# Regression test results",
        "",
        f"- Tests run: **{status['tests_run']}**",
        f"- Passed: **{status['tests_run'] - status['failures'] - status['errors'] - status['skipped']}**",
        f"- Failures: **{status['failures']}**",
        f"- Errors: **{status['errors']}**",
        f"- Skipped: **{status['skipped']}**",
        f"- Overall: **{'PASS' if status['successful'] else 'FAIL'}**",
        f"- Production database unchanged: **{status['default_database_unchanged']}**",
        "",
        "The existing `tests/test_siteguard.py` and `tests/test_rag_assistant.py` suites were discovered and run. Test database operations used an isolated temporary SQLite file via `SITEGUARD_DB_PATH`; only newly generated `static/rendered/det_*.jpg` previews were cleaned up.",
        "",
        "## Coverage observed",
        "",
        "- PPE label mapping, region geometry, duplicate suppression, and worker association.",
        "- Present/Missing/Unconfirmed state construction and safety assessment behavior.",
        "- Flask upload/validation, result rendering, API routes, storage persistence/settings, dashboard/history pages, and assistant routes.",
        "- Real CPU inference sample tests when model files are present.",
        "",
        "Full test output is in [`regression_suite.log`](regression_suite.log).",
        "",
    ]
    if status["skipped"]:
        md += ["## Skipped tests", ""] + [f"- `{name}`" for name in status["test_names"]] + [""]
    if status["failures"] or status["errors"]:
        md += ["## Failures and errors", ""]
        md += [f"- `{item['test']}`" for item in status["failure_details"] + status["error_details"]]
        md.append("")
    (OUT / "regression_results.md").write_text("\n".join(md), encoding="utf-8")
    (OUT / "regression_results.json").write_text(
        __import__("json").dumps(status, indent=2), encoding="utf-8")
    print(__import__("json").dumps(status, indent=2))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
