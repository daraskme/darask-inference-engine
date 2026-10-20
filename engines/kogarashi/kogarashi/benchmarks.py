"""Isolated experiment evidence: a failed run cannot inherit an old success.

Design references (independent implementation): DeepInfer PRs #115 and #102.
See research/reuse-decisions.json for revisions, applicability, and limitations.
"""
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import uuid


def write_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class BenchmarkRun:
    """One writer, one new directory. Only complete measured coverage may pass."""

    def __init__(self, root: Path, metadata: dict, expected_cases):
        self.expected = set(expected_cases)
        if not self.expected:
            raise ValueError("At least one expected measurement is required")
        self.completed = set()
        self.id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:12]
        self.path = root / self.id
        self.path.mkdir(parents=True, exist_ok=False)
        self.closed = False
        write_json(self.path / "metadata.json", {
            "run_id": self.id, "expected_cases": sorted(self.expected), "settings": metadata,
        })
        self._state("running")

    def _state(self, status, **details):
        write_json(self.path / "status.json", {
            "run_id": self.id, "status": status,
            "coverage_complete": status == "complete",
            "completed_cases": sorted(self.completed),
            "missing_cases": sorted(self.expected - self.completed), **details,
        })

    def record(self, case: str, measurement: dict):
        if self.closed or case not in self.expected or case in self.completed:
            raise ValueError(f"Unexpected, duplicate, or closed measurement: {case}")
        row = json.dumps({"run_id": self.id, "case": case, "measurement": measurement},
                         ensure_ascii=False, allow_nan=False)
        with (self.path / "measurements.jsonl").open("a") as output:
            output.write(row + "\n")
            output.flush()
        self.completed.add(case)
        self._state("running")

    def finish(self, summary: dict):
        if self.closed or self.completed != self.expected:
            raise ValueError("Cannot finish a closed or incomplete benchmark")
        write_json(self.path / "summary.json", {**summary, "run_id": self.id})
        self._state("complete")
        self.closed = True

    def fail(self, error: BaseException):
        if not self.closed:
            self._state("failed", error_type=type(error).__name__, error=str(error))
            self.closed = True


def draft_acceptance(accepted: int, proposed: int):
    """Measured ratio only; no inferred tokens/s from an acceptance estimate."""
    if type(accepted) is not int or type(proposed) is not int:
        raise ValueError("Draft counts must be integers")
    if proposed < 0 or not 0 <= accepted <= proposed:
        raise ValueError("Require 0 <= accepted <= proposed")
    return accepted / proposed if proposed else None


def observed_rate(tokens: int, seconds: float):
    if type(tokens) is not int or tokens < 0 or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Require a nonnegative token count and finite positive elapsed time")
    return tokens / seconds


def compare_greedy(records):
    """Compare full token IDs, including repeatability and active draft coverage."""
    def tokens(response):
        ids = response.get("tokens") or [p["id"] for p in response.get("completion_probabilities", [])]
        if not ids or any(type(token) is not int for token in ids):
            raise ValueError("Greedy comparison requires nonempty generated token IDs")
        return ids

    baselines = {}
    for row in records:
        if row["mode"] == "none":
            baselines.setdefault(row["prompt"], tokens(row["response"]))
    comparisons = []
    for row in records:
        name = row["prompt"]
        if name not in baselines:
            raise ValueError(f"Missing vanilla reference for {name}")
        reference, actual = baselines[name], tokens(row["response"])
        first = next((i for i, (a, b) in enumerate(zip(reference, actual)) if a != b), None)
        if first is None and len(reference) != len(actual):
            first = min(len(reference), len(actual))
        comparisons.append({"prompt": name, "mode": row["mode"], "trial": row["trial"],
                            "match": first is None, "first_mismatch": first,
                            "reference_tokens": len(reference), "actual_tokens": len(actual)})
    mtp = [row for row in records if row["mode"] == "draft-mtp"]
    coverage = set(baselines) == {row["prompt"] for row in mtp}
    active = bool(mtp) and all(row["response"].get("timings", {}).get("draft_n_accepted", 0) > 0 for row in mtp)
    return {"passed": bool(comparisons) and coverage and active and all(row["match"] for row in comparisons),
            "prompt_coverage_complete": coverage, "drafts_accepted_in_every_mtp_case": active,
            "comparisons": comparisons}
