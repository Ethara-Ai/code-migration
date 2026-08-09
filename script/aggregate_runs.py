"""
Aggregate a bundle tree into the paper's efficacy metrics.

Reads the trial result files rather than any in-memory state, so it works against a
run that is still going, or one that was interrupted -- which is the reason state
lives in the filesystem.

    eta_minimal = # repositories passing {r1,r2,r3,r4} / # total          (eq. 1)
    eta_maximal = # repositories passing {r1,r2,r3,r4,r5} / # total       (eq. 2)

A repository counts as passing if *any* of its attempts passed, which is pass@k
for k attempts. With one attempt each, that is the paper's pass@1.

Usage::

    python script/aggregate_runs.py
    python script/aggregate_runs.py --root bundles --json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from harness.utils.run import iter_trials  # noqa: E402

logger = logging.getLogger(__name__)


def collect(root: Path) -> Dict[str, Any]:
    """
    Fold a bundle tree's trials into per-repository and overall numbers.

    Reads the trial result files rather than any in-memory state, so it works on
    a job that is still running or was interrupted -- which is the reason state
    lives in the filesystem.
    """
    per_repo: Dict[str, Dict[str, Any]] = {}

    for trial in iter_trials(root):
        slug = Path(trial["task_id"]["path"]).name
        rewards = (trial.get("verifier_result") or {}).get("rewards") or {}

        entry = per_repo.setdefault(
            slug,
            {"repo": slug.replace("__", "/", 1), "attempts": 0, "minimal": False,
             "maximal": False, "completed": False, "reward": 0.0, "errors": [],
             "graded": False},
        )
        entry["attempts"] += 1
        # A repository counts towards eta only once something actually graded it.
        # An attempt the toolchain could not assess is not a failed migration, and
        # putting it in the denominator understates every rate computed here.
        entry["graded"] = entry["graded"] or rewards.get("reward") is not None
        entry["minimal"] = entry["minimal"] or bool(rewards.get("minimal"))
        entry["maximal"] = entry["maximal"] or bool(rewards.get("maximal"))
        entry["reward"] = max(entry["reward"], float(rewards.get("reward") or 0.0))
        entry["completed"] = entry["completed"] or (
            bool(trial.get("finished_at")) and not trial.get("exception_info")
        )
        if trial.get("exception_info"):
            entry["errors"].append(trial["exception_info"].get("exception_message", ""))

    total = len(per_repo)
    graded_entries = [e for e in per_repo.values() if e["graded"]]
    graded = len(graded_entries)
    ungraded = total - graded
    minimal = sum(1 for e in graded_entries if e["minimal"])
    maximal = sum(1 for e in graded_entries if e["maximal"])
    completed = sum(1 for e in per_repo.values() if e["completed"])
    rewards = [e["reward"] for e in graded_entries]

    return {
        "results_dir": str(root),
        "repositories": total,
        "graded": graded,
        "ungraded": ungraded,
        "completed": completed,
        "failed": total - completed,
        "minimal_success": minimal,
        "maximal_success": maximal,
        "eta_minimal_pct": round(100.0 * minimal / graded, 2) if graded else None,
        "eta_maximal_pct": round(100.0 * maximal / graded, 2) if graded else None,
        "mean_reward": round(sum(rewards) / len(rewards), 3) if rewards else None,
        "avg_turns": None,
        "per_repo": per_repo,
    }


def render(summary: Dict[str, Any], show_failures: bool) -> str:
    lines: List[str] = []
    total = summary["repositories"]
    lines.append("")
    lines.append(f"  results        {summary['results_dir']}")
    graded = summary.get("graded", total)
    lines.append(f"  repositories   {total}   completed {summary['completed']}   failed {summary['failed']}")
    if summary.get("ungraded"):
        lines.append(f"  ungraded       {summary['ungraded']}   (excluded from eta; grading did not run)")
    if graded:
        lines.append(f"  eta_minimal    {summary['eta_minimal_pct']:.2f}%   ({summary['minimal_success']}/{graded})")
        lines.append(f"  eta_maximal    {summary['eta_maximal_pct']:.2f}%   ({summary['maximal_success']}/{graded})")
    if summary.get("mean_reward") is not None:
        lines.append(f"  mean reward    {summary['mean_reward']}")

    if show_failures:
        failed = [e for e in summary["per_repo"].values() if not e["completed"]]
        if failed:
            lines.append("")
            lines.append("  failures:")
            for entry in failed[:20]:
                why = entry["errors"][-1] if entry["errors"] else "unknown"
                lines.append(f"    {entry['repo']:50} {why[:60]}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate a results tree")
    parser.add_argument("--root", "--job-dir", dest="root",
                        default=str(REPO_ROOT / "bundles"),
                        help="Bundle root to aggregate (default: bundles/).")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table")
    parser.add_argument("--failures", action="store_true", help="List failed repositories")
    parser.add_argument("--write", action="store_true",
                        help="Also write summary.json into the results directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    exp_dir = Path(args.root).expanduser().resolve()
    if not exp_dir.is_dir():
        logger.error("No such bundle root: %s", exp_dir)
        sys.exit(1)

    summary = collect(exp_dir)
    if summary["repositories"] == 0:
        logger.error("No trials under %s", exp_dir)
        sys.exit(1)

    if args.write:
        (exp_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2) if args.json else render(summary, args.failures))


if __name__ == "__main__":
    main()
