"""
Migrate legacy job output into the delivery bundle layout.

The driver writes bundles directly now (`harness.utils.run`). This converts trees
written by the previous layout -- `jobs/<job_name>/<slug>__<random id>/`, keyed by
a name a human invented and a token nothing can order -- into the same shape::

    <task-id>/
        task.toml · instruction.md · rubric.md
        environment/ · solution/ · tests/
        trajectories/<model>/<agent>/run_N/
            agent/ · artifacts/ · verifier/ · config.json · result.json
        trajectories/<model>/pass_summary.json

Task inputs and every run against them live in one directory, so a bundle is
shippable on its own. See NOMENCLATURE.md.

`run_N` is assigned by `started_at` across every job directory given, so the
attempt index reflects the real order in which the attempts happened rather than
which ad-hoc job folder each landed in.

Read-only with respect to its input. Once `jobs/` has been migrated this script
has no further use.

Usage::

    python script/export_bundle.py --out bundles                 # every job
    python script/export_bundle.py --out bundles jobs/run1 --force
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from harness.utils.run import (  # noqa: E402
    BUNDLE_TASK_DIRS as TASK_DIRS,
    BUNDLE_TASK_FILES as TASK_FILES,
    pass_summary,
    write_truth,
)

logger = logging.getLogger(__name__)

#: Copied out of a trial directory. `lock.json` is a runtime claim and `trial.log`
#: is driver chatter -- neither describes the attempt, so neither ships.
TRIAL_FILES = ("config.json", "result.json")
TRIAL_DIRS = ("agent", "artifacts", "verifier")

#: Stripped from the agent name to get the scaffold variant: `javamigration-rag`
#: is the eval key, `rag` is the condition the path records.
AGENT_PREFIX = "javamigration-"


def _iter_legacy_trials(job_dir: Path) -> Iterator[Dict[str, Any]]:
    """
    Yield each trial's result.json from the old flat job layout.

    `harness.utils.run.iter_trials` walks the bundle layout, which is the point of
    this script -- so the legacy shape needs its own reader rather than a shared
    one that has to understand both.
    """
    if not job_dir.is_dir():
        return
    for trial_dir in sorted(job_dir.iterdir()):
        result_path = trial_dir / "result.json"
        if not trial_dir.is_dir() or not result_path.is_file():
            continue
        try:
            data = json.loads(result_path.read_text())
        except (OSError, ValueError) as exc:
            logger.warning("Unreadable trial result in %s: %s", trial_dir, exc)
            continue
        data["_dir"] = str(trial_dir)
        yield data


def _identity(trial: Dict[str, Any]) -> Optional[Tuple[str, str, str]]:
    """
    The three dimensions that name a trial in a bundle: task, model, agent.

    Returns None when the trial cannot be placed -- a trial with no task path has
    nothing to be a run *of*, and guessing would file it under the wrong task.
    """
    task_path = ((trial.get("task_id") or {}).get("path")
                 or ((trial.get("config") or {}).get("task") or {}).get("path"))
    if not task_path:
        logger.warning("Trial %s has no task path; skipping", trial.get("_dir"))
        return None

    agent = (trial.get("config") or {}).get("agent") or {}
    model = strands_agent.get("model_name") or "unknown-model"
    name = strands_agent.get("name") or "unknown-agent"
    variant = name[len(AGENT_PREFIX):] if name.startswith(AGENT_PREFIX) else name
    return Path(task_path).name, model, variant


def collect(job_dirs: List[Path]) -> Dict[Tuple[str, str, str], List[Dict[str, Any]]]:
    """
    Group every trial in the given jobs by (task, model, agent), oldest first.

    Ordering is by `started_at` so that `run_N` is the attempt index a reader
    expects: run_1 is the first attempt that model made on that task, regardless
    of which job directory it happened to land in.
    """
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for job_dir in job_dirs:
        for trial in _iter_legacy_trials(job_dir):
            key = _identity(trial)
            if key is not None:
                grouped.setdefault(key, []).append(trial)

    for trials in grouped.values():
        trials.sort(key=lambda t: t.get("started_at") or "")
    return grouped


def _copy_task(slug: str, tasks_dir: Path, dest: Path) -> bool:
    """Copy a task's inputs into the bundle. False if the task is not on disk."""
    src = tasks_dir / slug
    if not src.is_dir():
        logger.warning("No task directory for %s; bundling trajectories only", slug)
        return False

    dest.mkdir(parents=True, exist_ok=True)
    for name in TASK_FILES:
        if (src / name).is_file():
            shutil.copy2(src / name, dest / name)
    for name in TASK_DIRS:
        if (src / name).is_dir():
            shutil.copytree(src / name, dest / name, dirs_exist_ok=True)
    write_truth(dest)
    return True


def _copy_trial(trial: Dict[str, Any], dest: Path) -> None:
    """Copy one trial's shippable contents into `run_N/`."""
    src = Path(trial["_dir"])
    dest.mkdir(parents=True, exist_ok=True)
    for name in TRIAL_FILES:
        if (src / name).is_file():
            shutil.copy2(src / name, dest / name)
    for name in TRIAL_DIRS:
        if (src / name).is_dir():
            shutil.copytree(src / name, dest / name, dirs_exist_ok=True)


def export(job_dirs: List[Path], out_dir: Path, tasks_dir: Path,
           force: bool) -> Dict[str, Any]:
    """Write every job's trials out as bundles. Returns a summary for the caller."""
    grouped = collect(job_dirs)
    if not grouped:
        return {"tasks": 0, "trials": 0, "bundles": []}

    # Group by task first so a task's inputs are copied once, not once per model.
    by_task: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]] = {}
    for (slug, model, agent), trials in grouped.items():
        by_task.setdefault(slug, {}).setdefault(model, {})[agent] = trials

    written: List[str] = []
    total_trials = 0

    for slug, by_model in sorted(by_task.items()):
        dest = out_dir / slug
        if dest.exists():
            if not force:
                logger.warning("%s exists; pass --force to replace it", dest)
                continue
            shutil.rmtree(dest)

        _copy_task(slug, tasks_dir, dest)

        for model, by_agent in sorted(by_model.items()):
            model_dir = dest / "trajectories" / model
            for agent, trials in sorted(by_agent.items()):
                for index, trial in enumerate(trials, start=1):
                    _copy_trial(trial, model_dir / agent / f"run_{index}")
                    total_trials += 1

            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "pass_summary.json").write_text(
                json.dumps(pass_summary(slug, model, model_dir), indent=2) + "\n")

        written.append(slug)
        logger.info("Bundled %s (%d model(s))", slug, len(by_model))

    return {"tasks": len(written), "trials": total_trials, "bundles": written}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate legacy job output into delivery bundles.")
    parser.add_argument("job_dirs", nargs="*", type=Path,
                        help="Job directories. Default: every directory under jobs/.")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "bundles",
                        help="Where bundles are written (default: bundles/).")
    parser.add_argument("--tasks-dir", type=Path, default=REPO_ROOT / "tasks",
                        help="Where task inputs are read from (default: tasks/).")
    parser.add_argument("--force", action="store_true",
                        help="Replace an existing bundle instead of skipping it.")
    parser.add_argument("--json", action="store_true", help="Emit the summary as JSON.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    job_dirs = args.job_dirs or sorted(
        d for d in (REPO_ROOT / "jobs").iterdir() if d.is_dir()
    ) if (REPO_ROOT / "jobs").is_dir() else []
    if not job_dirs:
        parser.error("No job directories found. Pass them explicitly.")

    summary = export(job_dirs, args.out, args.tasks_dir, args.force)

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"{summary['tasks']} task(s), {summary['trials']} trial(s) -> {args.out}")
        for slug in summary["bundles"]:
            print(f"  {slug}")


if __name__ == "__main__":
    main()
