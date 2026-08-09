"""
Generate task directories from the MigrationBench dataset.

Nothing per-repository is written by hand. Each task is rendered from
`templates/` plus one dataset row, so a change to the environment or the verifier
is made once and reapplied to every task by regenerating.

The output is Harbor's on-disk format, which is also our native input format --
so a task directory is both what the driver reads and a publishable bundle. There
is no export step.

Usage::

    python script/generate_tasks.py                       # all 300
    python script/generate_tasks.py --repos 284885166/spring-boot-hashids
    python script/generate_tasks.py --limit 10 --force
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

logger = logging.getLogger(__name__)

DEFAULT_DATASET = "AmazonScience/migration-bench-java-selected"
MIGRATIONBENCH_COMMIT = "d705e9b1a8b6fb24212248a373dfd3fbae614d07"

#: Below these, a repository gets the baseline allocation; above, the larger one.
#: Table 1 reports selected averaging 22,397 LOC and 6.6 modules, with a long
#: tail (max 1.7M LOC, 192 modules), so a flat allocation would either starve the
#: tail or waste capacity on the median.
LARGE_LOC = 100_000
LARGE_MODULES = 10


def _resources(row: Dict[str, Any]) -> Dict[str, Any]:
    """Scale container resources and timeouts to repository size."""
    loc = int(row.get("num_loc") or 0)
    modules = int(row.get("num_pom_xml") or 1)
    large = loc > LARGE_LOC or modules > LARGE_MODULES

    return {
        "cpus": 8 if large else 4,
        "memory_mb": 16384 if large else 8192,
        "agent_timeout": 7200.0 if large else 3600.0,
        "verifier_timeout": 3600.0 if large else 1800.0,
        "build_timeout": 4800.0 if large else 2400.0,
    }


def _render(template: str, values: Dict[str, str]) -> str:
    """Substitute __PLACEHOLDER__ tokens. Plain replacement, no template engine."""
    out = template
    for key, value in values.items():
        out = out.replace(f"__{key}__", str(value))
    return out


def generate_task(
    row: Dict[str, Any],
    tasks_dir: Path,
    templates: Path,
    *,
    base_image: str,
    dataset: str,
    tier: str,
    force: bool,
) -> Optional[Path]:
    """
    Render one task directory.

    Args:
        row: A dataset row (repo, base_commit, size metrics)
        tasks_dir: Where task directories are written
        templates: Template directory
        base_image: Pinned MigrationBench image the environment builds on
        dataset: Dataset name, recorded in metadata for provenance
        tier: "minimal" or "maximal" -- which criteria the verifier asserts
        force: Overwrite an existing directory

    Returns:
        The task directory, or None if it already existed and force was False
    """
    repo_id = row["repo"]
    slug = repo_id.replace("/", "__")
    task_dir = tasks_dir / slug

    if task_dir.exists():
        if not force:
            return None
        # A solution harvested from a run is expensive to recreate, so keep it.
        solution = task_dir / "solution" / "solve.sh"
        keep = solution.read_text() if solution.is_file() else None
        shutil.rmtree(task_dir)
    else:
        keep = None

    res = _resources(row)
    values = {
        "SLUG": slug,
        "REPO_ID": repo_id,
        "GITHUB_URL": f"https://github.com/{repo_id}",
        "REPO_URL": f"https://github.com/{repo_id}.git",
        "BASE_COMMIT": row.get("base_commit", ""),
        "BASE_IMAGE": base_image,
        "DATASET": dataset,
        "TIER": tier,
        "NUM_MODULES": row.get("num_pom_xml") or 1,
        "NUM_JAVA_FILES": row.get("num_java_files") or 0,
        "NUM_LOC": row.get("num_loc") or 0,
        "NUM_TEST_CASES": row.get("num_test_cases") or 0,
        "LICENSE": row.get("license") or "",
        "CPUS": res["cpus"],
        "MEMORY_MB": res["memory_mb"],
        "AGENT_TIMEOUT": res["agent_timeout"],
        "VERIFIER_TIMEOUT": res["verifier_timeout"],
        "BUILD_TIMEOUT": res["build_timeout"],
    }

    (task_dir / "environment").mkdir(parents=True, exist_ok=True)
    (task_dir / "tests").mkdir(parents=True, exist_ok=True)
    (task_dir / "solution").mkdir(parents=True, exist_ok=True)

    (task_dir / "task.toml").write_text(
        _render((templates / "task.toml.tmpl").read_text(), values)
    )
    (task_dir / "environment" / "Dockerfile").write_text(
        _render((templates / "environment" / "Dockerfile.tmpl").read_text(), values)
    )
    # Constant across every repository: the paper's prompt and the verifier.
    # The rubric restates what the gates enforce, so it is generated rather than
    # copied: a hand-maintained copy drifts from the code the first time a gate
    # changes, and a rubric that misstates the gates is worse than none.
    (task_dir / "rubric.md").write_text(
        _render((templates / "rubric.md.tmpl").read_text(), values)
    )
    # Generated for the same reason as rubric.md, and from the same source: the
    # machine-readable form the reference corpus ships at tests/rubrics.json.
    (task_dir / "rubric.json").write_text(
        _render((templates / "rubric.json.tmpl").read_text(), values)
    )
    (task_dir / "tests" / "test_weights.json").write_text(
        _render((templates / "tests" / "test_weights.json.tmpl").read_text(), values)
    )
    shutil.copy(templates / "instruction.md", task_dir / "instruction.md")
    shutil.copy(templates / "tests" / "test.sh", task_dir / "tests" / "test.sh")
    shutil.copy(templates / "tests" / "test_outputs.py", task_dir / "tests" / "test_outputs.py")
    (task_dir / "tests" / "test.sh").chmod(0o755)

    solve = task_dir / "solution" / "solve.sh"
    if keep is not None:
        solve.write_text(keep)
    else:
        # A real oracle is harvested from a scoring run by
        # script/harvest_solutions.py -- never authored by hand.
        solve.write_text(
            "#!/bin/bash\n"
            "# PLACEHOLDER. Populate with script/harvest_solutions.py once a run\n"
            "# has scored on this repository; the verified patch becomes the oracle.\n"
            "echo 'no solution harvested yet' >&2\n"
            "exit 1\n"
        )
    solve.chmod(0o755)

    return task_dir


def validate(task_dir: Path) -> bool:
    """
    Check a bundle against Harbor's real schema, if Harbor is installed.

    Worth doing: it is what caught a `verifier.collect` type error that would
    otherwise have surfaced only at run time.
    """
    try:
        from harbor.models.task.task import Task as HarborTask
    except ImportError:
        return True
    return bool(HarborTask.is_valid_dir(task_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate task directories from the dataset")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--tasks-dir", default=str(REPO_ROOT / "tasks"))
    parser.add_argument("--templates", default=str(REPO_ROOT / "templates"))
    parser.add_argument("--repos", nargs="+", default=None,
                        help="Only these repositories. Defaults to the whole dataset.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tier", default="minimal", choices=["minimal", "maximal"])
    parser.add_argument("--migrationbench-commit", default=MIGRATIONBENCH_COMMIT)
    parser.add_argument("--force", action="store_true",
                        help="Regenerate directories that already exist (harvested "
                             "solutions are preserved).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from datasets import load_dataset

    logger.info("Loading %s", args.dataset)
    rows: List[Dict[str, Any]] = list(load_dataset(args.dataset, split="test"))

    if args.repos:
        wanted = set(args.repos)
        rows = [r for r in rows if r["repo"] in wanted]
        missing = wanted - {r["repo"] for r in rows}
        if missing:
            logger.error("Not in %s: %s", args.dataset, sorted(missing))
            sys.exit(1)
    if args.limit:
        rows = rows[: args.limit]

    tasks_dir = Path(args.tasks_dir)
    templates = Path(args.templates)
    base_image = f"migration-bench:{args.migrationbench_commit[:7]}"

    written, skipped, invalid = 0, 0, []
    for row in rows:
        task_dir = generate_task(
            row, tasks_dir, templates,
            base_image=base_image, dataset=args.dataset,
            tier=args.tier, force=args.force,
        )
        if task_dir is None:
            skipped += 1
            continue
        written += 1
        if not validate(task_dir):
            invalid.append(task_dir.name)

    logger.info("Generated %d, skipped %d existing", written, skipped)
    if invalid:
        logger.error("Failed Harbor schema validation: %s", invalid[:5])
        sys.exit(1)
    logger.info("All generated tasks validate against the Harbor schema")


if __name__ == "__main__":
    main()
