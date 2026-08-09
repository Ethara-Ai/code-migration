"""
Task definitions in Harbor's on-disk format.

Harbor is used here as a *structure* only -- its task model parses and validates
a bundle, and `harbor task init` scaffolds one. Nothing from Harbor executes: the
agent comes from JavaMigration, the grading from MigrationBench, and the driver
is `eval/run_batch.py`.

Adopting Harbor's layout as the native format means a task directory is
simultaneously our input and a publishable bundle, so there is no export step and
no second artifact to drift.

A task directory::

    tasks/<owner>__<repo>/
        task.toml                 metadata, resources, timeouts
        instruction.md            the prompt (the paper's Section 6.2)
        environment/Dockerfile    base image + clone at base_commit + hardening
        tests/test.sh             writes /logs/verifier/reward.txt
        tests/test_outputs.py     calls MigrationBench run_eval
        solution/solve.sh         harvested from a scoring run, never hand-written
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Task:
    """One repository's migration task, read from a Harbor-format directory."""

    path: Path
    repo_id: str
    base_commit: str
    github_url: str
    instruction: str
    task_name: str
    tier: str
    cpus: Optional[int]
    memory_mb: Optional[int]
    agent_timeout_sec: float
    verifier_timeout_sec: float

    @property
    def slug(self) -> str:
        """Filesystem-safe identifier: `owner__repo`."""
        return self.repo_id.replace("/", "__")

    @property
    def environment_dir(self) -> Path:
        return self.path / "environment"

    @property
    def solution_path(self) -> Path:
        return self.path / "solution" / "solve.sh"

    @property
    def memory_arg(self) -> str:
        """Docker `--memory` string."""
        return f"{self.memory_mb}m" if self.memory_mb else "8g"

    @property
    def cpus_arg(self) -> str:
        return str(self.cpus or 4)


def _load_with_harbor(task_dir: Path) -> "Task":
    """
    Parse and validate via Harbor's own task model.

    Preferred because it validates against the real schema -- which is the whole
    reason to adopt the format. Hand-rolling a parser would reintroduce exactly
    the drift the format is meant to prevent.
    """
    from harbor.models.task.task import Task as HarborTask

    ht = HarborTask(task_dir)
    meta = ht.config.metadata or {}
    repo_id = meta.get("repo")
    if not repo_id:
        raise ValueError(f"{task_dir}/task.toml has no [metadata] repo")

    return Task(
        path=task_dir,
        repo_id=repo_id,
        base_commit=meta.get("base_commit", ""),
        github_url=meta.get("github_url") or f"https://github.com/{repo_id}",
        instruction=ht.instruction,
        task_name=(ht.config.task.name if ht.config.task else task_dir.name),
        tier=meta.get("tier", "minimal"),
        cpus=ht.config.environment.cpus,
        memory_mb=ht.config.environment.memory_mb,
        agent_timeout_sec=ht.config.agent.timeout_sec,
        verifier_timeout_sec=ht.config.verifier.timeout_sec,
    )


def _load_with_tomllib(task_dir: Path) -> "Task":
    """Fallback when Harbor is not installed. No schema validation."""
    import tomllib

    with (task_dir / "task.toml").open("rb") as fh:
        cfg = tomllib.load(fh)

    meta = cfg.get("metadata", {})
    repo_id = meta.get("repo")
    if not repo_id:
        raise ValueError(f"{task_dir}/task.toml has no [metadata] repo")

    env = cfg.get("environment", {})
    return Task(
        path=task_dir,
        repo_id=repo_id,
        base_commit=meta.get("base_commit", ""),
        github_url=meta.get("github_url") or f"https://github.com/{repo_id}",
        instruction=(task_dir / "instruction.md").read_text(encoding="utf-8"),
        task_name=cfg.get("task", {}).get("name", task_dir.name),
        tier=meta.get("tier", "minimal"),
        cpus=env.get("cpus"),
        memory_mb=env.get("memory_mb"),
        agent_timeout_sec=cfg.get("agent", {}).get("timeout_sec", 3600.0),
        verifier_timeout_sec=cfg.get("verifier", {}).get("timeout_sec", 1800.0),
    )


def load_task(task_dir: Path | str) -> Task:
    """
    Load one task directory, validating with Harbor when it is available.

    Args:
        task_dir: Directory containing task.toml

    Returns:
        The parsed task

    Raises:
        FileNotFoundError: If the directory has no task.toml
    """
    task_dir = Path(task_dir)
    if not (task_dir / "task.toml").is_file():
        raise FileNotFoundError(f"No task.toml in {task_dir}")

    try:
        return _load_with_harbor(task_dir)
    except ImportError:
        logger.debug("harbor not installed; parsing %s without validation", task_dir)
        return _load_with_tomllib(task_dir)


def iter_tasks(tasks_dir: Path | str) -> Iterator[Task]:
    """Yield every valid task directory under tasks_dir, in sorted order."""
    tasks_dir = Path(tasks_dir)
    for child in sorted(tasks_dir.iterdir()):
        if (child / "task.toml").is_file():
            yield load_task(child)


def load_tasks(
    tasks_dir: Path | str,
    repo_ids: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> List[Task]:
    """
    Select tasks to run.

    Args:
        tasks_dir: Directory of task directories
        repo_ids: Restrict to these repositories (accepts `owner/repo` or `owner__repo`)
        limit: Take at most this many, after filtering

    Returns:
        Selected tasks
    """
    tasks = list(iter_tasks(tasks_dir))

    if repo_ids:
        wanted = {r.replace("/", "__") for r in repo_ids}
        tasks = [t for t in tasks if t.slug in wanted]
        missing = wanted - {t.slug for t in tasks}
        if missing:
            raise ValueError(f"No task directory for: {sorted(missing)}")

    return tasks[:limit] if limit else tasks
