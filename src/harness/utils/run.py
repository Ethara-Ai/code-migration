"""
Run state in the delivery bundle layout, written by our own pipeline.

Harbor supplies the *format*, never the machinery: nothing here imports it, and
it is not a dependency. The agent is JavaMigration, the grading is
MigrationBench, the driver is `eval/run_batch.py`.

One directory per task, holding its inputs and every attempt against them::

    <root>/<slug>/
        task.toml · instruction.md · TRUTH.md · rubric.json · rubric.md
        environment/ · solution/ · tests/
        trajectories/<model>/<agent>/run_N/
            trial.log · config.json · lock.json · result.json
            agent/<agent_name>.txt · agent/exit-code.txt · agent/<slug>.diff
            verifier/reward.txt · verifier/ctrf.json
            verifier/test_function_outputs.json
            verifier/test.sh · test_outputs.py · test_weights.json
            artifacts/manifest.json · artifacts/*
        trajectories/<model>/pass_summary.json

Every path segment names a dimension -- task, model, agent, attempt -- so a
comparison is a directory listing rather than a crawl through `config.json`, and
`run_N` is a true attempt index that pass@k reads directly.

The atomic claim survives the move: `run_N` is created with
``mkdir(exist_ok=False)``, so the collision *is* the increment and parallel
workers cannot write the same tree. Resume stays a directory test -- a trial with
``finished_at`` set is done -- rather than bookkeeping.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
import tomllib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

#: Stripped from the agent name to get the scaffold variant the path records.
AGENT_PREFIX = "javamigration-"

#: Copied beside a task's trajectories: enough to rebuild and re-grade, and
#: nothing a run produced.
BUNDLE_TASK_FILES = ("task.toml", "instruction.md", "rubric.json", "rubric.md")

#: Copied into a run's verifier/ so the attempt carries what graded it.
VERIFIER_INPUTS = ("test.sh", "test_outputs.py", "test_weights.json")

#: Gate name -> the guardrail it backs in `tests/test_weights.json`. The corpus
#: names a guardrail after the thing it catches rather than the check that
#: catches it, so a reader of the weights file needs no knowledge of our gates.
#: The positive half of the weighted score. `_gate_report` names its CTRF
#: entries after what they report; `test_weights.json` names its checks after the
#: pytest functions in `tests/test_outputs.py`, because that file is what a
#: Harbor reader executes. The two vocabularies have to be joined somewhere, and
#: leaving them unjoined is not a silent no-op: every positive weight scores
#: zero, so a trial can report reward 1.0 and weighted_score 0.0 at once.
#: Which channel each check belongs to. Three channels, scored separately and
#: then composed, which keeps a deterministic channel and a judged channel from
#: contaminating each other's arithmetic.
#:
#: There is no settled convention for the split. Benchmarks whose tasks have no
#: executable oracle lean almost entirely on the judged channel; those with a
#: hard test outcome weight the judge at a few percent and treat it as a
#: tiebreak. These weights
#: are neither: the two executing channels carry 70% between them and the judged
#: one 30%, which makes the rubric material without letting it outvote what the
#: code actually did.
CHANNEL_OF = {
    # A -- gates. Executing code, host side, differential against the base commit.
    "test_minimal_migration_succeeds":               "gates",
    "test_maximal_migration_succeeds":               "gates",
    "test_test_methods_preserved":                   "gates",
    "test_repository_still_present":                 "gates",
    "test_agent_made_changes":                       "gates",
    "test_negative_weight_tests_disabled_or_failing": "gates",
    "test_negative_weight_coverage_dropped":         "gates",
    "test_negative_weight_assertions_weakened":      "gates",
    "test_negative_weight_deprecated_api_left":      "gates",
    # B -- unit tests written against the base commit and validated there before
    #      they are trusted. JUnit for Java, pytest for Python; same mechanism.
    "test_generated_contract_suite":                 "unit_tests",
}

#: Weight per channel. A channel that measured nothing is dropped and the rest
#: are renormalised -- otherwise a repository with no rubric could never exceed
#: 70%, which would be a statement about our coverage rather than about the
#: migration. Same rule the gates follow: absence of evidence is not evidence.
CHANNEL_WEIGHTS = {"gates": 0.35, "unit_tests": 0.35, "rubric": 0.30}


def channel_score(outcomes: Dict[str, Optional[bool]],
                  weights: Dict[str, float],
                  names) -> Optional[float]:
    """
    The weighted formula, over one channel's checks.

        score = (sum passed positive - sum |fired negative|) / sum all positive

    Positive-only denominator, which is not cosmetic: summing *all* weights
    overshoots 1 whenever a penalty exists, so a run that passes every check
    scores well under 1.0 purely because guardrails were declared.

    Returns None when the channel measured nothing at all, which is distinct
    from measuring zero.
    """
    earned = penalty = total_positive = 0.0
    measured = False
    for name in names:
        weight = weights.get(name)
        if weight is None:
            continue
        outcome = outcomes.get(name)
        if weight > 0:
            total_positive += weight
            if outcome is True:
                earned += weight
                measured = True
            elif outcome is False:
                measured = True
        elif outcome is False:
            penalty += abs(weight)
            measured = True
    if not measured or not total_positive:
        return None
    return (earned - penalty) / total_positive


POSITIVE_CHECK = {
    "minimal_migration":   "test_minimal_migration_succeeds",
    "maximal_migration":   "test_maximal_migration_succeeds",
    "tests_preserved":     "test_test_methods_preserved",
    "repository_present":  "test_repository_still_present",
    "agent_made_changes":  "test_agent_made_changes",
}

GATE_GUARDRAIL = {
    "test_execution":    "test_negative_weight_tests_disabled_or_failing",
    "coverage_delta":    "test_negative_weight_coverage_dropped",
    "mutation_delta":    "test_negative_weight_assertions_weakened",
    "deprecation_count": "test_negative_weight_deprecated_api_left",
}
BUNDLE_TASK_DIRS = ("environment", "solution", "tests")


def _now() -> str:
    """Harbor timestamps are ISO 8601, UTC, with a trailing Z."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def task_digest(task_dir: Path) -> str:
    """
    Stable digest of a task directory, for the lock file.

    Hashes relative paths and contents of every tracked file so that a task
    edited after a run is detectable -- which is the point of recording it.
    """
    digest = hashlib.sha256()
    for path in sorted(task_dir.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        digest.update(str(path.relative_to(task_dir)).encode())
        digest.update(path.read_bytes())
    return f"sha256:{digest.hexdigest()}"


@dataclass
class Phase:
    """One timed stage of a trial."""

    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    def start(self) -> "Phase":
        self.started_at = _now()
        return self

    def stop(self) -> "Phase":
        self.finished_at = _now()
        return self

    def as_dict(self) -> Optional[Dict[str, Any]]:
        if self.started_at is None:
            return None
        return {"started_at": self.started_at, "finished_at": self.finished_at}


@dataclass
class Trial:
    """One attempt at one task, written into a `run_N` directory."""

    path: Path
    trial_name: str
    task_name: str
    task_path: Path
    source: str
    job_id: str
    agent_name: str
    model_name: Optional[str] = None

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: str = field(default_factory=_now)
    finished_at: Optional[str] = None
    rewards: Dict[str, Any] = field(default_factory=dict)
    exception_info: Optional[Dict[str, Any]] = None

    environment_setup: Phase = field(default_factory=Phase)
    agent_setup: Phase = field(default_factory=Phase)
    agent_execution: Phase = field(default_factory=Phase)
    verifier: Phase = field(default_factory=Phase)

    @property
    def agent_dir(self) -> Path:
        return self.path / "agent"

    @property
    def verifier_dir(self) -> Path:
        return self.path / "verifier"

    @property
    def artifacts_dir(self) -> Path:
        return self.path / "artifacts"

    @property
    def log_path(self) -> Path:
        return self.path / "trial.log"

    def __post_init__(self) -> None:
        for sub in (self.agent_dir, self.verifier_dir, self.artifacts_dir):
            sub.mkdir(parents=True, exist_ok=True)
        self._write_lock()
        self.write()

    def write_agent_log(self, text: str, exit_code: Optional[int]) -> None:
        """Agent stdout/stderr and its exit status, as Harbor records them."""
        (self.agent_dir / f"{self.agent_name}.txt").write_text(text, errors="replace")
        (self.agent_dir / "exit-code.txt").write_text(
            "" if exit_code is None else str(exit_code)
        )

    def write_reward(self, reward: Optional[float],
                     extra: Optional[Dict[str, Any]] = None) -> None:
        """
        Record the scalar reward plus any additional named rewards.

        `reward` is the headline scalar every Harbor consumer reads. Minimal and
        maximal are carried alongside it because they are two booleans and the
        rewards field is a mapping -- collapsing them into one number would throw
        away which tier was reached.

        `None` means the trial was never graded, which is not the same as scoring
        zero: a migration that was never assessed says nothing about the agent.
        `reward.txt` still receives a number, because it is a Harbor file contract
        and an external reader must not meet a new type; the distinction lives in
        `rewards["reward"]`, which `Bundle.finish` already skips when it is None, and
        in the trial's exception info.
        """
        self.rewards = {"reward": None if reward is None else float(reward)}
        if extra:
            self.rewards.update(extra)
        self._write_reward_text()

    def _write_reward_text(self) -> None:
        """
        Write the scalar an external reader consumes.

        The weighted score when there is one, the binary reward otherwise. The
        reference corpus puts its graded scalar here, and `minimal` / `maximal`
        stay booleans in `rewards`, so eta and the MigrationBench comparison read
        those and are unaffected by what this file holds.

        `None` -- never graded -- still writes a number, because this is a file
        contract and an external reader must not meet a new type. The distinction
        lives in `rewards["reward"]`.
        """
        value = self.rewards.get("weighted_score")
        if value is None:
            value = self.rewards.get("reward")
        (self.verifier_dir / "reward.txt").write_text(
            "0.0" if value is None else str(value))

    def write_ctrf(self, checks: List[Dict[str, Any]], tool: str = "migration-bench") -> None:
        """
        Emit the gate outcomes as a CTRF report.

        CTRF is the format Harbor's own pytest verifier writes, so a consumer that
        can read a Harbor job can read our gates without special-casing.
        """
        now = time.time()
        passed = sum(1 for c in checks if c["status"] == "passed")
        failed = sum(1 for c in checks if c["status"] == "failed")
        skipped = sum(1 for c in checks if c["status"] == "skipped")
        report = {
            "results": {
                "tool": {"name": tool},
                "summary": {
                    "tests": len(checks),
                    "passed": passed,
                    "failed": failed,
                    "skipped": skipped,
                    "pending": 0,
                    "other": 0,
                    "start": now,
                    "stop": now,
                },
                "tests": checks,
            }
        }
        (self.verifier_dir / "ctrf.json").write_text(json.dumps(report, indent=2))

    def write_test_outputs(self, outcomes: Dict[str, Optional[bool]],
                           weights: Dict[str, float],
                           rubric_score: Optional[float] = None) -> Optional[float]:
        """
        Score each channel separately, then compose them.

        Three channels, because they are three different kinds of evidence and
        averaging them as one pool would let a judged verdict cancel an executed
        one::

            A  gates       0.35   executing code, differential against the base
            B  unit tests  0.35   generated against the base, trusted only if
                                  they pass there first
            C  rubric      0.30   judged

        Within a channel the formula uses a positive-only denominator::

            score = (sum passed positive - sum |fired negative|) / sum all positive

        Between channels the weights are renormalised over whichever channels
        produced a signal. A repository with no rubric is not a repository that
        scored zero on its rubric; it is one where that evidence does not exist,
        and capping it at 0.70 would report our coverage rather than the
        migration's quality.

        `minimal` and `maximal` stay the headline elsewhere, because they are
        what MigrationBench reports and what the published numbers compare
        against. This is the graded signal beside them.
        """
        rows = []
        for name, weight in weights.items():
            if name.startswith("_"):
                continue
            outcome = outcomes.get(name)
            rows.append({"name": name, "weight": weight, "outcome": outcome,
                         "channel": CHANNEL_OF.get(name, "gates"),
                         "fired": weight < 0 and outcome is False})

        per_channel: Dict[str, Optional[float]] = {}
        for channel in CHANNEL_WEIGHTS:
            if channel == "rubric":
                per_channel[channel] = rubric_score
                continue
            names = [r["name"] for r in rows if r["channel"] == channel]
            per_channel[channel] = channel_score(outcomes, weights, names)

        measured = {c: v for c, v in per_channel.items() if v is not None}
        if measured:
            live = sum(CHANNEL_WEIGHTS[c] for c in measured)
            score = sum(CHANNEL_WEIGHTS[c] * v for c, v in measured.items()) / live
        else:
            score = None

        # criteria_total == passed + failed + abstained. Abstained is a
        # first-class outcome, never folded into failed.
        graded = [r for r in rows if r["weight"] > 0]
        n_pass = sum(1 for r in graded if r["outcome"] is True)
        n_fail = sum(1 for r in graded if r["outcome"] is False)

        (self.verifier_dir / "test_function_outputs.json").write_text(json.dumps({
            "outputs": {r["name"]: r["outcome"] for r in rows},
            "checks": rows,
            "channels": {
                c: {"weight": CHANNEL_WEIGHTS[c], "score": per_channel[c],
                    "counted": c in measured}
                for c in CHANNEL_WEIGHTS
            },
            "criteria_total": len(graded),
            "criteria_passed": n_pass,
            "criteria_failed": n_fail,
            "criteria_abstained": len(graded) - n_pass - n_fail,
            "weighted_score": None if score is None else round(score, 4),
            "weights_percentage": None if score is None else round(score * 100.0, 2),
        }, indent=2) + "\n")
        return None if score is None else round(score, 4)

    def stage_verifier_inputs(self, task_path: Path) -> None:
        """
        Copy what graded this attempt in beside what it produced.

        `test.sh`, `test_outputs.py` and `test_weights.json` are the inputs that
        decided the score, so a run that does not carry them cannot be re-scored
        or audited once the task is edited -- and nothing records which version of
        the weights produced the number. The reference corpus copies all three
        into the verifier directory for exactly that reason.
        """
        for name in VERIFIER_INPUTS:
            src = task_path / "tests" / name
            if src.is_file():
                shutil.copy2(src, self.verifier_dir / name)

    def add_artifact(self, name: str, content: str) -> Path:
        """Store an artifact (patch, trajectory) under artifacts/."""
        dest = self.artifacts_dir / name
        dest.write_text(content, errors="replace")
        return dest

    def write_manifest(self) -> None:
        manifest = [
            {
                "source": f"/out/{p.name}",
                "destination": f"artifacts/{p.name}",
                "type": "file",
                "status": "empty" if p.stat().st_size == 0 else "collected",
                "service": None,
            }
            for p in sorted(self.artifacts_dir.iterdir())
            if p.is_file() and p.name != "manifest.json"
        ]
        (self.artifacts_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    def fail(self, exc: BaseException) -> None:
        self.exception_info = {
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
        }
        (self.path / "exception.txt").write_text(f"{type(exc).__name__}: {exc}")

    def finish(self) -> None:
        self.finished_at = _now()
        self.write_manifest()
        self.write()

    def _config(self) -> Dict[str, Any]:
        return {
            "task": {"path": str(self.task_path), "source": self.source},
            "trial_name": self.trial_name,
            "trials_dir": str(self.path.parent),
            "agent": {"name": self.agent_name, "model_name": self.model_name},
            "environment": {"type": "docker"},
            "verifier": {"disable": False},
            "job_id": self.job_id,
        }

    def _write_lock(self) -> None:
        lock = {
            "schema_version": SCHEMA_VERSION,
            "task": {
                "name": self.task_path.name,
                "type": "local",
                "digest": task_digest(self.task_path),
                "source": self.source,
                "path": str(self.task_path),
            },
            "agent": {"name": self.agent_name, "model_name": self.model_name},
            "environment": {"type": "docker"},
            "verifier": {"disable": False},
        }
        (self.path / "lock.json").write_text(json.dumps(lock, indent=2))

    def write(self) -> None:
        """Persist result.json. Called on creation and again at finish."""
        result = {
            "id": self.id,
            "task_name": self.task_name,
            "trial_name": self.trial_name,
            "trial_uri": self.path.resolve().as_uri(),
            "task_id": {"path": str(self.task_path)},
            "source": self.source,
            "task_checksum": task_digest(self.task_path).removeprefix("sha256:"),
            "config": self._config(),
            "agent_info": {"name": self.agent_name, "model_info": (
                {"name": self.model_name} if self.model_name else None
            )},
            "verifier_result": {"rewards": self.rewards} if self.rewards else None,
            "exception_info": self.exception_info,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "environment_setup": self.environment_setup.as_dict(),
            "agent_setup": self.agent_setup.as_dict(),
            "agent_execution": self.agent_execution.as_dict(),
            "verifier": self.verifier.as_dict(),
        }
        (self.path / "config.json").write_text(json.dumps(self._config(), indent=2))
        tmp = self.path / "result.json.tmp"
        tmp.write_text(json.dumps(result, indent=2))
        tmp.replace(self.path / "result.json")


class Bundle:
    """
    A directory of task bundles, each holding its own trajectories.

    One directory per task, and every run against that task lives inside it, so a
    task and its evidence travel together and a bundle is shippable on its own.
    See NOMENCLATURE.md.
    """

    def __init__(self, root: Path, agent_name: str,
                 model_name: Optional[str] = None, source: str = "tasks"):
        self.path = Path(root)
        self.path.mkdir(parents=True, exist_ok=True)
        self.id = str(uuid.uuid4())
        self.agent_name = agent_name
        self.model_name = model_name
        # The scaffold variant is the condition the path records; the eval key
        # keeps the full name. `javamigration-rag` -> `rag`.
        self.agent_variant = (agent_name[len(AGENT_PREFIX):]
                              if agent_name.startswith(AGENT_PREFIX) else agent_name)
        self.source = source
        self.started_at = _now()
        self.trials: List[Trial] = []

    def log(self, message: str) -> None:
        # Outside any task directory: a driver log is not part of a bundle.
        with (self.path / "run.log").open("a", encoding="utf-8") as fh:
            fh.write(f"{_now()} {message}\n")

    def task_dir(self, slug: str) -> Path:
        return self.path / slug

    def claim_trial(self, task_name: str, task_path: Path, slug: str) -> Trial:
        """
        Reserve the next attempt directory atomically.

        `mkdir(exist_ok=False)` raises if another worker took `run_N`, so the
        collision *is* the increment: a losing worker simply tries `run_N+1`.
        That gives a true sequential attempt index without a shared counter, and
        without the random token a reader cannot order.
        """
        model = self.model_name or "unknown-model"
        # Staged on claim rather than at finish: a bundle whose run was killed
        # halfway is still a bundle, and the inputs are what make it replayable.
        self.stage_task(slug, task_path)
        base = self.task_dir(slug) / "trajectories" / model / self.agent_variant
        base.mkdir(parents=True, exist_ok=True)

        attempt = 1
        while True:
            trial_path = base / f"run_{attempt}"
            try:
                trial_path.mkdir(exist_ok=False)
            except FileExistsError:
                attempt += 1
                continue
            trial = Trial(
                path=trial_path,
                trial_name=f"{model}/{self.agent_variant}/run_{attempt}",
                task_name=task_name, task_path=task_path, source=self.source,
                job_id=self.id, agent_name=self.agent_name,
                model_name=self.model_name,
            )
            self.trials.append(trial)
            return trial

    def stage_task(self, slug: str, task_path: Path) -> None:
        """
        Copy a task's inputs beside its trajectories, so the bundle stands alone.

        Copied once per task and skipped when already present, so re-running a
        task adds an attempt without rewriting its inputs.
        """
        dest = self.task_dir(slug)
        if (dest / "task.toml").is_file():
            return
        dest.mkdir(parents=True, exist_ok=True)
        for name in BUNDLE_TASK_FILES:
            if (task_path / name).is_file():
                shutil.copy2(task_path / name, dest / name)
        for name in BUNDLE_TASK_DIRS:
            if (task_path / name).is_dir():
                shutil.copytree(task_path / name, dest / name, dirs_exist_ok=True)
        write_truth(dest)

    def finish(self) -> Dict[str, Any]:
        """
        Write a `pass_summary.json` per model touched, and return run stats.

        The summary is written from the tree rather than from this run's trials,
        so it accounts for every attempt ever recorded against that task and
        model -- including earlier runs of the driver. `run_1` means the first
        attempt, not the first attempt today.
        """
        for slug, model in {(Path(t.task_path).name, t.model_name or "unknown-model")
                            for t in self.trials}:
            model_dir = self.task_dir(slug) / "trajectories" / model
            if model_dir.is_dir():
                (model_dir / "pass_summary.json").write_text(
                    json.dumps(pass_summary(slug, model, model_dir), indent=2) + "\n")

        completed = [t for t in self.trials if t.exception_info is None and t.finished_at]
        errored = [t for t in self.trials if t.exception_info is not None]

        reward_stats: Dict[str, List[str]] = {}
        rewards: List[float] = []
        for trial in self.trials:
            value = trial.rewards.get("reward")
            if value is None:
                continue
            rewards.append(float(value))
            reward_stats.setdefault(str(float(value)), []).append(trial.trial_name)

        eval_name = f"{self.agent_name}__{self.source}"
        return {
            "id": self.id,
            "started_at": self.started_at,
            "finished_at": _now(),
            "n_total_trials": len(self.trials),
            "stats": {
                "n_completed_trials": len(completed),
                "n_errored_trials": len(errored),
                "evals": {
                    eval_name: {
                        "n_trials": len(self.trials),
                        "n_errors": len(errored),
                        "metrics": [
                            {"mean": (sum(rewards) / len(rewards)) if rewards else 0.0}
                        ],
                        "reward_stats": {"reward": reward_stats},
                    }
                },
            },
        }


def write_truth(task_dir: Path) -> Optional[Path]:
    """
    Write `TRUTH.md`: what a correct migration of this repository had to do.

    A delivery artifact, not a task one -- eight of nine sample repositories in
    the reference corpus ship it and none of the harness repositories do, because
    it answers a reader holding the results rather than an agent about to run.
    Derived from `task.toml` so it cannot drift from the metadata the grading
    used, and skipped rather than guessed at when that file is unreadable.
    """
    config_path = task_dir / "task.toml"
    if not config_path.is_file():
        return None
    try:
        config = tomllib.loads(config_path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as exc:
        logger.warning("Cannot read %s for TRUTH.md: %s", config_path, exc)
        return None

    meta = config.get("metadata") or {}
    tier = meta.get("tier", "minimal")
    dest = task_dir / "TRUTH.md"
    dest.write_text(f"""# Ground truth — {meta.get('repo', task_dir.name)}

Migrate this repository from Java 8 to Java 17.

| | |
|---|---|
| Repository | {meta.get('github_url', '')} |
| Base commit | `{meta.get('base_commit', '')}` |
| Tier required | `{tier}` |
| Size | {meta.get('num_java_files', '?')} Java files, {meta.get('num_loc', '?')} LOC, \
{meta.get('num_modules', '?')} module(s) |
| Tests at base | {meta.get('num_test_cases', '?')} |
| Licence | {meta.get('license', '')} |

## What a correct migration had to do

1. `mvn clean verify` passes under JDK 17.
2. Compiled classes report class-file major version 61.
3. No test method removed or renamed, and the test count does not fall.
4. No test disabled, skipped, or left failing.
5. Line coverage does not fall by more than 5 percentage points.
6. Assertion strength is preserved.
7. No deprecated or removed JDK API is left in place.
{"8. Every dependency is at its latest major version (`maximal` only)." if tier == "maximal" else ""}

Criteria 1 and 2 are MigrationBench's r1 and r2; the dependency criterion is r5.
The rest are asserted independently of the test suite, because the suite is
editable by the agent being graded.

A criterion that could not be measured on this repository is recorded as `null`,
never as a pass. See `rubric.md` for the criteria a gate cannot judge.

## The reference solution

`solution/solve.sh` is harvested from a scoring run, never hand-written. It is one
correct migration, not the only one: any patch that satisfies the criteria above
is correct, and the grading asserts those criteria rather than similarity to this
diff.
""")
    return dest


def _elapsed(trial: Dict[str, Any]) -> Optional[float]:
    """Wall-clock seconds for a trial, or None if either timestamp is missing."""
    started, finished = trial.get("started_at"), trial.get("finished_at")
    if not started or not finished:
        return None
    try:
        return round((datetime.fromisoformat(finished.replace("Z", "+00:00"))
                      - datetime.fromisoformat(started.replace("Z", "+00:00"))
                      ).total_seconds(), 1)
    except ValueError:
        return None


def pass_summary(slug: str, model: str, model_dir: Path) -> Dict[str, Any]:
    """
    Fold one model's attempts on one task into mean and pass@k.

    Ungraded trials carry `reward: null` -- grading did not run, so the attempt
    is evidence of nothing. They stay in `run_count` and are excluded from both
    metrics rather than counted as failures, the rule `aggregate_runs.py` applies
    to eta.

    Mean rewards consistency, pass@k rewards capability: a model scoring
    0, 0, 0, 1 has mean 0.25 and pass@4 1.0. Both are reported; neither replaces
    the other.
    """
    runs: List[Dict[str, Any]] = []
    for agent_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
        for run_dir in sorted(agent_dir.glob("run_*"),
                              key=lambda p: int(p.name.split("_")[-1])):
            result = run_dir / "result.json"
            if not result.is_file():
                continue
            try:
                data = json.loads(result.read_text())
            except (OSError, ValueError):
                continue
            rewards = (data.get("verifier_result") or {}).get("rewards") or {}
            runs.append({
                "agent": agent_dir.name,
                "run": int(run_dir.name.split("_")[-1]),
                "reward": rewards.get("reward"),
                "minimal": rewards.get("minimal"),
                "maximal": rewards.get("maximal"),
                "elapsed_sec": _elapsed(data),
            })

    graded = [r["reward"] for r in runs if isinstance(r["reward"], (int, float))]
    return {
        "task": slug,
        "model": model,
        "runs": runs,
        "run_count": len(runs),
        "graded_count": len(graded),
        "average_reward": round(sum(graded) / len(graded), 4) if graded else None,
        "pass_at_k": max(graded) if graded else None,
        "k": len(graded),
    }


def iter_trials(root: Path) -> Iterator[Dict[str, Any]]:
    """
    Yield each trial's result.json under a bundle root.

    Walks `<slug>/trajectories/<model>/<agent>/run_N/`. The depth is fixed, so a
    glob is enough and no directory below a run is descended into -- an
    `artifacts/` copy of the repository can be large.
    """
    if not root.is_dir():
        return
    for result_path in sorted(root.glob("*/trajectories/*/*/run_*/result.json")):
        try:
            data = json.loads(result_path.read_text())
        except (OSError, ValueError) as exc:
            logger.warning("Unreadable trial result in %s: %s", result_path.parent, exc)
            continue
        data["_dir"] = str(result_path.parent)
        yield data


def completed_slugs(root: Path) -> set[str]:
    """
    Task slugs that already finished under this root, for --resume.

    A trial counts as finished when `finished_at` is set and no exception was
    recorded -- so an interrupted trial is retried rather than skipped.
    """
    done: set[str] = set()
    for trial in iter_trials(root):
        if trial.get("finished_at") and not trial.get("exception_info"):
            done.add(Path(trial["task_id"]["path"]).name)
    return done
