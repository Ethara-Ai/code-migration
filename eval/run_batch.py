"""
The driver: run a migration task per container, grade it, record it.

The only entry point for producing results. Everything it calls already exists:

* **JavaMigration** supplies the agent -- `strands_agent.main --repo-path`
  runs inside the container;
* **MigrationBench** supplies the grading -- `run_eval`, invoked from inside the
  same image;
* **Harbor** supplies only the *format*, for tasks and for results alike. Nothing
  from Harbor executes and it is not a dependency.

Output is written as delivery bundles (`harness.utils.run`) -- one directory per
task, holding its inputs and every attempt against it at
`trajectories/<model>/<agent>/run_N/`. Attempts are claimed atomically, so
`--resume` is a directory test rather than bookkeeping.

Usage::

    python eval/run_batch.py --agent-type rag \\
        --repos 284885166/spring-boot-hashids
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from harness.utils import docker_utils as du            # noqa: E402
from harness.utils.airgap import Airgap                  # noqa: E402
from harness.utils.litellm import (                      # noqa: E402
    DEFAULT_MODEL_ALIAS,
    LiteLLMSidecar,
    build_model_list,
)
from harness.utils import gates as gt                       # noqa: E402
from harness.utils.grading import grade_diff                # noqa: E402
from harness.utils.judge import grade_task_rubric              # noqa: E402
from harness.utils.testgen import default_llm                  # noqa: E402
from harness.utils.run import (  # noqa: E402
    GATE_GUARDRAIL,
    POSITIVE_CHECK,
    Bundle,
    Trial,
    completed_slugs,
)
from harness.utils.tasks import Task, load_tasks         # noqa: E402

logger = logging.getLogger(__name__)

#: The MigrationBench commit every result is graded against. r5 compares against
#: a 240-entry dependency_version.json in that repo, so leaving the ref floating
#: would make maximal verdicts depend on whatever `main` said that day.
MIGRATIONBENCH_COMMIT = "d705e9b1a8b6fb24212248a373dfd3fbae614d07"
MIGRATIONBENCH_REPO = "https://github.com/amazon-science/MigrationBench.git"


def base_image_tag(commit: str = MIGRATIONBENCH_COMMIT) -> str:
    """
    Tag the base image by the commit that built it.

    A fixed `:latest` would keep resolving to a stale image after the pin moved,
    which is exactly the drift pinning exists to prevent.
    """
    return f"migration-bench:{commit[:7]}"


def build_base_image(commit: str, force: bool = False) -> str:
    """
    Build MigrationBench's evaluation image directly from the pinned ref.

    Docker accepts a git URL as build context, so this needs no local checkout --
    MigrationBench stops being a second folder to keep in sync.
    """
    tag = base_image_tag(commit)
    if du._image_exists(tag) and not force:
        logger.info("Base image %s present", tag)
        return tag

    context = f"{MIGRATIONBENCH_REPO}#{commit}"
    logger.info("Building %s from %s", tag, context)
    built = du._run(["docker", "build", "-t", tag, context], timeout=3600)
    if built.returncode != 0:
        raise RuntimeError(f"Base image build failed:\n{built.stderr[-3000:]}")
    return tag


def build_task_image(task: Task, agent_image: str, force: bool = False) -> str:
    """
    Build the task's environment, layered on the agent image.

    The clone and the history hardening happen here, at build time, where the
    network is unrestricted. By the time the agent runs, the repository is already
    in place -- so the run itself needs neither the dataset nor github, which is
    what lets the airgap allowlist exclude both.

    The task's Dockerfile defaults to MigrationBench's grading image so any other
    harness can build it agent-agnostically; we override BASE_IMAGE to get
    JavaMigration in the same layer stack.
    """
    tag = f"jma-task-{task.slug.lower()}:{agent_image.split(':')[-1]}"
    if du._image_exists(tag) and not force:
        return tag

    logger.info("Building task environment %s", tag)
    built = du._run(
        ["docker", "build", "-t", tag,
         "--build-arg", f"BASE_IMAGE={agent_image}",
         str(task.environment_dir)],
        timeout=3600,
    )
    if built.returncode != 0:
        raise RuntimeError(f"Task image build failed for {task.repo_id}:\n{built.stderr[-3000:]}")
    return tag


def run_one(
    task: Task,
    *,
    job: Bundle,
    args: argparse.Namespace,
    image: str,
    network: Optional[str],
    model_provider: str,
    model_id: Optional[str],
    model_base_url: Optional[str],
    maven_settings: Optional[Path],
) -> Trial:
    """
    Run one task in its own container and record it as one attempt.

    The trial directory is written on claim and again at finish -- including from
    the failure path -- so an interrupted batch leaves a record of what was in
    flight rather than a directory that merely looks unclaimed.
    """
    trial = job.claim_trial(task.task_name, task.path, task.slug)
    job.log(f"claimed {trial.trial_name} for {task.repo_id}")

    try:
        trial.environment_setup.start()
        task_image = build_task_image(task, image, force=args.rebuild)
        trial.environment_setup.stop()

        trial.agent_execution.start()
        result = du.run_repository(
            task.repo_id,
            image=task_image,
            agent_type=args.agent_type,
            exp_id=args.run_label,
            hf_dataset=args.hf_dataset,
            model_provider=model_provider,
            model_id=model_id,
            model_base_url=model_base_url,
            max_messages=args.max_messages,
            shell_timeout=args.shell_timeout,
            run_timeout=int(task.agent_timeout_sec),
            output_dir=trial.artifacts_dir,
            cpus=task.cpus_arg,
            memory=task.memory_arg,
            aws_dir=Path(args.aws_dir).expanduser() if args.aws_dir else None,
            network=network,
            max_attempts=args.max_attempts,
            container_prefix=args.container_prefix,
            maven_settings=maven_settings,
            repo_path="/app/repo",
            github_url=task.github_url,
            solution_dir=task.path / "solution" if args.agent_type == "oracle" else None,
            # The task format calls instruction.md "the agent task description,
            # verbatim". Parsed but never delivered until now, so editing it
            # changed nothing and every task ran the agent's built-in prompt.
            instruction_file=task.path / "instruction.md",
            # Always grade outside. MigrationBench reaches GitHub, which an
            # airgapped agent must not -- and even unairgapped, grading in the
            # agent's own container left final_eval.py and the r5 reference list
            # writable by a root agent.
            skip_eval=True,
        )
        trial.agent_execution.stop()

        collected = trial.artifacts_dir / args.run_label
        log_text = ""
        for candidate in (collected, trial.artifacts_dir):
            log_file = candidate / f"{task.slug}.log"
            if log_file.is_file():
                log_text = log_file.read_text(errors="replace")
                break
        trial.write_agent_log(log_text, result.exit_code)
        trial.log_path.write_text(log_text, errors="replace")

        diff_path = collected / f"{task.slug}.diff"
        trajectory = collected / f"{task.slug}.json"
        if diff_path.is_file():
            trial.add_artifact("strands_agent.patch", diff_path.read_text(errors="replace"))
        if trajectory.is_file():
            trial.add_artifact("trajectory.json", trajectory.read_text(errors="replace"))

        trial.verifier.start()
        graded = grade_diff(
            task.github_url,
            diff_path if diff_path.is_file() else trial.artifacts_dir / "missing.diff",
        )
        trial.verifier.stop()

        tier_passed = graded.maximal if task.tier == "maximal" else graded.minimal
        checks = _gate_report(result.succeeded, graded, diff_path)
        gate_detail = {}

        trial.verifier.start()
        base8 = f"{task_image}-base8"
        du._run(["docker", "build", "-t", base8,
                 "--build-arg", f"BASE_IMAGE={gt.BASELINE_IMAGE}",
                 str(task.environment_dir)], timeout=3600)
        baseline = gt.capture_baseline(task_image, source_image=base8,
                                       generated_tests=args.generated_tests)
        for gate in gt.run_gates(base_image=task_image, target_image=task_image,
                                 patch=diff_path, baseline=baseline,
                                 generated_tests=args.generated_tests,
                                 enforce_generated=args.enforce_generated):
            logger.info("[%s] %s: %s", task.repo_id, gate.name, gate.detail)
            checks.append({"name": gate.name, "status": gate.status,
                           "duration": 0, "message": gate.detail})
            gate_detail[gate.name] = gate.passed
            # A gate that returns no verdict cannot measure this repository and
            # is not evidence against the migration; a gate that returns False
            # is. Reward has to move, or the gate is decoration -- measured here,
            # testFailureIgnore alone passes every one of MigrationBench's five
            # criteria with the suite failing.
            if gate.passed is False:
                tier_passed = False
        trial.verifier.stop()

        # A trial the grader could not assess is not a trial the agent failed.
        # Scoring it 0.0 puts a false negative into eta and into any training
        # signal derived from it; measured here, the grading toolchain produced
        # one spurious failure in seven runs of a byte-identical patch.
        if graded.error:
            job.log(f"{trial.trial_name} not graded: {graded.error}")
            trial.write_reward(None, {"minimal": None, "maximal": None,
                                      "grading_error": graded.error, **gate_detail})
            trial.fail(RuntimeError(f"not graded: {graded.error}"))
        else:
            trial.write_reward(
                1.0 if tier_passed else 0.0,
                {"minimal": graded.minimal, "maximal": graded.maximal, **gate_detail},
            )
        # The graded signal beside the binary one: signed weights from the task's
        # tests/test_weights.json, guardrail gates subtracting when they fire.
        weights_path = task.path / "tests" / "test_weights.json"
        if weights_path.is_file():
            try:
                weights = json.loads(weights_path.read_text())
            except ValueError as exc:
                job.log(f"{trial.trial_name} unreadable test_weights.json: {exc}")
            else:
                outcomes = {c["name"]: (None if c["status"] == "skipped"
                                        else c["status"] == "passed") for c in checks}
                # Both halves have to be translated into the weights file's
                # vocabulary. Only the negative half was, so every positive
                # weight scored zero and a passing trial reported 0.0.
                outcomes.update({POSITIVE_CHECK[k]: v for k, v in outcomes.items()
                                 if k in POSITIVE_CHECK})
                outcomes.update({GATE_GUARDRAIL[k]: v for k, v in gate_detail.items()
                                 if k in GATE_GUARDRAIL})

                # The judged channel. A rubric criterion is decided by reading
                # the patch, so this needs no container -- but it does need a
                # model, and grading must not depend on one being reachable. A
                # failure abstains and the executing channels absorb its weight;
                # it never fails the trial.
                rubric_score = None
                if args.judge:
                    try:
                        verdict = grade_task_rubric(
                            task.path, diff_path,
                            llm=default_llm,
                            council=args.judge_council or None)
                        (trial.verifier_dir / "rubric_score.json").write_text(
                            json.dumps(verdict.to_dict(), indent=2) + "\n")
                        rubric_score = verdict.overall_score
                        job.log(f"{trial.trial_name} rubric: {verdict.error or rubric_score}")
                    except Exception as exc:
                        job.log(f"{trial.trial_name} rubric not judged: "
                                f"{type(exc).__name__}: {exc}")

                score = trial.write_test_outputs(outcomes, weights, rubric_score)
                if score is not None:
                    trial.rewards["weighted_score"] = score
                    trial._write_reward_text()
                    trial.write()
        trial.stage_verifier_inputs(task.path)

        trial.write_ctrf(checks)

    except Exception as exc:  # noqa: BLE001 - one bad task must not kill the batch
        logger.error("[%s] crashed: %s", task.repo_id, exc, exc_info=True)
        trial.fail(exc)

    finally:
        trial.finish()

    return trial


def _gate_report(agent_ok: bool, graded, diff_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Express the outcome as CTRF test entries.

    Reported at the granularity the grader actually returns: MigrationBench gives
    one boolean per tier, not per criterion, so claiming five separate gate
    results here would be inventing detail we do not have.
    """
    def entry(name: str, ok: Optional[bool], note: str) -> Dict[str, Any]:
        return {
            "name": name,
            "status": "passed" if ok else ("skipped" if ok is None else "failed"),
            "duration": 0,
            "message": note,
        }

    return [
        entry("agent_run_completed", agent_ok, "agent container exited cleanly"),
        entry("minimal_migration", graded.minimal, "MigrationBench r1-r3"),
        entry("maximal_migration", graded.maximal, "MigrationBench r1-r3 + r5"),
        # r3 done so it fires on multi-module repositories; measured on jWebForm,
        # MigrationBench's own r3 compared an empty list to an empty list and
        # passed a repository whose suite is 48 tests across 5 modules.
        entry("tests_preserved", graded.tests_preserved,
              "no test method removed or renamed, read from the patch"),
        # Sanity checks the weights file scores at weight 1 each. A diff exists
        # only if the repository was there to diff, so the first is implied by
        # the second having anything to measure at all.
        entry("repository_present", diff_path is not None and diff_path.is_file(),
              "the repository survived the agent"),
        entry("agent_made_changes",
              diff_path is not None and diff_path.is_file()
              and diff_path.stat().st_size > 0,
              "the agent left a non-empty patch"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Java 8 to 17 migration tasks, one container per repository",
    )
    parser.add_argument("--run-label", default="run",
                        help="Tag for this invocation's collected artifacts. Not a "
                             "directory: attempts are numbered run_N per task, model "
                             "and agent, so a label never has to be unique.")
    parser.add_argument("--agent-type", default="rag",
                        choices=["baseline", "pe", "hybrid", "rag", "oracle"],
                        help="'oracle' runs the task's own solution instead of an agent: deterministic, needs no model, and exercises every other stage.")
    parser.add_argument("--tasks-dir", default=str(REPO_ROOT / "tasks"))
    parser.add_argument("--repos", nargs="+", default=None,
                        help="Restrict to these repositories. Defaults to every task.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=str(REPO_ROOT / "bundles"),
                        help="Bundle root: <out>/<task-slug>/trajectories/"
                             "<model>/<agent>/run_N/ (default: bundles/).")
    parser.add_argument("--resume", action="store_true",
                        help="Skip repositories that already have a finished run.")

    parser.add_argument("--judge", action="store_true",
                        help="Score the rubric's reviewer-written criteria with a "
                             "model. Needs a model endpoint; the channel abstains "
                             "if one is unreachable, and never fails a trial.")
    parser.add_argument("--judge-council", nargs="+", default=None, metavar="MODEL",
                        help="Judge model ids. Two or more resolve a criterion by "
                             "unanimity, falling back to the first as tiebreak.")
    parser.add_argument("--generated-tests", action="store_true",
                        help="Also generate a contract suite from the base commit and "
                             "check the migration preserves what the base tree confirmed. "
                             "Needs a model endpoint; advisory unless --enforce-generated.")
    parser.add_argument("--enforce-generated", action="store_true",
                        help="Make --generated-tests binding on the reward.")
    parser.add_argument("--airgap", action="store_true",
                        help="Put agents on an internal network whose only exit is an "
                             "allowlisting proxy, so the upstream repository cannot be "
                             "re-fetched.")
    parser.add_argument("--allow-host", action="append", default=None)

    parser.add_argument("--litellm", action="store_true")
    parser.add_argument("--llm-api-base", default=None,
                        help="Base URL the LiteLLM sidecar routes to.")
    parser.add_argument("--llm-api-key", default=None,
                        help="Credential for that upstream. Falls back to "
                             "LLM_API_KEY.")
    parser.add_argument("--model-provider", default="bedrock",
                        choices=["bedrock", "anthropic", "openai"])
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--model-base-url", default=None)

    parser.add_argument("--migrationbench-commit", default=MIGRATIONBENCH_COMMIT)
    parser.add_argument("--rebuild", action="store_true")

    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--max-messages", type=int, default=80)
    parser.add_argument("--shell-timeout", type=int, default=1800)
    parser.add_argument("--hf-dataset",
                        default="AmazonScience/migration-bench-java-selected")
    parser.add_argument("--aws-dir", default=None,
                        help="Mount these AWS credentials into the agent container. "
                             "Off by default: the agent has a shell, and read-only "
                             "prevents writes rather than reads. Only needed for "
                             "--model-provider bedrock.")
    parser.add_argument("--container-prefix", default="jma")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        du._require_docker()
    except RuntimeError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    tasks = load_tasks(args.tasks_dir, args.repos, args.limit)
    if not tasks:
        logger.error("No tasks under %s", args.tasks_dir)
        sys.exit(1)

    out_dir = Path(args.out).expanduser().resolve()
    if args.resume:
        before = len(tasks)
        done = completed_slugs(out_dir)
        tasks = [t for t in tasks if t.slug not in done]
        logger.info("Resume: %d of %d already finished", before - len(tasks), before)
        if not tasks:
            logger.info("Nothing left to run")
            sys.exit(0)

    base = build_base_image(args.migrationbench_commit, force=args.rebuild)
    image = du.resolve_image_tag()
    du.build_images(image=image, base_image=base, force=args.rebuild)

    model_provider = args.model_provider
    model_id = args.model_id
    model_base_url = args.model_base_url
    network: Optional[str] = None
    maven_settings: Optional[Path] = None
    extra_env: Dict[str, str] = {}

    # The oracle replays the task's own solution: no model runs, so recording the
    # provider here would name a directory after something that never executed.
    model_name = model_id or (None if args.agent_type == "oracle" else model_provider)

    job = Bundle(
        out_dir,
        agent_name=f"javamigration-{args.agent_type}",
        model_name=model_name or "none",
    )
    trials: List[Trial] = []
    with contextlib.ExitStack() as stack:
        if args.airgap:
            # Strict by default: the agent reaches only the sidecars attached to
            # its network, because no other route exists. Naming a host is the
            # explicit opt-in to the weaker, filtered form.
            if args.allow_host:
                airgap = stack.enter_context(
                    Airgap(mode="allowlist", allowed_hosts=tuple(args.allow_host)))
            else:
                airgap = stack.enter_context(Airgap())
            network = airgap.network
            maven_settings = airgap.maven_settings_path
            if airgap.strict:
                logger.info("Airgap on (strict): agents have no route out")
            else:
                logger.info("Airgap on (allowlist): agents reach only %s",
                            ", ".join(airgap.allowed_hosts))

        if args.litellm:
            sidecar = stack.enter_context(
                LiteLLMSidecar(
                    network=network or "jma-net",
                    model_list=build_model_list(
                        llm_api_base=args.llm_api_base,
                        llm_api_key=args.llm_api_key or os.environ.get("LLM_API_KEY"),
                    ),
                )
            )
            if args.airgap:
                # The proxy allowlist does not cover the model endpoint, so the
                # sidecar needs its own way out.
                airgap.attach(sidecar.container)
            network = sidecar.network
            model_provider = "openai"
            model_id = DEFAULT_MODEL_ALIAS
            model_base_url = sidecar.base_url
            extra_env["OPENAI_API_KEY"] = sidecar.master_key

        os.environ.update(extra_env)
        logger.info(
            "Job %s: %d tasks, %d at a time, agent=%s, provider=%s",
            args.run_label, len(tasks), args.max_parallel, args.agent_type, model_provider,
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
            futures = {
                pool.submit(
                    run_one, task,
                    job=job, args=args, image=image, network=network,
                    model_provider=model_provider, model_id=model_id,
                    model_base_url=model_base_url, maven_settings=maven_settings,
                ): task
                for task in tasks
            }
            for future in concurrent.futures.as_completed(futures):
                task = futures[future]
                try:
                    trials.append(future.result())
                except Exception as exc:  # noqa: BLE001
                    logger.error("[%s] driver crashed: %s", task.repo_id, exc)

    job_result = job.finish()
    total = len(trials)
    ok = [t for t in trials if t.exception_info is None]
    # eta is a rate over trials that were actually assessed. A trial the grading
    # toolchain could not assess belongs in neither numerator nor denominator.
    graded = [t for t in trials if t.rewards.get("reward") is not None]
    n = len(graded)
    minimal = sum(1 for t in graded if t.rewards.get("minimal"))
    maximal = sum(1 for t in graded if t.rewards.get("maximal"))

    print()
    print(f"  run          {args.run_label}  ({args.agent_type}, {model_provider})")
    print(f"  trials       {total}   completed {len(ok)}   errored {total - len(ok)}")
    if total != n:
        print(f"  ungraded     {total - n}   (excluded from eta; grading did not run)")
    if n:
        print(f"  eta_minimal  {100.0 * minimal / n:.2f}%   ({minimal}/{n})")
        print(f"  eta_maximal  {100.0 * maximal / n:.2f}%   ({maximal}/{n})")
        print(f"  mean reward  {job_result['stats']['evals'][f'javamigration-{args.agent_type}__tasks']['metrics'][0]['mean']:.3f}")
    print(f"  graded vs    MigrationBench@{args.migrationbench_commit[:7]}")
    print(f"  bundles      {job.path}")

    sys.exit(0 if len(ok) == total else 1)


if __name__ == "__main__":
    main()
