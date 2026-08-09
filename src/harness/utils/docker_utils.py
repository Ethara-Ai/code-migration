"""
Container primitives for one-repository-per-container migration runs.

The agent, the repository clone and the MigrationBench evaluation all execute
inside the container; callers stay on the host and supervise. That buys three
things:

* agent and grader share one JDK, so r2 (compiled class major version 61) stops
  depending on the host toolchain;
* each repository gets its own filesystem, so the shared `/tmp/workdir` basename
  collision in `Repository.from_huggingface` cannot bite;
* the agent's shell cannot reach the host, which the string-level checks in
  `tools/shell_tools.py` never actually guaranteed.

`run_repository` is the engine; `eval/run_batch.py` is the only driver.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from harness.utils.litellm import (
    DEFAULT_LITELLM_IMAGE,
    DEFAULT_MODEL_ALIAS,
    LiteLLMSidecar,
    build_model_list,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_BASE_IMAGE = "migration-bench:latest"
DEFAULT_IMAGE_NAME = "java-migration-agent"

#: Copied into the container by name only -- the values never appear in argv,
#: unlike `-e KEY=value`. Mounting ~/.aws is still the better path.
AWS_ENV_PASSTHROUGH = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_DEFAULT_REGION",
    "AWS_REGION",
    "AWS_PROFILE",
    "AWS_BEARER_TOKEN_BEDROCK",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
)

CONTAINER_OUTPUT_DIR = "/out"

#: Docker Desktop resolves this already; on Linux it needs the explicit
#: --add-host below. Either way it is how a container reaches a service bound to
#: the host's loopback.
HOST_ALIAS = "host.docker.internal"

#: Where the airgap's generated Maven settings.xml is mounted, and the proxy the
#: agent reaches the allowlisted hosts through.
MAVEN_SETTINGS_PATH = "/tmp/maven-settings.xml"
#: Where the task's instruction.md is mounted for the agent to read.
INSTRUCTION_PATH = "/app/instruction.md"
PROXY_HOST = "jma-proxy"
PROXY_PORT = 8888


def rewrite_loopback(url: Optional[str]) -> Optional[str]:
    """
    Point a host-loopback URL at the host as seen from inside a container.

    A loopback URL means the container itself once the agent is running
    inside one, where nothing is listening.
    """
    if not url:
        return url
    return re.sub(r"//(127\.0\.0\.1|localhost)\b", f"//{HOST_ALIAS}", url)


@dataclass
class ContainerRun:
    """Outcome of one repository's containerized run."""

    repo_id: str
    container_name: str
    succeeded: bool = False
    attempts: int = 0
    exit_code: Optional[int] = None
    error: str = ""
    artifacts: List[str] = field(default_factory=list)


def _run(cmd: List[str], timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _require_docker() -> None:
    if shutil.which("docker") is None:
        raise RuntimeError("docker not found on PATH")
    probe = _run(["docker", "info"])
    if probe.returncode != 0:
        raise RuntimeError(f"docker is not usable: {probe.stderr.strip()}")


def resolve_image_tag(name: str = DEFAULT_IMAGE_NAME) -> str:
    """
    Tag the agent image with the source that goes into it.

    The commit alone is not enough: with uncommitted changes the tag would not
    move, so `_image_exists` would keep returning a stale image built from older
    source -- and "traceable to exact code" would be false. When the tree is
    dirty, the digest of everything shipped into the image is appended instead.
    """
    rev = _run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"])
    base = rev.stdout.strip() if rev.returncode == 0 and rev.stdout.strip() else "nogit"
    if base == "nogit":
        logger.warning("Not a git checkout; image tag cannot identify the source")

    digest = _source_digest()
    return f"{name}:{base}" if digest is None else f"{name}:{base}-{digest}"


def _source_digest() -> Optional[str]:
    """
    Short digest of the source shipped into the image, or None if it is clean.

    Only what Dockerfile.agent copies is hashed -- src/ and pyproject.toml -- so
    editing a task or a result does not needlessly invalidate the image.
    """
    dirty = _run(["git", "-C", str(REPO_ROOT), "status", "--porcelain",
                  "--", "src", "pyproject.toml"])
    if dirty.returncode != 0 or not dirty.stdout.strip():
        return None

    digest = hashlib.sha1()
    for path in sorted((REPO_ROOT / "src").rglob("*.py")) + [REPO_ROOT / "pyproject.toml"]:
        if "__pycache__" in path.parts:
            continue
        digest.update(str(path.relative_to(REPO_ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:8]


def _image_exists(image: str) -> bool:
    probe = _run(["docker", "images", "-q", image])
    return probe.returncode == 0 and bool(probe.stdout.strip())


def build_images(
    image: str,
    base_image: str = DEFAULT_BASE_IMAGE,
    migrationbench_dir: Optional[str] = None,
    force: bool = False,
) -> None:
    """
    Ensure the base evaluation image and the agent image both exist.

    The base is MigrationBench's own Dockerfile (JDK 17 + Maven 3.9.6 + git +
    migration_bench). It is only built here if a checkout was supplied; otherwise
    it must already be present.
    """
    if force or not _image_exists(base_image):
        if not migrationbench_dir:
            raise RuntimeError(
                f"Base image {base_image} not found. Build it from a MigrationBench "
                f"checkout (`docker build -t {base_image} .`) or pass "
                f"--migrationbench-dir."
            )
        logger.info("Building base image %s from %s", base_image, migrationbench_dir)
        build = _run(
            ["docker", "build", "-t", base_image, str(migrationbench_dir)],
            timeout=3600,
        )
        if build.returncode != 0:
            raise RuntimeError(f"Base image build failed:\n{build.stderr[-4000:]}")

    if force or not _image_exists(image):
        logger.info("Building agent image %s", image)
        build = _run(
            [
                "docker", "build",
                "-f", str(REPO_ROOT / "docker" / "Dockerfile.agent"),
                "--build-arg", f"BASE_IMAGE={base_image}",
                "-t", image,
                str(REPO_ROOT),
            ],
            timeout=3600,
        )
        if build.returncode != 0:
            raise RuntimeError(f"Agent image build failed:\n{build.stderr[-4000:]}")

    logger.info("Images ready: %s (on %s)", image, base_image)


def _container_name(prefix: str, exp_id: str, repo_id: str, attempt: int) -> str:
    """
    Build a Docker-legal name that is unique per repository *and* attempt.

    Using the full repo_id rather than its basename matters: `Repository` derives
    its workspace from the basename alone, so `alice/foo` and `bob/foo` would
    otherwise collide.
    """
    slug = re.sub(r"[^A-Za-z0-9_.-]", "-", f"{exp_id}-{repo_id}")
    return f"{prefix}-{slug}-{attempt}"[:120]


def _remove_container(name: str) -> None:
    _run(["docker", "rm", "-f", name])


def _docker_run_argv(
    name: str,
    image: str,
    cpus: str,
    memory: str,
    aws_dir: Optional[Path],
    network: Optional[str],
    maven_settings: Optional[Path] = None,
    solution_dir: Optional[Path] = None,
    instruction_file: Optional[Path] = None,
) -> List[str]:
    argv = [
        "docker", "run", "-d",
        "--name", name,
        "--cpus", cpus,
        "--memory", memory,
        # Needed on Linux for HOST_ALIAS to resolve; harmless on Docker Desktop.
        "--add-host", f"{HOST_ALIAS}:host-gateway",
    ]

    if network:
        # Joining a user-defined network lets the agent reach a sidecar such as
        # the upstream by container name, with no port published on the host.
        argv += ["--network", network]

    if solution_dir and Path(solution_dir).is_dir():
        argv += ["-v", f"{Path(solution_dir).resolve()}:/solution:ro"]

    if instruction_file and Path(instruction_file).is_file():
        argv += ["-v", f"{Path(instruction_file).resolve()}:{INSTRUCTION_PATH}:ro"]

    if maven_settings and Path(maven_settings).is_file():
        # Maven's resolver ignores -Dhttp.proxyHost; it reads settings.xml. Passing
        # the proxy any other way silently fails to route, and Maven then cannot
        # even resolve a plugin prefix.
        argv += ["-v", f"{maven_settings}:{MAVEN_SETTINGS_PATH}:ro"]
        proxy = f"http://{PROXY_HOST}:{PROXY_PORT}"
        # http_proxy covers curl and friends; MAVEN_ARGS is what actually makes
        # Maven read the settings file (3.9+ honours it).
        argv += [
            "-e", f"http_proxy={proxy}",
            "-e", f"https_proxy={proxy}",
            "-e", f"MAVEN_ARGS=-s {MAVEN_SETTINGS_PATH}",
        ]

    if aws_dir and aws_dir.is_dir():
        argv += ["-v", f"{aws_dir}:/home/migrationbench/.aws:ro"]

    for key in AWS_ENV_PASSTHROUGH:
        if key in os.environ:
            # Name only: docker reads the value from our environment, so it never
            # lands in the host process list.
            argv += ["-e", key]

    argv += [image, "/bin/bash", "-c", "tail -f /dev/null"]
    return argv


def _oracle_argv(exp_id: str, repo_id: str) -> List[str]:
    """
    Run the task's own solution instead of an agent.

    Deterministic, needs no model, and exercises every other stage -- image
    build, container lifecycle, artifact collection, grading, gates. It is also
    the check that a task is solvable at all.

    The diff is captured here rather than assumed. Grading reads a patch from
    /out, and only JavaMigration wrote one; any other agent left it empty and the
    run graded as a failure for lack of an artifact rather than lack of a
    migration. Capturing it in the harness makes the contract the same for every
    agent.
    """
    slug = repo_id.replace("/", "__")
    out = f"{CONTAINER_OUTPUT_DIR}/{exp_id}"
    return ["bash", "-c",
            "bash /solution/solve.sh; "
            f"mkdir -p {out}; cd /app/repo; "
            "BASE=$(python3 -c \"import json;print(json.load(open('/app/task_meta.json'))['base_commit'])\"); "
            f"git diff \"$BASE\" -- . ':(exclude)target/*' ':(exclude)*/target/*' > {out}/{slug}.diff; "
            f"test -s {out}/{slug}.diff"]


def _agent_argv(
    repo_id: str,
    agent_type: str,
    exp_id: str,
    hf_dataset: str,
    model_provider: str,
    model_id: Optional[str],
    model_base_url: Optional[str],
    max_messages: int,
    shell_timeout: int,
    maven_settings: Optional[Path] = None,
    repo_path: Optional[str] = None,
    github_url: Optional[str] = None,
    skip_eval: bool = False,
    instruction_file: Optional[Path] = None,
) -> List[str]:
    argv = [
        "python", "-m", "strands_agent.main",
        "--agent-type", agent_type,
        "--exp-id", exp_id,
        "--model-provider", model_provider,
        "--max-messages", str(max_messages),
        "--shell-timeout", str(shell_timeout),
        "--output-dir", CONTAINER_OUTPUT_DIR,
    ]
    if repo_path:
        # The task environment already cloned and hardened the repository, so the
        # agent needs neither the dataset nor github -- which is what lets the
        # airgap allowlist exclude both.
        argv += ["--repo-path", repo_path]
        if github_url:
            argv += ["--github-url", github_url]
    if instruction_file:
        argv += ["--instruction-file", INSTRUCTION_PATH]
    if skip_eval:
        argv += ["--skip-eval"]
    else:
        argv += ["--repo-id", repo_id, "--hf-dataset", hf_dataset]
    if model_id:
        argv += ["--model-id", model_id]
    if model_base_url:
        argv += ["--model-base-url", rewrite_loopback(model_base_url)]
    return argv


def _collect_artifacts(name: str, exp_id: str, dest: Path) -> List[str]:
    """Copy the container's output directory back to the host."""
    dest.mkdir(parents=True, exist_ok=True)
    copy = _run(["docker", "cp", f"{name}:{CONTAINER_OUTPUT_DIR}/{exp_id}/.", str(dest)])
    if copy.returncode != 0:
        logger.warning("[%s] no artifacts collected: %s", name, copy.stderr.strip())
        return []
    return sorted(p.name for p in dest.iterdir() if p.is_file())


def run_repository(
    repo_id: str,
    *,
    image: str,
    agent_type: str,
    exp_id: str,
    hf_dataset: str,
    model_provider: str,
    model_id: Optional[str],
    model_base_url: Optional[str],
    max_messages: int,
    shell_timeout: int,
    run_timeout: int,
    output_dir: Path,
    cpus: str,
    memory: str,
    aws_dir: Optional[Path],
    network: Optional[str],
    max_attempts: int,
    container_prefix: str,
    maven_settings: Optional[Path] = None,
    repo_path: Optional[str] = None,
    github_url: Optional[str] = None,
    skip_eval: bool = False,
    solution_dir: Optional[Path] = None,
    instruction_file: Optional[Path] = None,
) -> ContainerRun:
    """
    Migrate one repository inside a throwaway container.

    Retries escalate CPU and memory (`base * 2**failures`), so a build killed for
    running out of memory gets a second chance instead of being lost.
    """
    dest = output_dir / exp_id
    result = ContainerRun(repo_id=repo_id, container_name="")

    for attempt in range(1, max_attempts + 1):
        factor = 2 ** (attempt - 1)
        name = _container_name(container_prefix, exp_id, repo_id, attempt)
        result.container_name = name
        result.attempts = attempt

        scaled_cpus = str(float(cpus) * factor)
        scaled_memory = _scale_memory(memory, factor)

        _remove_container(name)
        logger.info(
            "[%s] attempt %d/%d (cpus=%s memory=%s)",
            name, attempt, max_attempts, scaled_cpus, scaled_memory,
        )

        try:
            started = _run(
                _docker_run_argv(
                    name, image, scaled_cpus, scaled_memory, aws_dir, network,
                    maven_settings, solution_dir, instruction_file,
                )
            )
            if started.returncode != 0:
                result.error = f"container start failed: {started.stderr.strip()}"
                logger.error("[%s] %s", name, result.error)
                continue

            if agent_type == "oracle":
                argv = _oracle_argv(exp_id, repo_id)
            else:
                argv = _agent_argv(
                    repo_id, agent_type, exp_id, hf_dataset,
                    model_provider, model_id, model_base_url,
                    max_messages, shell_timeout,
                    repo_path=repo_path, github_url=github_url, skip_eval=skip_eval,
                    instruction_file=instruction_file,
                )
            log_path = dest / f"{repo_id.replace('/', '__')}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                with log_path.open("w", encoding="utf-8") as log_file:
                    proc = subprocess.run(
                        ["docker", "exec", name, *argv],
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        timeout=run_timeout,
                    )
                result.exit_code = proc.returncode
            except subprocess.TimeoutExpired:
                result.exit_code = None
                result.error = f"timed out after {run_timeout}s"
                logger.error("[%s] %s", name, result.error)

            result.artifacts = _collect_artifacts(name, exp_id, dest)

            if result.exit_code == 0:
                result.succeeded = True
                result.error = ""
                logger.info("[%s] done: %s", name, ", ".join(result.artifacts) or "no files")
                return result

            if not result.error:
                result.error = f"agent exited {result.exit_code}"
            logger.warning("[%s] %s", name, result.error)

        finally:
            _remove_container(name)

    return result


def _scale_memory(memory: str, factor: int) -> str:
    """Scale a docker memory string such as '8g' or '512m' by an integer factor."""
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([bkmgBKMG]?)", memory.strip())
    if not match:
        return memory
    value, unit = match.groups()
    return f"{int(float(value) * factor)}{unit}"
