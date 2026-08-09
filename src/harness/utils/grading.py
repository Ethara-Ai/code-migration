"""
Grading, outside the agent's container.

MigrationBench's evaluator resolves commit ids against GitHub, so it needs
network the agent must not have -- the whole point of the airgap is that the
upstream repository, and therefore the answer, is unreachable. The two cannot
share a network policy while they share a container.

So grading moves out: the agent produces a patch, its container is destroyed, and
the patch is graded in a fresh container with `run_eval_in_docker`. That is
already in MigrationBench and mounts its input read-only, which also closes the
other hole -- with grading inside the agent's container the agent ran as root and
could rewrite `final_eval.py` or the r5 reference list.

Grading a *patch* rather than a directory means the grader never sees the agent's
filesystem at all.
"""

from __future__ import annotations

import contextlib
import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


#: A test file, in any module -- not just <root>/src/test.
_TEST_PATH = re.compile(r"(^|/)src/test/")
#: `public void foo(` and friends; captures the method name.
#:
#: The return type must start with a word character. The character class that
#: spells a generic type has to allow spaces -- `Map<String, Integer>` -- and
#: without that anchor the class also matches the line's own indentation, so
#: `        assertTrue(x);` parses as return type "        ", method "assertTrue".
#: Measured on a patch that removed assertions and no methods: r3 reported
#: assertEquals, assertTrue and assertFalse as deleted tests.
_METHOD = re.compile(r"^[-+]\s*(?:@\w+[^\n]*\s+)?(?:public|protected|private)?\s*"
                     r"(?:static\s+)?[\w<>\[\]][\w<>\[\], .]*\s+(\w+)\s*\([^)]*\)"
                     r"\s*(?:throws [\w., ]+)?\s*\{?")


#: MigrationBench reports an infrastructure failure as a criterion failure.
#: `check_version` (r5) shells out to `mvn dependency:tree` with no retry, and any
#: non-zero exit becomes "the dependencies are not at their latest major version".
#: Measured on a byte-identical patch: one spurious maximal failure in seven
#: gradings, against twelve transient resolution errors across four runs.
#:
#: Only these two phrases are used. "Could not resolve dependencies" is NOT --
#: MigrationBench retries that path, and it appears in every passing grading here
#: (3, 3, 3 and 12 times), so matching it would fail every run.
_INFRA_FAILURE = re.compile(
    r"Failed to generate effective POM|Error generating dependency-tree"
)


@dataclass
class Verdicts:
    """Both tiers for one repository, plus the checks MigrationBench misses."""

    minimal: Optional[bool] = None
    maximal: Optional[bool] = None
    error: str = ""
    tests_preserved: Optional[bool] = None
    removed_tests: List[str] = field(default_factory=list)

    @property
    def graded(self) -> bool:
        return self.minimal is not None


def check_tests_preserved(diff_path: Path) -> Tuple[bool, List[str]]:
    """
    r3, done so that it actually fires.

    MigrationBench globs `<root>/src/test`, which finds nothing in a multi-module
    project where tests live in `<module>/src/test`. It then compares an empty
    list to an empty list and passes, so on most of `selected` -- 6.6 modules on
    average -- the guard against deleting or renaming tests never runs.

    This reads the patch instead of the tree, so it needs no container and works
    for any layout: any method name that a test file removes and does not add
    back has been deleted or renamed.

    Args:
        diff_path: The migration patch

    Returns:
        (preserved, names removed and not re-added)
    """
    if not diff_path.is_file():
        return True, []

    removed: Set[str] = set()
    added: Set[str] = set()
    in_test_file = False

    for line in diff_path.read_text(errors="replace").splitlines():
        if line.startswith("diff --git "):
            in_test_file = bool(_TEST_PATH.search(line.split(" b/")[-1]))
            continue
        if not in_test_file or line.startswith(("+++", "---")):
            continue

        match = _METHOD.match(line)
        if not match:
            continue
        (removed if line.startswith("-") else added).add(match.group(1))

    # A rename shows the old name removed and the new one added, so set
    # difference catches renames and deletions alike. An import-only change
    # never touches a method line, so it cannot produce a false positive.
    missing = sorted(removed - added)
    return not missing, missing


def grade_diff(
    github_url: str,
    diff_path: Path,
    *,
    use_docker: bool = True,
    confirm_unstable: bool = True,
) -> Verdicts:
    """
    Grade a migration patch with MigrationBench.

    Maximal is evaluated first: eq. (2) is eq. (1) plus r5, so a maximal pass
    implies a minimal pass and the second call can be skipped. With the paper's
    rates that averages ~1.5 evaluations instead of 2.

    Args:
        github_url: Repository the patch applies to
        diff_path: Patch produced by the agent
        use_docker: Grade in a fresh container. False grades in-process, which
            requires JDK 17 and Maven on this host.
        confirm_unstable: Re-check a maximal failure that r5 alone caused. Costs
            one extra evaluation on exactly the shape the dependency-tree flake
            produces, and nothing at all on a passing or minimally-failing run.

    Returns:
        Verdicts, with `error` set if grading could not run. A verdict of `None`
        means the migration was never graded and must not be counted as a
        failure -- absence of evidence is not evidence of failure.
    """
    if not diff_path.is_file() or diff_path.stat().st_size == 0:
        return Verdicts(minimal=False, maximal=False, error="empty or missing diff")

    try:
        if use_docker:
            from migration_bench.common.docker_utils import run_eval_in_docker as run_eval
        else:
            from migration_bench.eval.final_eval import run_eval
    except ImportError as exc:
        return Verdicts(error=f"migration_bench not importable: {exc}")

    def _eval(maximal: bool) -> Tuple[bool, str]:
        """Evaluate, keeping the grader's own output so failures can be attributed.

        `run_eval_in_docker` prints the container's combined output and returns
        only a bool, so the reason a criterion failed is otherwise lost -- which
        is what makes a broken toolchain indistinguishable from a bad migration.
        """
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            passed = bool(run_eval(
                github_url=github_url,
                git_diff_file=str(diff_path),
                require_maximal_migration=maximal,
                # r4 is not run. The `test_execution` gate subsumes it and is
                # stricter -- r4 counts a skipped test as present -- and r4 is a
                # second full `mvn test` on top of the `mvn clean verify` r1
                # already does. Keeping both would execute the suite three times
                # per graded attempt for the weaker of the two checks.
                eval_num_tests=False,
            ))
        log = buffer.getvalue()
        print(log, end="")  # keep the operator-visible behaviour unchanged
        return passed, log

    preserved, missing = check_tests_preserved(diff_path)
    if not preserved:
        logger.warning("Test methods removed or renamed: %s", missing)

    try:
        maximal, log = _eval(True)
        if _INFRA_FAILURE.search(log):
            return Verdicts(error="grading toolchain failed (mvn dependency:tree); "
                                  "the maximal verdict is not about this migration",
                            tests_preserved=preserved, removed_tests=missing)

        # Maximal implies minimal by construction, so only ask when it failed.
        if maximal:
            minimal = True
        else:
            minimal, log = _eval(False)
            if _INFRA_FAILURE.search(log):
                return Verdicts(error="grading toolchain failed (mvn dependency:tree); "
                                      "the minimal verdict is not about this migration",
                                tests_preserved=preserved, removed_tests=missing)

            # Minimal passing while maximal fails is exactly the shape the
            # dependency-tree flake produces, because r5 is the only criterion
            # that shells out to it. Confirming here costs one evaluation on the
            # rare case and none on the common ones.
            if minimal and confirm_unstable:
                again, _ = _eval(True)
                if again != maximal:
                    return Verdicts(
                        error="maximal verdict did not reproduce "
                              f"({maximal} then {again}); trial is unusable",
                        tests_preserved=preserved, removed_tests=missing)

        # A migration that dropped a test has not preserved behaviour, whatever
        # MigrationBench says -- its own r3 cannot see this on multi-module repos.
        if not preserved:
            minimal = maximal = False
        return Verdicts(minimal=minimal, maximal=maximal,
                        tests_preserved=preserved, removed_tests=missing)
    except Exception as exc:  # noqa: BLE001 - a grading failure must not lose the run
        logger.error("Grading failed for %s: %s", github_url, exc)
        return Verdicts(error=str(exc), tests_preserved=preserved,
                        removed_tests=missing)
