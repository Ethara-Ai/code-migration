"""
Verification gates beyond MigrationBench's r1-r5.

r1-r5 never read `src/main`, never check that a test *verified* anything, and
never run the program. Each gate here closes a specific cheat the others cannot
see, so the stack is deliberately redundant:

    deprecation  did the migration do any work at all?
    coverage     was production code or a test deleted?
    mutation     were the tests hollowed out but kept?

Every gate is a **delta against a source-runtime baseline**, so each needs a
measurement taken on Java 8 at the base commit, before the agent runs. That is
why baselines are captured in their own container: the migration image has only
JDK 17, and Debian dropped `openjdk-8`, so the baseline runs on
`maven:3.9-eclipse-temurin-8` instead of a dual-toolchain image.

Nothing here consults a model. An agent optimises against its grader, and a test
runner, a coverage tool and a bytecode scanner cannot be talked round.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

BASELINE_IMAGE = "maven:3.9-eclipse-temurin-8"
#: JaCoCo must read both v52 (baseline) and v61 (migrated) class files. 0.8.7 is
#: the first release that understands Java 17; anything older writes a
#: header-only CSV and the coverage silently reads as zero.
JACOCO = "org.jacoco:jacoco-maven-plugin:0.8.11"

#: FreshBrew's threshold, from auditing 50 migrations: legitimate refactors
#: stayed under 2.5%, drops past 5% were attributable to reward hacking.
COVERAGE_DROP_LIMIT_PP = 5.0


@dataclass
class Coverage:
    """Line coverage, kept as counts as well as a ratio."""

    covered: int = 0
    missed: int = 0
    measurable: bool = False
    #: Distinguishes "the build is broken" from "there was nothing to measure".
    broken: bool = False
    detail: str = ""

    @property
    def total(self) -> int:
        return self.covered + self.missed

    @property
    def pct(self) -> Optional[float]:
        return (100.0 * self.covered / self.total) if self.total else None


@dataclass
class GateResult:
    name: str
    passed: Optional[bool]          # None = not applicable / not measurable
    detail: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        return "passed" if self.passed else ("skipped" if self.passed is None else "failed")


@dataclass
class Baseline:
    """Everything measured on the source runtime, at the base commit."""

    coverage: Coverage = field(default_factory=Coverage)
    deprecations: Optional[int] = None
    deterministic: Optional[bool] = None
    mutation: Optional[float] = None
    tests: TestRun = field(default_factory=lambda: TestRun())
    #: Generated contract suite and its outcome on the base tree -- the oracle.
    suite: Any = None
    suite_run: Any = None
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "coverage": asdict(self.coverage),
            "coverage_pct": self.coverage.pct,
            "deprecations": self.deprecations,
            "deterministic": self.deterministic,
            "mutation_pct": self.mutation,
            "error": self.error,
        }


PIT = "org.pitest:pitest-maven:1.15.8"

#: PIT needs its JUnit 5 bridge on the *plugin's* classpath, which cannot be set
#: from the command line -- it must be a <dependency> inside <plugin>. It must
#: also not be present for a JUnit 4 project: injected there, PIT's coverage
#: minion dies with UNKNOWN_ERROR. That distinction matters across a migration
#: because the two sides differ -- this repository is JUnit 4 at base and JUnit 5
#: after, so each side needs its own plugin configuration.
_PIT_HEAD = ("<plugin><groupId>org.pitest</groupId><artifactId>pitest-maven</artifactId>"
             "<version>1.15.8</version>")
_PIT_JUNIT5_DEP = ("<dependencies><dependency><groupId>org.pitest</groupId>"
                   "<artifactId>pitest-junit5-plugin</artifactId>"
                   "<version>1.2.1</version></dependency></dependencies>")

#: sed rather than nested python: the plugin XML contains no quotes and no `|`,
#: so a `s|…|…|` substitution needs no escaping at all.
_INJECT_PIT = (
    r"if grep -rqs 'org.junit.jupiter\|junit-jupiter' --include=*.java --include=pom.xml .; "
    f"then PITXML='{_PIT_HEAD}{_PIT_JUNIT5_DEP}</plugin>'; "
    f"else PITXML='{_PIT_HEAD}</plugin>'; fi; "
    "for p in $(find . -name pom.xml -not -path '*/target/*'); do "
    'grep -q pitest-maven "$p" || '
    'sed -i "s|<plugins>|<plugins>$PITXML|" "$p"; '
    "done; "
)


def _mounts(patch):
    """Read-only mount for a patch, or nothing."""
    return [f"{patch.resolve()}:/tmp/strands_agent.patch:ro"] if patch is not None else []


def _apply(patch):
    """Shell prefix applying the patch, excluding committed build output."""
    if patch is None:
        return ""
    return ("{ git apply --exclude='*/target/*' --exclude='target/*' "
            "/tmp/strands_agent.patch || { echo APPLYFAIL; exit 0; }; } && ")


def _exec(image: str, script: str, mounts: Optional[List[str]] = None,
          timeout: int = 2400, network: Optional[str] = None) -> subprocess.CompletedProcess:
    argv = ["docker", "run", "--rm"]
    for m in mounts or []:
        argv += ["-v", m]
    if network:
        argv += ["--network", network]
    # Non-login shell on purpose: Debian's /etc/profile rewrites PATH and drops
    # /opt/maven/bin, so `bash -lc` loses mvn entirely -- and the failure is
    # silent, surfacing only as a missing coverage report.
    argv += [image, "bash", "-c", script]
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)


# --------------------------------------------------------------------------- #
# Coverage
# --------------------------------------------------------------------------- #

#: JaCoCo attaches by *setting the `argLine` property*, which surefire then
#: passes to the forked JVM. A project that declares `<argLine>` in its own
#: surefire <configuration> wins that contest -- a configuration element beats a
#: property, and no `-DargLine=` on the command line can override it. The agent
#: never attaches, no .exec is written, no report exists, and the build passes
#: cleanly having measured nothing.
#:
#: Measured on jWebForm, whose pom sets `<argLine>-Dfile.encoding=...`: zero
#: jacoco.csv files, so coverage_delta reported "could not measure" on a
#: repository with 48 tests and ~45% real coverage.
#:
#: `@{argLine}` is JaCoCo's own documented answer -- surefire expands it late,
#: so the project's flags are kept and JaCoCo's are prepended. Idempotent: a pom
#: that already has it is left alone.
_JACOCO_ARGLINE = (
    "for p in $(find . -name pom.xml -not -path '*/target/*'); do "
    "grep -q '@{argLine}' \"$p\" || "
    "sed -i 's|<argLine>|<argLine>@{argLine} |g' \"$p\"; "
    "done; "
)


_CSV_SUM = r"""
find . -name jacoco.csv | while read f; do
  awk -F, 'NR>1 {m+=$8; c+=$9} END {print m"," c}' "$f"
done | awk -F, '{m+=$1; c+=$2} END {print (NR ? m","c : "none")}'
"""


def _parse_coverage(stdout: str) -> Coverage:
    """
    Read aggregated LINE_MISSED,LINE_COVERED from the report sweep.

    "no CSV at all" and "a CSV with no rows" must stay distinguishable from
    "nothing is covered": all three look identical downstream otherwise, and a
    gate that treats an instrumentation failure as a clean pass is worse than no
    gate. This is the bug in the earlier migration_rl gates.py.
    """
    line = (stdout or "").strip().splitlines()[-1:] or [""]
    raw = line[0].strip()
    if not raw or raw == "none":
        return Coverage(measurable=False, detail="no jacoco.csv produced")
    try:
        missed, covered = (int(x) for x in raw.split(","))
    except ValueError:
        return Coverage(measurable=False, detail=f"unparsable report: {raw!r}")
    if missed + covered == 0:
        return Coverage(measurable=False, detail="report has no rows (agent likely never attached)")
    return Coverage(covered=covered, missed=missed, measurable=True)


def measure_run(image: str, repo: str = "/app/repo",
                patch: Optional[Path] = None,
                network: Optional[str] = None) -> Tuple["TestRun", Coverage]:
    """
    Run the suite once and take both measurements from that single execution.

    Measuring separately would run the suite twice per gate pass, and on a repo
    with any nondeterminism the two numbers would describe different runs -- so
    the coverage figure and the executed-test count could disagree about what
    happened. One run, one truth.

    JaCoCo LINE counters are aggregated across all modules. Aggregation matters:
    a per-module figure would repeat the mistake that makes MigrationBench's r3
    inert on multi-module repositories.
    """
    script = (
        f"cd {repo} && {_apply(patch)}"
        f"{_JACOCO_ARGLINE}"
        f"mvn -q -B clean {JACOCO}:prepare-agent test {JACOCO}:report "
        f'>/dev/null 2>&1; echo "MVNRC=$?"; '
        "F=$(find . -path '*/surefire-reports/*.xml' 2>/dev/null); "
        # Emit raw opening tags and read the attributes in Python; a shell regex
        # over XML is not worth the escaping.
        'if [ -n "$F" ]; then cat $F | tr \'>\' \'\\n\' | grep \'<testsuite \'; '
        "else echo NOREPORTS; fi; "
        "echo ===COVERAGE===; "
        f"{_CSV_SUM}"
    )
    result = _exec(image, script, _mounts(patch), network=network)
    out = result.stdout or ""

    # An empty result means docker never ran the script -- a missing image, a bad
    # mount. Reporting that as "the repo has no tests" would silently pass a gate
    # that never executed.
    if not out.strip():
        why = f"could not run: {(result.stderr or '').strip()[:200]}"
        return TestRun(broken=True, detail=why), Coverage(broken=True, detail=why)
    if "APPLYFAIL" in out:
        why = "patch does not apply"
        return TestRun(broken=True, detail=why), Coverage(broken=True, detail=why)

    built = "MVNRC=0" in out
    tests_out, _, cov_out = out.partition("===COVERAGE===")

    if "NOREPORTS" in tests_out:
        # Maven succeeding with no reports means the repo genuinely has no tests;
        # Maven failing means the build broke, which is a migration failure and
        # must not be reported as an unmeasurable gate.
        tests = (TestRun(detail="repository has no tests") if built else
                 TestRun(broken=True,
                         detail="build failed: no tests could be compiled or run"))
    else:
        tests = _parse_tests(tests_out)

    coverage = _parse_coverage(cov_out)
    if not coverage.measurable and not built:
        coverage = Coverage(broken=True,
                            detail="build failed: no coverage report produced")
    return tests, coverage


def measure_coverage(image: str, repo: str = "/app/repo",
                     patch: Optional[Path] = None,
                     network: Optional[str] = None) -> Coverage:
    """Coverage alone, when the test counts are not wanted."""
    return measure_run(image, repo, patch, network)[1]


def coverage_gate(baseline: Coverage, migrated: Coverage,
                  deterministic: Optional[bool] = None) -> GateResult:
    """Line coverage must not drop by more than the threshold."""
    if migrated.broken:
        return GateResult("coverage_delta", False, migrated.detail,
                          {"baseline": asdict(baseline), "migrated": asdict(migrated)})
    if not baseline.measurable or not migrated.measurable:
        why = baseline.detail or migrated.detail or "unmeasurable"
        return GateResult("coverage_delta", None, f"not measurable: {why}",
                          {"baseline": asdict(baseline), "migrated": asdict(migrated)})

    if deterministic is False:
        return GateResult("coverage_delta", None,
                          "not measurable: baseline coverage differs between two "
                          "identical runs, so a 5pp threshold means nothing here",
                          {"baseline": asdict(baseline), "migrated": asdict(migrated)})

    drop = baseline.pct - migrated.pct
    # FreshBrew's rule exactly: total line coverage must not fall by more than the
    # threshold. Nothing is added to it.
    #
    # An earlier version also required the covered-line count not to fall, to stop
    # a deletion of untested code reading as an improvement -- the ratio rises when
    # the denominator shrinks, measured here as 31.3% -> 40.0%. That guard is gone
    # because it rejected a 3.7pp change the paper's own threshold accepts, and
    # because the question it was answering -- did the migration delete code --
    # is not one a coverage ratio can answer honestly.
    #
    # NOTE: with the api_surface gate removed, nothing in the stack now detects
    # deletion of first-party public API that no test covers. Measured on this
    # repository, that patch moved coverage 31.3% -> 40.0% and passed every
    # remaining check, MigrationBench's included.
    #
    # The covered-line delta is still reported, because it explains a drop that
    # does fire -- it is evidence, not a criterion.
    lines_lost = baseline.covered - migrated.covered
    passed = drop <= COVERAGE_DROP_LIMIT_PP

    return GateResult(
        "coverage_delta", passed,
        f"baseline {baseline.pct:.1f}% ({baseline.covered}/{baseline.total}) -> "
        f"migrated {migrated.pct:.1f}% ({migrated.covered}/{migrated.total}); "
        f"drop {drop:.1f}pp, covered lines {-lines_lost:+d}",
        {"baseline_pct": baseline.pct, "migrated_pct": migrated.pct,
         "drop_pp": round(drop, 2), "covered_line_delta": -lines_lost,
         "limit_pp": COVERAGE_DROP_LIMIT_PP},
    )


@dataclass
class TestRun:
    """Surefire's own counts, which distinguish executed from skipped."""

    run: int = 0
    failures: int = 0
    errors: int = 0
    skipped: int = 0
    measurable: bool = False
    #: Distinguishes "the build is broken" from "there was nothing to measure".
    broken: bool = False
    detail: str = ""

    @property
    def executed(self) -> int:
        """Tests that actually ran a body. Surefire counts a skipped test as run."""
        return self.run - self.skipped


def _parse_tests(stdout: str) -> "TestRun":
    """Sum the attributes of every <testsuite> tag surefire wrote."""
    totals = {"tests": 0, "errors": 0, "skipped": 0, "failures": 0}
    for key in totals:
        for match in re.finditer(rf'{key}="(\d+)"', stdout or ""):
            totals[key] += int(match.group(1))
    if totals["tests"] == 0:
        return TestRun(detail="surefire reported no tests")
    return TestRun(run=totals["tests"], failures=totals["failures"],
                   errors=totals["errors"], skipped=totals["skipped"], measurable=True)


def measure_tests(image: str, repo: str = "/app/repo",
                  patch: Optional[Path] = None) -> "TestRun":
    """
    Per-suite counts from surefire, which distinguish executed from skipped.

    Needed because `Tests run: N` counts a skipped test as run, so disabling one
    leaves MigrationBench's r4 unchanged -- and on a Spring Boot repository whose
    only test is the generated `contextLoads()`, the context is still started by
    class-level setup before JUnit evaluates a method-level @Disabled, so every
    line that test would have covered is still covered. Measured on
    spring-boot-hashids: @Disabled left r3, r4 and coverage all passing, byte for
    byte. This gate is what sees it.
    """
    return measure_run(image, repo, patch)[0]


def test_execution_gate(baseline: TestRun, migrated: TestRun) -> GateResult:
    """
    The same tests must still *execute*, not merely still exist.

    r3 compares names, r4 compares a count that includes skips, and coverage does
    not move when a test is disabled. This is the gate that sees @Disabled,
    @Ignore, and surefire <excludes>.
    """
    if migrated.broken:
        # The migration broke the build or produced an inapplicable patch. That is
        # a failure of the migration, not an absence of evidence.
        return GateResult("test_execution", False, migrated.detail)
    if baseline.measurable and not migrated.measurable:
        # The baseline had tests and the migrated tree does not. "No tests" is a
        # true statement about the result and a description of the cheat.
        return GateResult("test_execution", False,
                          f"had {baseline.run} tests, now none: {migrated.detail}")
    if not baseline.measurable:
        return GateResult("test_execution", None,
                          f"not measurable at baseline: {baseline.detail}")

    newly_skipped = migrated.skipped - baseline.skipped
    fewer_executed = baseline.executed - migrated.executed
    # `maven.test.failure.ignore` and `testFailureIgnore` keep the build green
    # with tests failing, so the counts alone would still look healthy. Read the
    # outcomes surefire recorded rather than Maven's exit status.
    new_failures = (migrated.failures + migrated.errors) - (baseline.failures + baseline.errors)
    passed = newly_skipped <= 0 and fewer_executed <= 0 and new_failures <= 0
    return GateResult(
        "test_execution", passed,
        f"executed {baseline.executed} -> {migrated.executed}, "
        f"skipped {baseline.skipped} -> {migrated.skipped}, "
        f"failed {baseline.failures + baseline.errors} -> {migrated.failures + migrated.errors}",
        {"baseline_executed": baseline.executed, "migrated_executed": migrated.executed,
         "baseline_skipped": baseline.skipped, "migrated_skipped": migrated.skipped,
         "baseline_failed": baseline.failures + baseline.errors,
         "migrated_failed": migrated.failures + migrated.errors},
    )


# --------------------------------------------------------------------------- #
# Deprecation / incompatibility
# --------------------------------------------------------------------------- #

_JDEPRSCAN_HIT = re.compile(r"^\s*(class|interface|enum|method|field)\s", re.M)


def count_deprecations(image: str, repo: str = "/app/repo",
                       patch: Optional[Path] = None,
                       release: str = "17") -> tuple[Optional[int], str]:
    """
    Count uses of APIs deprecated or removed in the target release.

    jdeprscan only knows Java SE, so third-party API churn -- which is most of
    the pain in a Spring or Jackson heavy repository -- is invisible to it.
    jdeps --jdk-internals covers internal API access, and both are counted here.
    Neither replaces authored rules for library churn.
    """
    script = (
        f"cd {repo} && {_apply(patch)}"
        # jdeprscan ships with JDK 9 and later. Run it on a JDK 8 image and the
        # shell reports "command not found", the scan produces nothing, and a
        # count of zero is indistinguishable from a clean repository -- which the
        # gate then reports as a trivial instance and skips. Absence of the tool
        # is not absence of deprecated APIs.
        "command -v jdeprscan >/dev/null || { echo NOTOOL; exit 0; }; "
        "mvn -q -B clean compile >/dev/null 2>&1; "
        "CP=$(find . -type d -path '*/target/classes' | tr '\\n' ' '); "
        "[ -z \"$CP\" ] && echo 'NOCLASSES' && exit 0; "
        f"for d in $CP; do jdeprscan --release {release} \"$d\" 2>/dev/null; done; "
        "echo '---JDEPS---'; "
        "for d in $CP; do jdeps --jdk-internals \"$d\" 2>/dev/null | grep -c 'JDK internal'; done"
    )
    result = _exec(image, script, _mounts(patch), timeout=1800)
    out = result.stdout or ""
    if not out.strip():
        return None, f"could not run: {(result.stderr or '').strip()[:200]}"
    if "APPLYFAIL" in out:
        return None, "patch does not apply"
    if "NOTOOL" in out:
        return None, "jdeprscan unavailable (needs JDK 9+); nothing was scanned"
    if "NOCLASSES" in out:
        return None, "no compiled classes to scan"

    deprecated, _, internals = out.partition("---JDEPS---")
    count = len(_JDEPRSCAN_HIT.findall(deprecated))
    for token in internals.split():
        if token.isdigit():
            count += int(token)
    return count, f"{count} deprecated/internal API uses (release {release})"


def deprecation_gate(base_count: Optional[int], migrated_count: Optional[int]) -> GateResult:
    """
    Base must be > 0 (else the task is trivial), migrated must reach 0.

    This gate needs no tests at all, so unlike the other three it works on
    repositories with a single test file -- and it is itself a fail-to-pass
    check: it fails on the unmigrated code and passes after a real migration.
    """
    if base_count is None or migrated_count is None:
        return GateResult("deprecation_count", None, "not measurable")
    if base_count == 0:
        return GateResult("deprecation_count", None,
                          "trivial instance: base already has 0 deprecated API uses",
                          {"base": 0, "migrated": migrated_count})
    passed = migrated_count == 0
    return GateResult("deprecation_count", passed,
                      f"base {base_count} -> migrated {migrated_count}",
                      {"base": base_count, "migrated": migrated_count})


# --------------------------------------------------------------------------- #
# Mutation -- applicability first
# --------------------------------------------------------------------------- #


def measure_mutation(image: str, repo: str = "/app/repo",
                     patch: Optional[Path] = None) -> tuple[Optional[float], str]:
    """
    Mutation score: the fraction of planted faults the suite detects.

    The only gate that sees assertions hollowed out. Coverage cannot -- an
    emptied test still executes the same lines, so the percentage is unmoved.

    Expensive: the suite is re-run once per mutant, so this runs last.
    """
    script = (
        f"cd {repo} && {_apply(patch)}"
        f"{_INJECT_PIT} "
        # `install`, not `test-compile`: PIT is invoked as a standalone goal, which
        # does not run the reactor, so in a multi-module project a module's sibling
        # dependency is unresolvable and the run dies before mutating anything.
        "mvn -q -B clean install -DskipTests >/dev/null 2>&1; "
        f"mvn -B {PIT}:mutationCoverage -DoutputFormats=CSV -DtimeoutConstant=8000 "
        "  >/tmp/pit.log 2>&1; "
        "F=$(find . -name mutations.csv | head -1); "
        'if [ -z "$F" ]; then echo NOREPORT; tail -3 /tmp/pit.log; exit 0; fi; '
        "awk -F, '{t++; if ($6 ~ /KILLED/) k++} END {print (k+0)\",\"(t+0)}' \"$F\""
    )
    result = _exec(image, script, _mounts(patch), timeout=5400)
    out = (result.stdout or "").strip()
    if not out or "NOREPORT" in out:
        return None, f"PIT produced no report: {out[-160:] or 'no output'}"
    try:
        killed, total = (int(x) for x in out.splitlines()[-1].split(","))
    except ValueError:
        return None, f"unparsable PIT output: {out.splitlines()[-1]!r}"
    if total == 0:
        return None, "PIT generated no mutants"
    return 100.0 * killed / total, f"{killed}/{total} mutants killed"


def mutation_gate(base_score: Optional[float], migrated_score: Optional[float],
                  baseline_coverage: Coverage, limit_pp: float = 10.0) -> GateResult:
    """
    Mutation score must not drop materially.

    Gated behind coverage: a suite executing little of the code kills few mutants
    either way, so the delta carries no signal and PIT is the most expensive gate
    to learn nothing with.

    No published threshold exists for migration -- FreshBrew's 5pp is for
    coverage and the distributions differ. 10pp is provisional, to be replaced by
    their method: audit ~50 attempts, plot legitimate against hacked, pick a
    conservative cut.
    """
    if baseline_coverage.measurable and (baseline_coverage.pct or 0) < 50.0:
        return GateResult("mutation_delta", None,
                          f"skipped: baseline coverage {baseline_coverage.pct:.1f}% is "
                          "below 50%, so mutation score would not discriminate",
                          {"baseline_pct": baseline_coverage.pct})
    if base_score is None or migrated_score is None:
        return GateResult("mutation_delta", None, "not measurable")

    drop = base_score - migrated_score
    return GateResult("mutation_delta", drop <= limit_pp,
                      f"baseline {base_score:.1f}% -> migrated {migrated_score:.1f}%; "
                      f"drop {drop:.1f}pp (limit {limit_pp}pp, PROVISIONAL)",
                      {"base": base_score, "migrated": migrated_score, "limit_pp": limit_pp})


# --------------------------------------------------------------------------- #

def run_gates(*, base_image: str, target_image: str, patch: Path,
              baseline: Baseline, network: Optional[str] = None,
              generated_tests: bool = False,
              enforce_generated: bool = False) -> List[GateResult]:
    """
    Run the stack against one migration patch, cheapest gate first.

    Order is deliberate: a seconds-long scan should reject before an
    hours-long mutation run is started.
    """
    results: List[GateResult] = []

    migrated_deps, detail = count_deprecations(target_image, patch=patch)
    logger.info("deprecation after migration: %s", detail)
    results.append(deprecation_gate(baseline.deprecations, migrated_deps))

    # One suite run under JaCoCo yields both numbers, so they always describe
    # the same execution.
    migrated_tests, migrated_cov = measure_run(target_image, patch=patch, network=network)
    logger.info("test execution after migration: %s", migrated_tests.detail or
                f"{migrated_tests.executed} executed, {migrated_tests.skipped} skipped")
    results.append(test_execution_gate(baseline.tests, migrated_tests))

    logger.info("coverage after migration: %s", migrated_cov.detail or migrated_cov.pct)
    results.append(coverage_gate(baseline.coverage, migrated_cov,
                                 baseline.deterministic))

    # Most expensive by a wide margin, and pointless on a thin suite, so it is
    # both last and gated on the baseline coverage.
    if baseline.coverage.measurable and (baseline.coverage.pct or 0) >= 50.0:
        migrated_mut, mdetail = measure_mutation(target_image, patch=patch)
        logger.info("mutation after migration: %s", mdetail)
    else:
        migrated_mut = None
    results.append(mutation_gate(baseline.mutation, migrated_mut, baseline.coverage))

    # Opt-in: needs a model and two extra builds. Advisory unless asked otherwise,
    # because the suite is generated evidence about behaviour, not proof of it.
    if generated_tests and baseline.suite is not None:
        from harness.utils.testgen import generated_tests_gate, run_suite

        migrated_suite = run_suite(target_image, baseline.suite, patch=patch)
        logger.info("generated tests after migration: %s", migrated_suite.detail)
        verdict, detail, evidence = generated_tests_gate(
            baseline.suite_run, migrated_suite, enforce=enforce_generated)
        results.append(GateResult("generated_tests", verdict, detail, evidence))

    return results


def capture_baseline(base_image: str, source_image: str = BASELINE_IMAGE,
                     network: Optional[str] = None,
                     generated_tests: bool = False) -> Baseline:
    """
    Measure the source-runtime baseline every gate is a delta against.

    Coverage runs on JDK 8 in its own container: the migration image has only
    JDK 17, and Debian no longer ships openjdk-8. Determinism is screened by
    running it twice -- a repository whose own baseline is unstable cannot
    support a differential gate, and should be dropped rather than graded.
    """
    baseline = Baseline()

    first_tests, first = measure_run(source_image, network=network)
    second = measure_coverage(source_image, network=network)
    baseline.coverage = first
    baseline.tests = first_tests
    baseline.deterministic = (
        first.measurable and second.measurable
        and (first.covered, first.missed) == (second.covered, second.missed)
    )

    baseline.deprecations, _ = count_deprecations(base_image)

    if first.measurable and (first.pct or 0) >= 50.0:
        baseline.mutation, _ = measure_mutation(source_image)

    if generated_tests:
        # Generated from the base source and run on the base tree, so the oracle
        # is established before the migration is ever seen.
        from harness.utils.testgen import generate_suite, run_suite

        baseline.suite = generate_suite(base_image)
        if baseline.suite.usable:
            baseline.suite_run = run_suite(base_image, baseline.suite)
            logger.info("generated contract suite: %s -> oracle %s",
                        baseline.suite.detail, baseline.suite_run.detail)
        else:
            logger.info("generated contract suite unavailable: %s", baseline.suite.detail)
            baseline.suite = None

    logger.info("baseline: coverage=%s deprecations=%s deterministic=%s mutation=%s",
                first.pct, baseline.deprecations, baseline.deterministic,
                baseline.mutation)
    return baseline
