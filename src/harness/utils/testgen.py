"""
Differential validation with generated tests.

The paper's Appendix C.1 suggests generating extra unit tests to raise
behavioural confidence beyond a repository's own suite. Doing that naively puts a
model in the seat that decides the reward, which the gate design forbids: an
agent optimising against a grader will learn to satisfy whatever the grader
believes, and a model can be talked into believing things.

This keeps the model out of that seat by making the *original code* the oracle:

    1. a model writes tests against the source at the base commit
    2. the same tests run on the base tree and on the migrated tree
    3. only tests that PASS on the base tree are trusted -- a wrong assertion
       fails there too, and is discarded before it can accuse anything
    4. a trusted test that then fails on the migrated tree is a regression

So the model proposes and executing code disposes. A hallucinated assertion
costs a little time and changes no verdict.

The design is java_harness's (`multilang/llm_testgen.py`), which implements it for
Python and pytest. Java needs a different shape -- generated tests must compile
against the project's own classpath, so they are written into the module they
target and run through Maven rather than dropped in a directory -- but the oracle
filtering, the advisory default and the honest-skip behaviour are carried across
unchanged.

Advisory by default. `generated_tests_gate` returns a verdict only when
`enforce=True`, because a generated suite is evidence about behaviour, not proof
of it, and no threshold for it has been derived on real migrations.
"""

from __future__ import annotations

import base64
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: The generated class name, fixed so the run can select it with -Dtest and so a
#: previous run's file is always overwritten rather than accumulating.
CLASS_NAME = "HarnessGeneratedContractTest"

_PROMPT = """You are writing a BEHAVIOURAL CONTRACT test suite.

These tests will be run against TWO builds of the same code -- the ORIGINAL and a
migrated one -- to detect any behavioural difference between them.

Write a JUnit 5 test class for the ORIGINAL Java source below.

Requirements:
- The class MUST be named {cls} and MUST be in package {pkg}.
- Exercise every public method with CONCRETE inputs, including edge cases: empty,
  zero, negative, boundary, and error paths via assertThrows.
- Assert on RETURN VALUES and OBSERVABLE BEHAVIOUR only. Never assert on private
  fields or implementation details.
- Encode the ORIGINAL behaviour faithfully. An assertion that is wrong for the
  original is worse than a missing test, because it will be discarded.
- Deterministic and self-contained: no network, no clock, no filesystem, no
  randomness, no Spring context.
- Use only org.junit.jupiter.api and java.* imports, plus the class under test.
- Prefer many small @Test methods with descriptive names.

Return ONLY one java code block containing the complete file.

ORIGINAL SOURCE ({path}):
```java
{source}
```"""


@dataclass
class GeneratedSuite:
    """One generated test class and where it belongs."""

    source: str = ""
    package: str = ""
    module: str = ""
    target: str = ""
    detail: str = ""

    @property
    def usable(self) -> bool:
        return bool(self.source and "@Test" in self.source)


@dataclass
class SuiteRun:
    """Per-test outcomes from one execution of the generated suite."""

    results: Dict[str, bool] = field(default_factory=dict)
    ran: bool = False
    detail: str = ""

    @property
    def passing(self) -> set:
        return {name for name, ok in self.results.items() if ok}


def default_llm(prompt: str, timeout: int = 180) -> str:
    """
    Call whatever endpoint the run is already configured to use.

    Plain HTTP over the standard library rather than the `litellm` package: the
    harness runs LiteLLM as a *sidecar container*, so importing it here would add
    a heavy host dependency for a feature the host does not otherwise need. Both
    shapes the harness can be pointed at speak JSON over one POST.

    Reads the same variables the agent container is given, so a run that can reach
    a model to migrate can reach one to write contracts. No new configuration.
    """
    import json
    import urllib.request

    base = (os.environ.get("HARNESS_TESTGEN_BASE_URL")
            or os.environ.get("LITELLM_API_BASE")
            or os.environ.get("ANTHROPIC_BASE_URL")
            or os.environ.get("ANTHROPIC_API_BASE")
            or os.environ.get("OPENAI_BASE_URL"))
    key = (os.environ.get("LITELLM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
           or os.environ.get("OPENAI_API_KEY"))
    if not (base or key):
        raise RuntimeError("no model endpoint configured; set ANTHROPIC_BASE_URL, "
                           "LITELLM_API_BASE or an API key")

    model = (os.environ.get("HARNESS_TESTGEN_MODEL")
             or os.environ.get("ANTHROPIC_MODEL_ID")
             or "claude-sonnet-4-5-20250929")
    anthropic = bool(os.environ.get("ANTHROPIC_API_KEY")
                     or os.environ.get("ANTHROPIC_BASE_URL")
                     or os.environ.get("ANTHROPIC_API_BASE"))
    base = (base or ("https://api.anthropic.com" if anthropic else "https://api.openai.com")).rstrip("/")

    if anthropic:
        url = f"{base}/v1/messages"
        headers = {"content-type": "application/json", "anthropic-version": "2023-06-01"}
        if key:
            headers["x-api-key"] = key
        body = {"model": model, "max_tokens": 4000,
                "messages": [{"role": "user", "content": prompt}]}
    else:
        url = f"{base}/v1/chat/completions"
        headers = {"content-type": "application/json"}
        if key:
            headers["authorization"] = f"Bearer {key}"
        body = {"model": model, "max_tokens": 4000,
                "messages": [{"role": "user", "content": prompt}]}

    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode())

    if anthropic:
        return "".join(block.get("text", "") for block in payload.get("content", []))
    return payload["choices"][0]["message"]["content"] or ""


def _extract_code(text: str) -> str:
    match = re.search(r"```(?:java)?\s*(.*?)```", text, re.S)
    return (match.group(1) if match else text).strip()


def _largest_source(image: str, repo: str, exec_fn) -> Tuple[str, str, str, str]:
    """
    Pick the source file to write a contract for, and locate its module.

    Largest first is crude, but the alternative is asking a model which file
    matters, which would put it one step closer to the reward.

    Returns:
        (path, package, module_dir, source) -- all empty when there is nothing
    """
    script = (
        f"cd {repo} && "
        "F=$(find . -path '*/src/main/java/*.java' -not -path '*/target/*' "
        "     -printf '%s\\t%p\\n' 2>/dev/null | sort -rn | head -1 | cut -f2); "
        'if [ -z "$F" ]; then echo NOSOURCE; exit 0; fi; '
        'echo "===PATH:$F"; '
        # The module is everything above src/main/java; the package is below it.
        'echo "===MODULE:$(echo "$F" | sed -E "s|/src/main/java/.*||")"; '
        'echo "===PKG:$(grep -m1 "^package " "$F" | sed -E "s/^package +//; s/;.*//")"; '
        'echo "===SOURCE"; cat "$F"'
    )
    out = (exec_fn(image, script).stdout or "")
    if "NOSOURCE" in out or "===SOURCE" not in out:
        return "", "", "", ""

    def field_of(name: str) -> str:
        m = re.search(rf"==={name}:(.*)", out)
        return m.group(1).strip() if m else ""

    return (field_of("PATH"), field_of("PKG"), field_of("MODULE"),
            out.split("===SOURCE", 1)[1].lstrip("\n"))


def generate_suite(image: str, repo: str = "/app/repo",
                   llm: Optional[Callable[[str], str]] = None,
                   exec_fn=None, max_source_chars: int = 12000) -> GeneratedSuite:
    """
    Ask a model for a contract suite describing the code at the base commit.

    Generated from the *original* source on purpose: the tests must encode the
    behaviour that already exists, not the behaviour the migration produced.
    """
    from harness.utils.gates import _exec

    exec_fn = exec_fn or _exec
    path, package, module, source = _largest_source(image, repo, exec_fn)
    if not source:
        return GeneratedSuite(detail="no main sources to write a contract for")

    prompt = _PROMPT.format(cls=CLASS_NAME, pkg=package or "com.example",
                            path=path, source=source[:max_source_chars])
    try:
        raw = (llm or default_llm)(prompt)
    except Exception as exc:  # noqa: BLE001 - an absent model is a skip, not a failure
        return GeneratedSuite(detail=f"model unavailable: {str(exc)[:160]}")

    code = _extract_code(raw)
    suite = GeneratedSuite(
        source=code, package=package, module=module,
        target=f"{module}/src/test/java/{package.replace('.', '/')}/{CLASS_NAME}.java",
    )
    if not suite.usable:
        return GeneratedSuite(detail="model returned nothing runnable")
    suite.detail = f"contract for {path} ({len(code.splitlines())} lines)"
    return suite


def run_suite(image: str, suite: GeneratedSuite, repo: str = "/app/repo",
              patch: Optional[Path] = None, timeout: int = 2400) -> SuiteRun:
    """
    Run the generated suite in one tree and report each test method's outcome.

    JUnit 5 has to be on the test classpath for this to compile at all. Rather
    than editing the pom -- which would change what is being measured -- the
    dependency is added only when the project does not already have it, inside
    the throwaway container.
    """
    from harness.utils.gates import _apply, _exec, _mounts

    # base64 rather than hex+xxd: xxd lives in vim-common and is absent from the
    # maven images, while base64 is coreutils and is always there.
    encoded = base64.b64encode(suite.source.encode("utf-8")).decode("ascii")
    junit = ("<dependency><groupId>org.junit.jupiter</groupId>"
             "<artifactId>junit-jupiter</artifactId><version>5.10.2</version>"
             "<scope>test</scope></dependency>")

    script = (
        f"cd {repo} && {_apply(patch)}"
        f"mkdir -p $(dirname {suite.target}) && "
        f"echo {encoded} | base64 -d > {suite.target} && "
        # Only add the dependency when it is genuinely missing, so a project that
        # already uses JUnit 5 is built exactly as it would be otherwise.
        f'P="{suite.module}/pom.xml"; '
        'grep -q "junit-jupiter" "$P" || '
        f'  sed -i "0,/<dependencies>/s||<dependencies>{junit}|" "$P"; '
        # -DfailIfNoTests=false is the one that matters on a multi-module build:
        # `-Dtest=` filters to a class that exists in exactly one module, so every
        # *other* module in the reactor runs zero tests and surefire fails the
        # whole build before the generated suite is ever reached. Measured on
        # gbif/name-parser, whose reactor builds name-parser-api first: the run
        # reported "did not compile" when the suite had compiled fine.
        # failIfNoSpecifiedTests is the newer spelling and does not cover this.
        f'mvn -q -B clean test -Dtest={CLASS_NAME} -DfailIfNoTests=false '
        '  -DfailIfNoSpecifiedTests=false -Dsurefire.failIfNoSpecifiedTests=false '
        '  >/dev/null 2>&1; echo "MVNRC=$?"; '
        f"F=$(find . -path '*/surefire-reports/*{CLASS_NAME}.xml' 2>/dev/null); "
        'if [ -z "$F" ]; then echo NOREPORT; exit 0; fi; '
        # Keep the failure and error lines too: surefire records them as children
        # of their testcase, so filtering to testcase lines alone makes every test
        # look like a pass -- measured, a deliberately wrong assertion reported
        # PASS until these were kept.
        "cat $F | tr '<' '\\n' | grep -E '^(testcase |failure|error|skipped)'"
    )
    result = _exec(image, script, _mounts(patch), timeout=timeout)
    out = result.stdout or ""

    if not out.strip():
        return SuiteRun(detail=f"could not run: {(result.stderr or '').strip()[:160]}")
    if "APPLYFAIL" in out:
        return SuiteRun(detail="patch does not apply")
    if "NOREPORT" in out:
        # The generated suite failing to compile is a fact about the generated
        # suite, not about the migration, so it must not read as a regression.
        return SuiteRun(detail="generated suite did not compile or produced no report")

    results = _parse_testcases(out)
    return SuiteRun(results=results, ran=True,
                    detail=f"{sum(results.values())}/{len(results)} passed")


def _parse_testcases(out: str) -> Dict[str, bool]:
    """
    Read per-method outcomes from surefire XML split on `<`.

    Surefire records a failure as a child element of its testcase, so order
    matters: a testcase is a pass until a failure or error line follows it.
    """
    results: Dict[str, bool] = {}
    current = None
    for line in out.splitlines():
        name = re.search(r'^testcase .*?name="([^"]+)"', line)
        if name:
            current = name.group(1)
            results[current] = True
        elif current and line.startswith(("failure", "error")):
            results[current] = False
    return results


def generated_tests_gate(base: SuiteRun, migrated: SuiteRun,
                         enforce: bool = False) -> Tuple[Optional[bool], str, dict]:
    """
    Compare the two runs, trusting only what the original tree confirmed.

    Args:
        base: the suite run against the base commit -- the oracle
        migrated: the same suite run against the migrated tree
        enforce: return a pass/fail verdict rather than reporting only. Off by
            default: no threshold for this has been derived on real migrations,
            and a gate nobody has calibrated should not zero a run.

    Returns:
        (verdict or None, detail, evidence)
    """
    if not base.ran:
        return None, f"no oracle: {base.detail}", {}
    oracle = base.passing
    if not oracle:
        return None, "no generated test passed on the original; nothing to trust", {}
    if not migrated.ran:
        return None, f"oracle of {len(oracle)}, but migrated run failed: {migrated.detail}", {}

    regressions = sorted(oracle - migrated.passing)
    preserved = len(oracle) - len(regressions)
    detail = (f"{preserved}/{len(oracle)} oracle behaviours preserved"
              + (f"; regressions: {', '.join(regressions[:4])}" if regressions else ""))
    evidence = {
        "oracle_tests": len(oracle),
        "preserved": preserved,
        "regressions": regressions[:20],
        "generated_total": len(base.results),
        "discarded_by_oracle": len(base.results) - len(oracle),
        "enforced": enforce,
    }
    if not enforce:
        return None, f"advisory: {detail}", evidence
    return (not regressions), detail, evidence
