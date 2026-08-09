"""
Verify the migration with MigrationBench's own evaluator.

Deliberately not a hand-rolled check: `run_eval` is the same function the paper
reports numbers from, so a reward here means the same thing a MigrationBench
result means. It is graded at the *minimal* tier -- build passes under
Java 17, compiled classes are major version 61, test methods unchanged, test
count not reduced. Maximal (r5, every dependency at its latest major) is a
different task and would need the dependency reference pinned.
"""

import json
import pathlib
import re
import subprocess

REPO = pathlib.Path("/app/repo")
META = json.loads(pathlib.Path("/app/task_meta.json").read_text())


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout


def test_repository_still_present() -> None:
    """The agent must not have deleted or replaced the checkout."""
    assert REPO.is_dir(), "/app/repo is missing"
    assert (REPO / ".git").exists(), "/app/repo is no longer a git repository"
    assert list(REPO.rglob("pom.xml")), "no pom.xml left in the repository"


def test_agent_made_changes() -> None:
    """A migration that changed nothing cannot have happened."""
    diff = _git("diff", META["base_commit"])
    assert diff.strip(), "no changes relative to the base commit"


def test_history_was_not_used_to_cheat() -> None:
    """
    The environment strips post-base history so the upstream migration is not
    readable. Fail loudly if something restored it, because a pass would then
    mean nothing.
    """
    reachable = _git("rev-list", "--all", "--count").strip()
    from_head = _git("rev-list", "HEAD", "--count").strip()
    assert reachable == from_head, (
        f"{reachable} commits reachable but {from_head} from HEAD: "
        "post-base history was restored"
    )


def test_test_methods_preserved() -> None:
    """
    r3, done properly: no test method may disappear.

    MigrationBench's own r3 globs `<root>/src/test`, which finds nothing in a
    multi-module project where tests live in `<module>/src/test`. It then compares
    an empty list to an empty list and passes -- so on any multi-module repository
    the guard against deleting or renaming tests never fires. `selected` averages
    6.6 modules, so that is most of the benchmark.

    This walks every module instead. It is a superset check: r4 (test count via
    `mvn test`) catches outright deletion, but only a name comparison catches a
    rename that keeps the count the same.
    """
    method = re.compile(r"@Test\b")
    name = re.compile(r"(?:public|protected|private)?\s*\w[\w<>\[\], ]*\s+(\w+)\s*\(")

    def methods_at(ref: str | None) -> set[str]:
        """Test method names in the tree, at HEAD or at a given commit."""
        found: set[str] = set()
        if ref:
            listing = subprocess.run(
                ["git", "ls-tree", "-r", "--name-only", ref],
                cwd=REPO, capture_output=True, text=True, check=True,
            ).stdout.split()
            paths = [p for p in listing if "/src/test/" in p or p.startswith("src/test/")]
            read = lambda p: subprocess.run(  # noqa: E731
                ["git", "show", f"{ref}:{p}"], cwd=REPO,
                capture_output=True, text=True, check=True).stdout
        else:
            paths = [
                str(p.relative_to(REPO))
                for p in REPO.rglob("*.java")
                if "/src/test/" in str(p) and "/target/" not in str(p)
            ]
            read = lambda p: (REPO / p).read_text(encoding="utf-8", errors="ignore")  # noqa: E731

        for path in paths:
            if not path.endswith(".java"):
                continue
            lines = read(path).splitlines()
            for i, line in enumerate(lines):
                if not method.search(line):
                    continue
                for follow in lines[i + 1: i + 4]:
                    hit = name.search(follow)
                    if hit:
                        found.add(f"{pathlib.PurePosixPath(path).name}::{hit.group(1)}")
                        break
        return found

    before = methods_at(META["base_commit"])
    after = methods_at(None)

    if not before:
        # Nothing to protect; say so rather than passing silently the way
        # MigrationBench's own r3 does.
        print("no test methods at base commit; r3 has nothing to check")
        return

    missing = before - after
    assert not missing, f"test methods removed or renamed: {sorted(missing)}"


def test_minimal_migration_succeeds() -> None:
    """
    MigrationBench's r1-r3: builds under Java 17, classes are major version 61,
    test methods unchanged. r4 (test count) is not run -- the harness's
    `test_execution` gate subsumes it and is stricter.
    """
    from migration_bench.eval.final_eval import run_eval

    assert run_eval(
        github_url=META["github_url"],
        migrated_root_dir=str(REPO),
        require_maximal_migration=False,
    ), "MigrationBench minimal migration check failed"
