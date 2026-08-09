"""Repository management for Java migration."""

import logging
import os
import shutil
import subprocess
import tarfile
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Repository:
    """Represents a Java repository to be migrated."""

    path: str
    repo_id: str
    github_url: Optional[str] = None
    base_commit: Optional[str] = None


    @classmethod
    def from_huggingface(cls, dataset_name: str, repo_id: str, exp_id: str) -> "Repository":
        """
        Load a repository from a HuggingFace dataset by cloning from GitHub.

        Args:
            dataset_name: HuggingFace dataset name (e.g., "AmazonScience/migration-bench-java-selected")
            repo_id: Repository identifier from the dataset (format: "owner/repo")
            exp_id: Experiment identifier for workspace organization

        Returns:
            Repository object
        """
        repo_path = _load_repo_from_huggingface(dataset_name, repo_id, exp_id)

        # Build GitHub URL from repo_id (format: "owner/repo")
        github_url = f"https://github.com/{repo_id}"

        # Get base commit (already checked out in _load_repo_from_huggingface)
        base_commit = _get_base_commit_id(repo_path)

        return cls(
            path=repo_path,
            repo_id=repo_id,
            github_url=github_url,
            base_commit=base_commit,
        )

    @classmethod
    def from_local(cls, path: str, repo_id: Optional[str] = None) -> "Repository":
        """
        Load a repository from a local path.

        Args:
            path: Local path to repository
            repo_id: Optional repository identifier (defaults to directory name)

        Returns:
            Repository object
        """
        if repo_id is None:
            repo_id = os.path.basename(os.path.abspath(path))

        base_commit = _get_base_commit_id(path)

        return cls(
            path=path,
            repo_id=repo_id,
            base_commit=base_commit,
        )

    def generate_diff(self) -> str:
        """
        Generate a git diff showing all changes since base_commit.

        Build output is excluded. Some repositories commit `target/`, so a plain
        diff carries binary .class hunks -- and a patch containing those cannot be
        applied by `git apply` ("cannot apply binary patch without full index
        line"). That makes the diff unusable both as a MigrationBench input and as
        a task oracle. They are derived artifacts that `mvn clean verify`
        regenerates, so excluding them changes nothing that is graded.

        Returns:
            Diff output string
        """
        if not self.base_commit:
            return "Error: No base commit available"

        try:
            result = subprocess.run(
                ["git", "diff", self.base_commit, "--",
                 ".", ":(exclude)target/*", ":(exclude)*/target/*"],
                cwd=self.path,
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            return f"Error generating diff: {e}"


def _get_base_commit_id(repo_path: str) -> Optional[str]:
    """
    Get the current HEAD commit ID.

    Args:
        repo_path: Path to repository

    Returns:
        Commit ID or None if error
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None


def _harden_git_history(repo_path: str, base_commit: str) -> None:
    """
    Make every commit that is not an ancestor of base_commit unreachable.

    `git clone` fetches the whole history, so for any repository that was later
    migrated to Java 17 upstream the answer is still sitting in `git log --all`
    after we check out the base commit -- and the agent has shell access. Detach
    at the base commit, drop all other refs, remotes and reflogs, then verify
    that nothing but the base commit's ancestry survived.

    Args:
        repo_path: Path to the cloned repository
        base_commit: The commit the agent is supposed to start from

    Raises:
        ValueError: If the repository is not left at exactly base_commit's history
    """

    def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=check,
        )

    logger.info(f"Hardening git history at {base_commit}")

    git("checkout", "--detach", base_commit)

    for remote in git("remote").stdout.split():
        git("remote", "remove", remote, check=False)

    for ref in git("for-each-ref", "--format=%(refname)").stdout.split():
        git("update-ref", "-d", ref, check=False)

    git("reflog", "expire", "--expire=now", "--all", check=False)
    git("gc", "--prune=now", check=False)

    head = git("rev-parse", "HEAD").stdout.strip()
    if head != base_commit:
        raise ValueError(f"HEAD is {head} after hardening, expected {base_commit}")

    # `--all` covers every remaining ref plus detached HEAD, so once the other
    # refs are gone it should describe exactly the base commit's ancestry.
    reachable = git("rev-list", "--all", "--count").stdout.strip()
    from_head = git("rev-list", "HEAD", "--count").stdout.strip()
    if reachable != from_head:
        raise ValueError(
            f"{reachable} commits reachable but only {from_head} from HEAD: "
            "history hardening left future commits in place"
        )

    logger.info(f"History hardened: {from_head} commits, all ancestors of {base_commit}")


def _load_repo_from_huggingface(dataset_name: str, repo_id: str, exp_id: str) -> str:
    """
    Download a repository from GitHub based on HuggingFace dataset metadata.

    Args:
        dataset_name: HuggingFace dataset name (e.g., "AmazonScience/migration-bench-java-selected")
        repo_id: Repository identifier (e.g., "owner/repo" or "15093015999/EJServer")
        exp_id: Experiment identifier for workspace organization

    Returns:
        Local path to cloned repository
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required to load from HuggingFace. "
            "Install it with: pip install datasets"
        )

    logger.info(f"Loading dataset {dataset_name} from HuggingFace")

    # Load dataset from test split
    dataset = load_dataset(dataset_name, split="test")

    # Find the repo in the dataset by matching the "repo" column
    repo_data = None
    for item in dataset:
        # The column is called "repo" and contains values like "15093015999/EJServer"
        if item.get("repo") == repo_id:
            repo_data = item
            break

    if repo_data is None:
        raise ValueError(f"Repository {repo_id} not found in dataset {dataset_name}")

    # Parse GitHub URL from repo field (format: "owner/repo")
    github_url = f"https://github.com/{repo_id}.git"

    # Get base_commit from dataset metadata
    base_commit = repo_data.get("base_commit")
    if not base_commit:
        logger.warning(f"No base_commit found in dataset for {repo_id}")

    # Set up workspace
    workdir = f"/tmp/workdir/{exp_id}"
    os.makedirs(workdir, exist_ok=True)

    # Extract repository name from repo_id (e.g., "EJServer" from "15093015999/EJServer")
    repo_name = repo_id.split("/")[-1]
    repo_path = os.path.join(workdir, repo_name)

    # Clean up if it exists
    if os.path.exists(repo_path):
        shutil.rmtree(repo_path)

    # Clone repository from GitHub
    logger.info(f"Cloning repository from {github_url}")
    try:
        subprocess.run(
            ["git", "clone", github_url, repo_path],
            capture_output=True,
            text=True,
            check=True,
            timeout=300,
        )
        logger.info(f"Successfully cloned {repo_id}")
    except subprocess.CalledProcessError as e:
        raise ValueError(f"Failed to clone repository {github_url}: {e.stderr}")
    except subprocess.TimeoutExpired:
        raise ValueError(f"Timeout while cloning repository {github_url}")

    if not os.path.exists(repo_path):
        raise ValueError(f"{repo_path} does not exist after cloning")

    # Checkout the base commit if available
    if base_commit:
        logger.info(f"Checking out base_commit: {base_commit}")
        try:
            subprocess.run(
                ["git", "checkout", base_commit],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info(f"Successfully checked out base_commit: {base_commit}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to checkout base_commit {base_commit}: {e.stderr}")
            raise ValueError(f"Failed to checkout base_commit {base_commit}")

        _harden_git_history(repo_path, base_commit)
    else:
        logger.warning(
            "No base_commit for %s: skipping history hardening, the full upstream "
            "history stays reachable from the working tree",
            repo_id,
        )

    return repo_path
