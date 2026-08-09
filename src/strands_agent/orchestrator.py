"""Orchestrator for running migration experiments."""

import concurrent.futures
import logging
import os
from typing import Optional, Type

from strands_agent.config.settings import Config
from strands_agent.core.base_agent import BaseMigrationAgent
from strands_agent.core.repository import Repository
from strands_agent.utils.hf_utils import get_repo_ids_from_dataset

logger = logging.getLogger(__name__)


class MigrationOrchestrator:
    """Orchestrates batch migration experiments."""

    def __init__(
        self,
        agent_class: Type[BaseMigrationAgent],
        config: Config,
    ):
        """
        Initialize the orchestrator.

        Args:
            agent_class: The agent class to use (BaselineAgent, HybridAgent, etc.)
            config: Configuration object
        """
        self.agent_class = agent_class
        self.config = config

        if not self.config.experiment:
            raise ValueError("Experiment configuration is required")

    def migrate_single_repo(self, repo_id: str) -> bool:
        """
        Migrate a single repository from HuggingFace dataset.

        Args:
            repo_id: Repository identifier

        Returns:
            True if the repository was migrated and its results written. A False
            here means the run itself broke, not that the migration was graded
            unsuccessful -- the verdicts live in the results JSON.
        """
        try:
            # Load repository from HuggingFace
            repository = Repository.from_huggingface(
                dataset_name=self.config.experiment.hf_dataset,
                repo_id=repo_id,
                exp_id=self.config.experiment.exp_id,
            )
            logger.info(f"Loaded repository: {repo_id} at {repository.path}")

            # Create and run agent
            agent = self.agent_class(self.config)
            results = agent.migrate(repository)

            # Save results
            agent.save_results(repo_id, results)
            self._save_diff(repo_id, repository)

            logger.info(
                f"Completed {repo_id}: max_success={results['max_success']}, "
                f"min_success={results['min_success']}"
            )

            # An agent that produced nothing at all did not "migrate unsuccessfully",
            # it never ran -- usually bad credentials or an unreachable endpoint.
            # Report that as a failed run so the caller can retry rather than
            # recording a False verdict it would have to trust.
            if results.get("error") and not results.get("messages"):
                logger.error(
                    f"Agent produced no output for {repo_id}: {results['error']}"
                )
                return False

            return True

        except Exception as e:
            logger.error(f"Error migrating repository {repo_id}: {e}", exc_info=True)
            return False

    def migrate_local_repo(
        self, repo_path: str, github_url: Optional[str] = None,
        skip_eval: bool = False,
    ) -> bool:
        """
        Migrate a checkout that already exists on disk.

        For environments that provision the repository themselves -- a Harbor
        task, for example -- so the agent neither looks up the dataset nor clones.

        Args:
            repo_path: Path to the existing checkout
            github_url: GitHub URL, which MigrationBench needs to evaluate

        Returns:
            True if the migration ran and its results were written
        """
        try:
            repository = Repository.from_local(repo_path)
            if github_url:
                repository.github_url = github_url
                # from_local names the repo after its directory ("/app/repo" ->
                # "repo"), which would make every task's artifacts collide under
                # one filename. The URL carries the real owner/repo.
                owner_repo = github_url.rstrip("/").removesuffix(".git")
                repository.repo_id = "/".join(owner_repo.split("/")[-2:])
            if not repository.github_url:
                logger.warning(
                    "No github_url for %s: MigrationBench cannot grade it and both "
                    "verdicts will be False",
                    repo_path,
                )

            logger.info(
                f"Migrating local repository {repository.repo_id} at {repo_path} "
                f"(base {repository.base_commit})"
            )

            agent = self.agent_class(self.config)
            results = agent.migrate(repository, skip_eval=skip_eval)
            agent.save_results(repository.repo_id, results)
            self._save_diff(repository.repo_id, repository)

            logger.info(
                f"Completed {repository.repo_id}: max_success={results['max_success']}, "
                f"min_success={results['min_success']}"
            )

            if results.get("error") and not results.get("messages"):
                logger.error(f"Agent produced no output: {results['error']}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error migrating {repo_path}: {e}", exc_info=True)
            return False

    def _save_diff(self, repo_id: str, repository: Repository) -> None:
        """
        Write the migration patch next to the results JSON.

        save_results() records the verdicts and the trajectory but not the change
        itself, so without this the only copy of the patch dies with the workspace.

        Args:
            repo_id: Repository identifier
            repository: The migrated repository
        """
        if not self.config.experiment:
            return

        output_dir = os.path.join(
            self.config.experiment.output_dir, self.config.experiment.exp_id
        )
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, repo_id.replace("/", "__") + ".diff")

        try:
            with open(filepath, "w") as f:
                f.write(repository.generate_diff())
            logger.info(f"Diff saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save diff for {repo_id}: {e}")

    def run_batch_migration(self, max_workers: Optional[int] = None) -> None:
        """
        Run migration on all repositories from HuggingFace dataset.

        Args:
            max_workers: Maximum number of parallel workers (defaults to config value)
        """
        if max_workers is None:
            max_workers = self.config.experiment.max_workers

        # Get repository IDs from HuggingFace dataset
        repo_ids = get_repo_ids_from_dataset(self.config.experiment.hf_dataset)
        logger.info(
            f"Starting batch migration for {len(repo_ids)} repositories "
            f"from HuggingFace dataset: {self.config.experiment.hf_dataset}"
        )

        # Process in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.migrate_single_repo, repo_id): repo_id
                for repo_id in repo_ids
            }

            for future in concurrent.futures.as_completed(futures):
                repo_id = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error processing repository {repo_id}: {e}")

        logger.info("Batch migration completed")
