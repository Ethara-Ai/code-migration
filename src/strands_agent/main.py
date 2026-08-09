"""Main entry point for running migration experiments."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from strands_agent.agents import BaselineAgent, HybridAgent, RAGAgent
from strands_agent.config.settings import Config, ModelConfig, AgentConfig, ExperimentConfig
from strands_agent.orchestrator import MigrationOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


AGENT_TYPES = {
    "baseline": BaselineAgent,
    "pe": BaselineAgent,
    "hybrid": HybridAgent,
    "rag": RAGAgent,
}


def _read_instruction(path: Optional[str]) -> Optional[str]:
    """
    Load the task's instruction file, or None.

    A missing or unreadable file is a warning rather than a failure: the built-in
    prompt still runs the task, and killing a run over a prompt file would trade a
    graded result for none. It is logged loudly because a task that ships an
    instruction and is silently graded on a different prompt is worse than either.
    """
    if not path:
        return None
    try:
        text = Path(path).read_text(encoding="utf-8").strip()
    except OSError as exc:
        logger.warning("Cannot read --instruction-file %s: %s; using the built-in "
                       "prompt", path, exc)
        return None
    if not text:
        logger.warning("--instruction-file %s is empty; using the built-in prompt", path)
        return None
    logger.info("Prompt: %s (%d chars)", path, len(text))
    return text


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run Java 8 to 17 migration experiments")

    parser.add_argument(
        "--agent-type",
        type=str,
        required=True,
        choices=list(AGENT_TYPES.keys()),
        help="Type of agent to use (baseline, pe, hybrid, or rag)",
    )

    parser.add_argument(
        "--exp-id",
        type=str,
        required=True,
        help="Experiment identifier for organizing results",
    )

    # Data source configuration
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default="AmazonScience/migration-bench-java-selected",
        help="HuggingFace dataset name. "
        "Options: AmazonScience/migration-bench-java-selected or AmazonScience/migration-bench-java-full",
    )

    # Model configuration
    parser.add_argument(
        "--model-provider",
        type=str,
        default="bedrock",
        choices=["bedrock", "anthropic", "openai"],
        help="Where to get the LLM. 'bedrock' is the paper's setup; 'anthropic' talks "
        "the Messages API, and --model-base-url points it at any compatible "
        "serves it from a Claude Code OAuth session instead of an AWS account.",
    )

    parser.add_argument(
        "--instruction-file",
        type=str,
        default=None,
        help="Path to the task's instruction.md inside the container. Its contents "
             "become the system prompt, so a task that ships one is graded against "
             "the prompt it ships rather than a built-in default.",
    )

    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Model ID. Defaults to the paper's Bedrock Sonnet for --model-provider "
        "bedrock, or claude-sonnet-4-6 for anthropic.",
    )

    parser.add_argument(
        "--model-base-url",
        type=str,
        default=None,
        help="Base URL for the anthropic provider. Falls back to "
        "ANTHROPIC_BASE_URL.",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Max output tokens per response (anthropic provider)",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Model temperature",
    )

    # Agent configuration
    parser.add_argument(
        "--max-messages",
        type=int,
        default=80,
        help="Maximum number of messages per conversation",
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum number of parallel workers",
    )

    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Migrate exactly one repository (e.g. 'owner/repo') instead of the whole "
        "dataset. This is the entry point used inside a per-repository container by "
        "runner/docker_runner.py.",
    )

    parser.add_argument(
        "--repo-path",
        type=str,
        default=None,
        help="Migrate a repository already present on disk, skipping the dataset "
        "lookup and clone. Used when the environment provisions the checkout -- a "
        "Harbor task, for instance. Grading needs --github-url too.",
    )

    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Do not grade inside the container. Use when the caller grades the "
        "patch separately -- MigrationBench needs GitHub, which an airgapped "
        "agent container must not have.",
    )

    parser.add_argument(
        "--github-url",
        type=str,
        default=None,
        help="GitHub URL for --repo-path, required by MigrationBench to evaluate.",
    )

    parser.add_argument(
        "--shell-timeout",
        type=int,
        default=300,
        help="Per-command timeout in seconds for the agent's shell tool. Raise it for "
        "containerized runs, where the first `mvn clean verify` populates a cold cache.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./migration_results",
        help="Directory for storing results",
    )

    args = parser.parse_args()

    default_model_id = {
        "bedrock": "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "anthropic": "claude-sonnet-4-6",
    }[args.model_provider]

    # Create configuration
    config = Config(
        model=ModelConfig(
            model_id=args.model_id or default_model_id,
            temperature=args.temperature,
            provider=args.model_provider,
            base_url=args.model_base_url,
            max_tokens=args.max_tokens,
        ),
        agent=AgentConfig(
            max_messages=args.max_messages,
            shell_timeout=args.shell_timeout,
            instruction=_read_instruction(args.instruction_file),
        ),
        experiment=ExperimentConfig(
            exp_id=args.exp_id,
            hf_dataset=args.hf_dataset,
            output_dir=args.output_dir,
            max_workers=args.max_workers,
        ),
    )

    # Get agent class
    agent_class = AGENT_TYPES[args.agent_type]

    # Create orchestrator and run
    orchestrator = MigrationOrchestrator(agent_class=agent_class, config=config)

    logger.info(f"Starting {args.agent_type} agent with experiment ID: {args.exp_id}")
    logger.info(f"Model: {config.model.model_id} via {args.model_provider}")
    logger.info(f"HuggingFace Dataset: {args.hf_dataset}")

    try:
        if args.repo_path:
            logger.info(f"Local-repository mode: {args.repo_path}")
            if not orchestrator.migrate_local_repo(
                args.repo_path, args.github_url, skip_eval=args.skip_eval
            ):
                logger.error(f"Migration run failed for {args.repo_path}")
                sys.exit(1)
        elif args.repo_id:
            logger.info(f"Single-repository mode: {args.repo_id}")
            if not orchestrator.migrate_single_repo(args.repo_id):
                logger.error(f"Migration run failed for {args.repo_id}")
                sys.exit(1)
        else:
            orchestrator.run_batch_migration()
        logger.info("Migration completed successfully")
    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
