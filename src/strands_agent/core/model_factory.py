"""Factory for creating LLM models."""

import logging
import os

from strands_agent.config.settings import ModelConfig

logger = logging.getLogger(__name__)


def create_bedrock_model(config: ModelConfig):
    """
    Create a Bedrock model from configuration.

    Args:
        config: Model configuration

    Returns:
        Configured BedrockModel instance
    """
    from botocore.config import Config as BotocoreConfig
    from strands.models import BedrockModel

    return BedrockModel(
        model_id=config.model_id,
        region_name=config.region_name,
        temperature=config.temperature,
        boto_client_config=BotocoreConfig(
            retries={"max_attempts": config.max_retries, "mode": "standard"}
        ),
        additional_request_fields={
            "thinking": {"type": "enabled", "budget_tokens": config.thinking_budget_tokens}
        },
    )


def create_anthropic_model(config: ModelConfig):
    """
    Create a model that talks the Anthropic Messages API.

    Points at api.anthropic.com by default, or at any compatible endpoint via
    base_url, so a run needs no Bedrock account.

    Args:
        config: Model configuration

    Returns:
        Configured AnthropicModel instance

    Raises:
        RuntimeError: If no API key is available
    """
    from strands.models.anthropic import AnthropicModel

    api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No Anthropic API key. Set ANTHROPIC_API_KEY, or pass api_key."
        )

    client_args = {"api_key": api_key}
    base_url = config.base_url or os.environ.get("ANTHROPIC_BASE_URL")
    if base_url:
        client_args["base_url"] = base_url
        logger.info(f"Anthropic endpoint: {base_url}")

    params = {"temperature": config.temperature}
    if config.thinking_budget_tokens:
        # Anthropic rejects temperature alongside extended thinking.
        params = {
            "thinking": {
                "type": "enabled",
                "budget_tokens": config.thinking_budget_tokens,
            }
        }

    return AnthropicModel(
        client_args=client_args,
        model_id=config.model_id,
        max_tokens=config.max_tokens,
        params=params,
    )


def create_openai_model(config: ModelConfig):
    """
    Create a model that talks the OpenAI Chat Completions API.

    This is how runs reach a LiteLLM proxy: LiteLLM's OpenAI-compatible surface
    is its primary interface, so whatever it is configured to route to upstream
    is invisible from here.

    Args:
        config: Model configuration

    Returns:
        Configured OpenAIModel instance

    Raises:
        RuntimeError: If no API key is available
    """
    from strands.models.openai import OpenAIModel

    api_key = config.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No API key. Set OPENAI_API_KEY -- for a LiteLLM proxy that is its "
            "master key, not an OpenAI key."
        )

    client_args = {"api_key": api_key}
    base_url = config.base_url or os.environ.get("OPENAI_BASE_URL")
    if base_url:
        client_args["base_url"] = base_url
        logger.info(f"OpenAI-compatible endpoint: {base_url}")

    return OpenAIModel(
        client_args=client_args,
        model_id=config.model_id,
        params={
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        },
    )


#: Provider name -> factory. `bedrock` is what the paper used.
MODEL_PROVIDERS = {
    "bedrock": create_bedrock_model,
    "anthropic": create_anthropic_model,
    "openai": create_openai_model,
}


def create_model(config: ModelConfig):
    """
    Create the model for the configured provider.

    Args:
        config: Model configuration

    Returns:
        A Strands model instance

    Raises:
        ValueError: If config.provider is not a known provider
    """
    factory = MODEL_PROVIDERS.get(config.provider)
    if factory is None:
        raise ValueError(
            f"Unknown model provider {config.provider!r}, "
            f"expected one of {sorted(MODEL_PROVIDERS)}"
        )
    logger.info(f"Model provider: {config.provider} ({config.model_id})")
    return factory(config)
