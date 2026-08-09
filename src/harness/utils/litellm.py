"""
LiteLLM proxy as a sidecar container.

One proxy serves every agent container on a private Docker network, so the agent
only ever knows one OpenAI-compatible endpoint and the choice of upstream --
which upstream serves them is a config concern.

This deliberately differs from the more common pattern, which
`pip install`s LiteLLM inside each task container and starts it with `docker exec
-d`. That costs roughly 90 seconds per container by its own comment and gives
every run its own proxy. A sidecar pays that once.

Nothing is published to the host: the proxy is reachable only at
``http://<container>:<port>`` from containers attached to the same network.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

DEFAULT_LITELLM_IMAGE = "ghcr.io/berriai/litellm:main-stable"
DEFAULT_PORT = 4000
READINESS_TIMEOUT = 120

#: The model name agents ask for. LiteLLM maps it to whichever upstream is
#: configured, so the agent's --model-id never has to change.
DEFAULT_MODEL_ALIAS = "claude-sonnet"


def _run(cmd: List[str], timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def build_model_list(
    *,
    alias: str = DEFAULT_MODEL_ALIAS,
    llm_api_base: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    upstream_model: str = "claude-sonnet-4-6",
) -> List[dict]:
    """
    Assemble LiteLLM's model_list from whatever credentials are present.

    Order matters: an explicit upstream wins, because on a machine with no cloud
    credentials it is the only upstream that can actually serve a request.

    Args:
        alias: Model name the agent will request
        llm_api_base: upstream base URL as seen from the proxy container
        llm_api_key: upstream credential
        upstream_model: Anthropic model id to ask the upstream for

    Returns:
        LiteLLM model_list entries; empty if nothing is configured
    """
    models: List[dict] = []

    if llm_api_base:
        models.append(
            {
                "model_name": alias,
                "litellm_params": {
                    "model": f"anthropic/{upstream_model}",
                    "api_base": llm_api_base,
                    "api_key": llm_api_key or "not-needed",
                },
            }
        )
        return models

    if os.environ.get("ANTHROPIC_API_KEY"):
        models.append(
            {
                "model_name": alias,
                "litellm_params": {
                    "model": f"anthropic/{upstream_model}",
                    "api_key": os.environ["ANTHROPIC_API_KEY"],
                },
            }
        )

    if os.environ.get("AWS_BEARER_TOKEN_BEDROCK") or os.environ.get("AWS_ACCESS_KEY_ID"):
        models.append(
            {
                "model_name": alias if not models else f"{alias}-bedrock",
                "litellm_params": {
                    "model": "bedrock/global.anthropic.claude-sonnet-4-5-20250929-v1:0",
                    "aws_region_name": os.environ.get("AWS_REGION", "us-west-2"),
                },
            }
        )

    if os.environ.get("OPENAI_API_KEY"):
        models.append(
            {
                "model_name": f"{alias}-openai",
                "litellm_params": {
                    "model": "gpt-5.5",
                    "api_key": os.environ["OPENAI_API_KEY"],
                },
            }
        )

    return models


def render_config(model_list: List[dict], master_key: str) -> str:
    """
    Render litellm's config file.

    YAML is a superset of JSON, and json.dumps quotes reliably, so this avoids a
    PyYAML dependency on the host just to write four keys.
    """
    return json.dumps(
        {
            "model_list": model_list,
            "general_settings": {"master_key": master_key},
            "litellm_settings": {"drop_params": True},
        },
        indent=2,
    )


@dataclass
class LiteLLMSidecar:
    """
    A LiteLLM proxy container, scoped to a `with` block.

    Attributes:
        network: Docker network the proxy and agents share
        model_list: LiteLLM model_list entries
        container: Container name, also the hostname agents dial
        image: LiteLLM image
        port: Proxy port inside the network
        master_key: Bearer token agents authenticate with
    """

    network: str
    model_list: List[dict]
    container: str = "jma-litellm"
    image: str = DEFAULT_LITELLM_IMAGE
    port: int = DEFAULT_PORT
    master_key: str = field(default_factory=lambda: f"sk-{secrets.token_hex(16)}")
    _tmpdir: Optional[tempfile.TemporaryDirectory] = field(default=None, repr=False)

    @property
    def base_url(self) -> str:
        """Endpoint as seen by containers on the same network."""
        return f"http://{self.container}:{self.port}/v1"

    def ensure_network(self) -> None:
        if _run(["docker", "network", "inspect", self.network]).returncode != 0:
            logger.info("Creating network %s", self.network)
            created = _run(["docker", "network", "create", self.network])
            if created.returncode != 0:
                raise RuntimeError(f"Could not create network: {created.stderr.strip()}")

    def start(self) -> "LiteLLMSidecar":
        if not self.model_list:
            raise RuntimeError(
                "No LiteLLM upstream configured. Pass --llm-api-base, or set "
                "ANTHROPIC_API_KEY / AWS credentials / OPENAI_API_KEY."
            )

        self.ensure_network()
        _run(["docker", "rm", "-f", self.container])

        self._tmpdir = tempfile.TemporaryDirectory(prefix="jma-litellm-")
        config_path = Path(self._tmpdir.name) / "config.yaml"
        config_path.write_text(render_config(self.model_list, self.master_key))
        config_path.chmod(0o644)

        logger.info(
            "Starting LiteLLM sidecar %s on %s (models: %s)",
            self.container,
            self.network,
            ", ".join(m["model_name"] for m in self.model_list),
        )
        started = _run(
            [
                "docker", "run", "-d",
                "--name", self.container,
                "--network", self.network,
                # host-gateway lets the proxy reach a bridge running on the host.
                "--add-host", "host.docker.internal:host-gateway",
                "-v", f"{config_path}:/app/config.yaml:ro",
                self.image,
                "--config", "/app/config.yaml",
                "--port", str(self.port),
            ]
        )
        if started.returncode != 0:
            raise RuntimeError(f"LiteLLM failed to start: {started.stderr.strip()}")

        self._await_ready()
        return self

    def _await_ready(self) -> None:
        """Poll /health/readiness from inside the container (no host port needed)."""
        probe = (
            "import urllib.request,sys;"
            f"sys.exit(0 if urllib.request.urlopen("
            f"'http://localhost:{self.port}/health/readiness', timeout=2)"
            f".status==200 else 1)"
        )
        deadline = time.monotonic() + READINESS_TIMEOUT
        while time.monotonic() < deadline:
            if _run(["docker", "exec", self.container, "python", "-c", probe]).returncode == 0:
                logger.info("LiteLLM ready at %s", self.base_url)
                return
            if _run(["docker", "inspect", "-f", "{{.State.Running}}", self.container]).stdout.strip() != "true":
                logs = _run(["docker", "logs", "--tail", "40", self.container])
                raise RuntimeError(f"LiteLLM exited during startup:\n{logs.stdout}{logs.stderr}")
            time.sleep(2)

        logs = _run(["docker", "logs", "--tail", "40", self.container])
        raise RuntimeError(
            f"LiteLLM not ready after {READINESS_TIMEOUT}s:\n{logs.stdout}{logs.stderr}"
        )

    def stop(self) -> None:
        _run(["docker", "rm", "-f", self.container])
        if self._tmpdir:
            self._tmpdir.cleanup()
            self._tmpdir = None
        logger.info("LiteLLM sidecar stopped")

    def __enter__(self) -> "LiteLLMSidecar":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
