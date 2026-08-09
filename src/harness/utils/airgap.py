"""
Airgapped agent networking: no route to the internet except an allowlist.

The repository under migration is public, so for anything migrated upstream the
answer is one `git clone` away. Hardening the local git history does nothing
about re-fetching it, and Harbor's own network policy cannot help on Docker
Desktop -- its egress control needs CONFIG_NFT_FIB_INET, which the LinuxKit
kernel does not compile in. This does the same job with plain Docker primitives,
so it works on any host::

        +-- jma-airgap (--internal: no external route) --+
        |                                                |
    [agent container]                        [egress proxy]  --bridge--> internet
     no route out                            [litellm]        (allowlisted hosts only)

**Strict** is the default and the stronger claim: nothing is started but the
network, so a host is unreachable because no route exists rather than because a
filter declined it. There is no regex to get wrong. Maven still resolves, because
the task image pre-fetches the Java-17 target versions at build time -- the
offline package store the reference corpus builds for pip and apt, in Maven form.

**Allowlist** adds a dual-homed tinyproxy for the case staging cannot cover: an
artifact the image did not pre-fetch. Two layers then apply -- bypassing the proxy
fails on routing, and using it to reach github.com fails on the filter. Prefer
widening the pre-fetch over widening the allowlist.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

DEFAULT_NETWORK = "jma-airgap"
DEFAULT_PROXY_CONTAINER = "jma-proxy"
DEFAULT_PROXY_IMAGE = "jma-egress-proxy:latest"
PROXY_PORT = 8888

#: Hosts the migration task legitimately needs. Maven Central, its mirror, and
#: the Spring milestone repo that several MigrationBench repos declare.
#: github.com is deliberately absent -- that is the point.
DEFAULT_ALLOWED_HOSTS = (
    "repo.maven.apache.org",
    "repo1.maven.org",
    "repo.spring.io",
    "central.sonatype.com",
)

#: FilterDefaultDeny makes anything unlisted a hard deny rather than a default
#: allow, and ConnectPort 443 keeps CONNECT from becoming a generic tunnel.
_TINYPROXY_CONF = """\
User tinyproxy
Group tinyproxy
Port {port}
Timeout 600
LogLevel Info
MaxClients 200
Allow 0.0.0.0/0
ConnectPort 443
Filter "/etc/tinyproxy/filter"
FilterDefaultDeny Yes
FilterExtended On
"""

_PROXY_DOCKERFILE = """\
FROM alpine:3.20
RUN apk add --no-cache tinyproxy
COPY tinyproxy.conf /etc/tinyproxy/tinyproxy.conf
COPY filter /etc/tinyproxy/filter
EXPOSE {port}
ENTRYPOINT ["tinyproxy", "-d", "-c", "/etc/tinyproxy/tinyproxy.conf"]
"""

#: Maven's resolver ignores the JVM's http.proxyHost properties -- it reads
#: settings.xml. Passing MAVEN_OPTS instead silently fails to route, and Maven
#: then cannot even resolve a plugin prefix.
#: `-s` replaces the global settings.xml rather than merging with it, so the
#: shared local repository has to be repeated here. Omitting it sends Maven back
#: to ~/.m2, where the image's staged artifacts are not.
_MAVEN_SETTINGS = """\
<settings xmlns="http://maven.apache.org/SETTINGS/1.0.0">
  <localRepository>/opt/m2</localRepository>
  <proxies>
    <proxy>
      <id>egress-http</id>
      <active>true</active>
      <protocol>http</protocol>
      <host>{host}</host>
      <port>{port}</port>
    </proxy>
    <proxy>
      <id>egress-https</id>
      <active>true</active>
      <protocol>https</protocol>
      <host>{host}</host>
      <port>{port}</port>
    </proxy>
  </proxies>
</settings>
"""


def _run(cmd: List[str], timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def build_proxy_image(
    image: str = DEFAULT_PROXY_IMAGE,
    allowed_hosts: tuple = DEFAULT_ALLOWED_HOSTS,
    force: bool = False,
) -> None:
    """Build the allowlisting forward proxy image."""
    exists = _run(["docker", "images", "-q", image]).stdout.strip()
    if exists and not force:
        return

    with tempfile.TemporaryDirectory(prefix="jma-proxy-") as tmp:
        ctx = Path(tmp)
        (ctx / "tinyproxy.conf").write_text(_TINYPROXY_CONF.format(port=PROXY_PORT))
        # Anchored, with an optional port: tinyproxy matches CONNECT against the
        # bare host but plain HTTP against `host:port`, so `^h$` alone silently
        # rejects every non-TLS request. Keeping the anchors matters -- an
        # unanchored pattern would also match `evil-repo.maven.apache.org.attacker.com`.
        (ctx / "filter").write_text(
            "\n".join(f"^{h.replace('.', chr(92) + '.')}(:[0-9]+)?$" for h in allowed_hosts) + "\n"
        )
        (ctx / "Dockerfile").write_text(_PROXY_DOCKERFILE.format(port=PROXY_PORT))

        logger.info("Building egress proxy %s (allowing %s)", image, ", ".join(allowed_hosts))
        built = _run(["docker", "build", "-t", image, str(ctx)], timeout=900)
        if built.returncode != 0:
            raise RuntimeError(f"Proxy image build failed:\n{built.stderr[-2000:]}")


@dataclass
class Airgap:
    """
    An internal Docker network, and optionally one allowlisted exit.

    Two modes, and `strict` is the default because it is the stronger claim.

    In **strict** mode nothing is started but the network. The agent has no route
    anywhere except the sidecars deliberately attached to it, so a host is
    unreachable because no path exists -- not because a filter declined it. There
    is no regex to get wrong and nothing to audit. This is the reference corpus's
    design: agent containers sit on an `--internal` bridge and the only
    dual-homed thing is the model gateway.

    In **allowlist** mode a tinyproxy sidecar is dual-homed and permits
    `allowed_hosts`. It exists for the case strict mode cannot serve: a migration
    that must resolve a Maven artifact absent from the image's warmed cache.
    Prefer widening the pre-fetch over widening the allowlist -- a package staged
    at build time costs nothing at run time and cannot be reached for anything
    else, which is the whole argument for an offline package store.

    Attributes:
        network: Internal network name; agent containers join only this.
        mode: `strict` (no exit) or `allowlist` (proxied exit).
        proxy_container: Dual-homed proxy, reachable from the network by name.
        allowed_hosts: Hosts the proxy will permit, in allowlist mode.
    """

    network: str = DEFAULT_NETWORK
    mode: str = "strict"
    proxy_container: str = DEFAULT_PROXY_CONTAINER
    proxy_image: str = DEFAULT_PROXY_IMAGE
    allowed_hosts: tuple = DEFAULT_ALLOWED_HOSTS
    _settings_dir: Optional[tempfile.TemporaryDirectory] = field(default=None, repr=False)

    @property
    def strict(self) -> bool:
        return self.mode == "strict"

    @property
    def proxy_url(self) -> Optional[str]:
        """None in strict mode: there is no exit to point anything at."""
        return None if self.strict else f"http://{self.proxy_container}:{PROXY_PORT}"

    @property
    def maven_settings_path(self) -> Optional[Path]:
        """
        Host path of the generated settings.xml, or None in strict mode.

        Maven ignores `-Dhttp.proxyHost` and reads settings.xml, so a proxy that
        is not written here is a proxy Maven will not use. In strict mode there is
        no proxy, and Maven resolves from the cache the image warmed at build.
        """
        if self.strict:
            return None
        if self._settings_dir is None:
            raise RuntimeError("Airgap not started")
        return Path(self._settings_dir.name) / "settings.xml"

    def _ensure_network(self) -> None:
        if _run(["docker", "network", "inspect", self.network]).returncode == 0:
            return
        logger.info("Creating internal network %s", self.network)
        created = _run(["docker", "network", "create", "--internal", self.network])
        if created.returncode != 0:
            raise RuntimeError(f"Could not create network: {created.stderr.strip()}")

    def attach(self, container: str) -> None:
        """
        Give a container a route out, by attaching it to a normal bridge too.

        For sidecars such as LiteLLM, which must reach a model endpoint that the
        allowlist does not cover.
        """
        _run(["docker", "network", "connect", "bridge", container])

    def start(self) -> "Airgap":
        self._ensure_network()
        if self.strict:
            logger.info("Airgap ready: %s, no exit (strict)", self.network)
            return self

        build_proxy_image(self.proxy_image, self.allowed_hosts)
        _run(["docker", "rm", "-f", self.proxy_container])

        # Start on the bridge (so the proxy itself has egress), then attach the
        # internal network so agents can reach it by name.
        started = _run(
            ["docker", "run", "-d", "--name", self.proxy_container,
             "--network", "bridge", self.proxy_image]
        )
        if started.returncode != 0:
            raise RuntimeError(f"Proxy failed to start: {started.stderr.strip()}")

        connected = _run(["docker", "network", "connect", self.network, self.proxy_container])
        if connected.returncode != 0:
            raise RuntimeError(f"Could not dual-home proxy: {connected.stderr.strip()}")

        self._settings_dir = tempfile.TemporaryDirectory(prefix="jma-mvn-")
        settings = Path(self._settings_dir.name) / "settings.xml"
        settings.write_text(
            _MAVEN_SETTINGS.format(host=self.proxy_container, port=PROXY_PORT)
        )
        settings.chmod(0o644)

        logger.info("Airgap ready: %s via %s", self.network, self.proxy_url)
        return self

    def stop(self) -> None:
        if not self.strict:
            _run(["docker", "rm", "-f", self.proxy_container])
        if self._settings_dir:
            self._settings_dir.cleanup()
            self._settings_dir = None
        logger.info("Airgap torn down")

    def __enter__(self) -> "Airgap":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
