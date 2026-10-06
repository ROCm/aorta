"""Building an LM must not change TLS verification for the process.

``aorta.cia.llm`` used to set SSL_CERT_FILE and REQUESTS_CA_BUNDLE to the
certifi bundle when it built an LM. Those variables are process-global, so
everything imported afterwards -- unrelated aorta code, the chat provider
layer, anything the user imported alongside -- verified against certifi
instead of the system store, without asking and without saying so.

Certifi is right for a site behind a TLS interception proxy whose CA certifi
knows. It is wrong for the opposite site, whose corporate CA is in the system
store and not in certifi, and there it breaks TLS that worked before the
import. The choice now lives on the provider client's TLS context.
"""

from __future__ import annotations

import inspect
import json
import os
import pathlib
import shutil
import ssl
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from aorta.cia import llm as llm_mod

CA_VARS = ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE")


@pytest.fixture
def no_ca_env(monkeypatch):
    for var in (*CA_VARS, "SSL_CERT_DIR", "CIA_SSL_USE_CERTIFI"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def lm_built(monkeypatch):
    """Build an LM without reaching a network."""
    seen = {}
    sync_client = object()
    async_client = object()
    async_builds = []
    monkeypatch.setattr(llm_mod.dspy, "configure", lambda **kw: None)
    monkeypatch.setattr(llm_mod, "_configured", False)
    monkeypatch.setattr(
        llm_mod,
        "chat_provider",
        lambda **_k: ("http://vllm:8000/v1", "EMPTY", "unused", "vllm"),
    )

    def build_clients(**kwargs):
        seen.update(kwargs)

        def build_async_client():
            async_builds.append(True)
            return async_client

        return sync_client, build_async_client

    monkeypatch.setattr(llm_mod, "_openai_clients", build_clients)
    seen["sync_client"] = sync_client
    seen["async_client"] = async_client
    seen["async_builds"] = async_builds
    return seen


def _ca_env() -> dict[str, str | None]:
    return {var: os.environ.get(var) for var in CA_VARS}


@pytest.fixture
def private_ca_server(tmp_path):
    """An OpenAI-compatible HTTPS endpoint signed by a test-only private CA."""
    openssl = shutil.which("openssl")
    if openssl is None:
        pytest.skip("the private-CA transport test needs openssl")

    ca_key = tmp_path / "ca.key"
    ca_cert = tmp_path / "ca.pem"
    server_key = tmp_path / "server.key"
    server_csr = tmp_path / "server.csr"
    server_cert = tmp_path / "server.pem"
    server_ext = tmp_path / "server.ext"

    def run_openssl(*args):
        subprocess.run(
            [openssl, *args],
            check=True,
            capture_output=True,
            text=True,
        )

    run_openssl(
        "req",
        "-x509",
        "-newkey",
        "rsa:2048",
        "-nodes",
        "-sha256",
        "-days",
        "1",
        "-subj",
        "/CN=AORTA test CA",
        "-addext",
        "basicConstraints=critical,CA:TRUE",
        "-addext",
        "keyUsage=critical,keyCertSign,cRLSign",
        "-keyout",
        str(ca_key),
        "-out",
        str(ca_cert),
    )
    run_openssl(
        "req",
        "-newkey",
        "rsa:2048",
        "-nodes",
        "-sha256",
        "-subj",
        "/CN=127.0.0.1",
        "-keyout",
        str(server_key),
        "-out",
        str(server_csr),
    )
    server_ext.write_text(
        "subjectAltName=IP:127.0.0.1\n"
        "basicConstraints=critical,CA:FALSE\n"
        "keyUsage=critical,digitalSignature,keyEncipherment\n"
        "extendedKeyUsage=serverAuth\n",
        encoding="utf-8",
    )
    run_openssl(
        "x509",
        "-req",
        "-in",
        str(server_csr),
        "-CA",
        str(ca_cert),
        "-CAkey",
        str(ca_key),
        "-CAcreateserial",
        "-days",
        "1",
        "-sha256",
        "-extfile",
        str(server_ext),
        "-out",
        str(server_cert),
    )

    bodies = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length))
            bodies.append(body)
            if "ssl_verify" in body:
                payload = {"error": "transport settings are not request fields"}
                status = 400
            else:
                payload = {
                    "id": "chatcmpl-private-ca",
                    "object": "chat.completion",
                    "created": 0,
                    "model": body.get("model", "test-model"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "private-ca-ok",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                }
                status = 200
            encoded = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, _format, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    tls.load_cert_chain(server_cert, server_key)
    server.socket = tls.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield (
            f"https://127.0.0.1:{server.server_address[1]}/v1",
            ca_cert,
            bodies,
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


class TestImportingIsInert:
    def test_importing_the_module_sets_no_ca_variable(self):
        """In a fresh interpreter, so an earlier import cannot mask it."""
        code = (
            "import os;"
            "import aorta.cia.llm;"
            "print(bool(os.environ.get('SSL_CERT_FILE') or "
            "os.environ.get('REQUESTS_CA_BUNDLE')))"
        )
        env = {k: v for k, v in os.environ.items() if k not in CA_VARS}
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
        assert out.stdout.strip().endswith("False"), out.stdout + out.stderr

    def test_the_module_has_no_import_time_side_effect_in_its_source(self):
        header = inspect.getsource(llm_mod).split("def ")[0]
        assert "os.environ.setdefault" not in header
        assert "certifi.where()" not in header
        assert "os.environ[" not in header


class TestBuildingLeavesTheProcessAlone:
    def test_building_an_lm_does_not_write_ca_variables(self, no_ca_env, lm_built):
        before = _ca_env()
        lm = llm_mod.build_lm()
        assert _ca_env() == before
        assert all(value is None for value in before.values())
        assert "certifi" in str(lm_built["verify"])
        assert "ssl_verify" not in lm.kwargs
        assert lm._sync_client is lm_built["sync_client"]
        assert lm._async_client is None
        assert lm_built["async_builds"] == []

    def test_an_existing_setting_is_somebody_having_decided(self, no_ca_env, lm_built, monkeypatch):
        monkeypatch.setenv("SSL_CERT_FILE", "/etc/ssl/corporate.pem")
        before = _ca_env()
        lm = llm_mod.build_lm()
        assert _ca_env() == before
        assert lm_built["verify"] == "/etc/ssl/corporate.pem"
        assert "ssl_verify" not in lm.kwargs

    def test_the_other_variable_alone_also_counts_as_decided(
        self, no_ca_env, lm_built, monkeypatch
    ):
        monkeypatch.setenv("REQUESTS_CA_BUNDLE", "/etc/ssl/corporate.pem")
        before = _ca_env()
        lm = llm_mod.build_lm()
        assert _ca_env() == before
        assert os.environ.get("SSL_CERT_FILE") is None
        assert lm_built["verify"] == "/etc/ssl/corporate.pem"
        assert "ssl_verify" not in lm.kwargs

    def test_it_can_use_system_trust_for_the_site_certifi_would_break(
        self, no_ca_env, lm_built, monkeypatch
    ):
        monkeypatch.setenv("CIA_SSL_USE_CERTIFI", "0")
        before = _ca_env()
        lm = llm_mod.build_lm()
        assert _ca_env() == before
        assert lm_built["verify"] is True
        assert "ssl_verify" not in lm.kwargs

    def test_a_missing_certifi_is_not_a_crash(self, no_ca_env, lm_built, monkeypatch):
        """certifi arrived transitively; it may not be there at all."""
        real_import = __import__

        def no_certifi(name, *args, **kwargs):
            if name == "certifi":
                raise ImportError("no certifi here")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", no_certifi)
        before = _ca_env()
        lm = llm_mod.build_lm()
        assert _ca_env() == before
        assert lm_built["verify"] is True
        assert "ssl_verify" not in lm.kwargs

    def test_build_lm_never_assigns_the_process_ca_variables(self):
        source = inspect.getsource(llm_mod.build_lm) + inspect.getsource(llm_mod._ssl_verify)
        assert "os.environ[" not in source
        assert "os.environ.setdefault" not in source


class TestEachCallGetsTheMatchingProviderClient:
    def test_sync_calls_receive_the_sync_client(self, monkeypatch):
        sync_client = object()
        seen = {}

        def forward(_self, prompt=None, messages=None, **kwargs):
            seen.update(kwargs)
            return []

        monkeypatch.setattr(llm_mod.dspy.LM, "forward", forward)
        lm = llm_mod.RedactingLM(model="openai/test", sync_client=sync_client)

        lm.forward(messages=[{"role": "user", "content": "hello"}])

        assert seen["client"] is sync_client

    async def test_async_calls_receive_the_async_client(self, monkeypatch):
        async_client = object()
        seen = {}

        async def aforward(_self, prompt=None, messages=None, **kwargs):
            seen.update(kwargs)
            return []

        monkeypatch.setattr(llm_mod.dspy.LM, "aforward", aforward)
        lm = llm_mod.RedactingLM(model="openai/test", async_client=async_client)

        await lm.aforward(messages=[{"role": "user", "content": "hello"}])

        assert seen["client"] is async_client


class TestProviderClientLifecycle:
    def test_close_releases_the_sync_client_once(self):
        class SyncClient:
            closes = 0

            def close(self):
                self.closes += 1

        sync_client = SyncClient()
        lm = llm_mod.RedactingLM(model="openai/test", sync_client=sync_client)

        lm.close()
        lm.close()

        assert sync_client.closes == 1
        assert lm._sync_client is None

    async def test_async_client_is_lazy_and_aclose_releases_both(self, monkeypatch):
        class SyncClient:
            closes = 0

            def close(self):
                self.closes += 1

        class AsyncClient:
            closes = 0

            async def close(self):
                self.closes += 1

        sync_client = SyncClient()
        async_client = AsyncClient()
        builds = []
        seen = []

        async def aforward(_self, prompt=None, messages=None, **kwargs):
            seen.append(kwargs["client"])
            return []

        monkeypatch.setattr(llm_mod.dspy.LM, "aforward", aforward)
        lm = llm_mod.RedactingLM(
            model="openai/test",
            sync_client=sync_client,
            async_client_factory=lambda: (builds.append(True) or async_client),
        )
        assert builds == []

        await lm.aforward(messages=[{"role": "user", "content": "one"}])
        await lm.aforward(messages=[{"role": "user", "content": "two"}])

        assert builds == [True]
        assert seen == [async_client, async_client]

        await lm.aclose()
        await lm.aclose()

        assert sync_client.closes == 1
        assert async_client.closes == 1
        assert lm._sync_client is None
        assert lm._async_client is None


class TestCustomCAReachesTheTransport:
    def test_openai_route_completes_real_https_without_leaking_ca_into_json(
        self,
        no_ca_env,
        private_ca_server,
        monkeypatch,
    ):
        api_base, ca_cert, bodies = private_ca_server
        monkeypatch.setenv("SSL_CERT_FILE", str(ca_cert))
        monkeypatch.setenv("NO_PROXY", "127.0.0.1")
        monkeypatch.setattr(llm_mod, "chat_provider", lambda **_kwargs: None)
        monkeypatch.setattr(llm_mod, "_legacy_env", lambda: None)
        before = _ca_env()

        lm = llm_mod.build_lm(
            model="test-model",
            api_base=api_base,
            api_key="private-ca-key",
            max_tokens=8,
        )
        lm.num_retries = 0
        try:
            result = lm.forward(
                messages=[{"role": "user", "content": "hello"}],
            )
        finally:
            lm.close()

        assert result.choices[0].message.content == "private-ca-ok"
        assert bodies and "ssl_verify" not in bodies[0]
        assert _ca_env() == before

    def test_cia_sends_qwen_without_thinking(
        self,
        no_ca_env,
        private_ca_server,
        monkeypatch,
    ):
        api_base, ca_cert, bodies = private_ca_server
        monkeypatch.setenv("SSL_CERT_FILE", str(ca_cert))
        monkeypatch.setenv("NO_PROXY", "127.0.0.1")
        monkeypatch.setattr(llm_mod, "chat_provider", lambda **_kwargs: None)
        monkeypatch.setattr(llm_mod, "_legacy_env", lambda: None)

        lm = llm_mod.build_cia_lm(
            api_base=api_base,
            api_key="cia-qwen-key",
            max_tokens=8,
        )
        lm.num_retries = 0
        try:
            result = lm.forward(
                messages=[{"role": "user", "content": "hello"}],
            )
        finally:
            lm.close()

        assert result.choices[0].message.content == "private-ca-ok"
        assert len(bodies) == 1
        assert bodies[0]["model"] == llm_mod.CIA_MODEL
        assert bodies[0]["chat_template_kwargs"] == {
            "enable_thinking": False,
        }

    def test_system_mode_uses_openssl_default_trust_not_certifi(
        self,
        no_ca_env,
        private_ca_server,
        monkeypatch,
        tmp_path,
    ):
        import certifi
        import httpx

        api_base, ca_cert, bodies = private_ca_server
        request_url = f"{api_base}/chat/completions"

        # The private CA is not in certifi, so this is a real distinction rather
        # than two ways of constructing the same trust source.
        certifi_context = ssl.create_default_context(cafile=certifi.where())
        with httpx.Client(verify=certifi_context, trust_env=False) as client:
            with pytest.raises(httpx.ConnectError):
                client.post(
                    request_url,
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                )
        assert bodies == []

        # OpenSSL's system trust uses a hashed certificate directory. Point its
        # default path at a directory containing only this CA; do not provide
        # either explicit CA-file variable that CIA treats as a custom bundle.
        openssl = shutil.which("openssl")
        assert openssl is not None
        cert_hash = subprocess.run(
            [openssl, "x509", "-hash", "-noout", "-in", str(ca_cert)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        trust_dir = tmp_path / "system-trust"
        trust_dir.mkdir()
        shutil.copyfile(ca_cert, trust_dir / f"{cert_hash}.0")

        monkeypatch.setenv("CIA_SSL_USE_CERTIFI", "0")
        monkeypatch.setenv("SSL_CERT_DIR", str(trust_dir))
        monkeypatch.setenv("NO_PROXY", "127.0.0.1")
        monkeypatch.setattr(llm_mod, "chat_provider", lambda **_kwargs: None)
        monkeypatch.setattr(llm_mod, "_legacy_env", lambda: None)
        assert llm_mod._ssl_verify() is True
        before = _ca_env()

        lm = llm_mod.build_lm(
            model="test-model",
            api_base=api_base,
            api_key="system-trust-key",
            max_tokens=8,
        )
        lm.num_retries = 0
        try:
            result = lm.forward(
                messages=[{"role": "user", "content": "hello"}],
            )
        finally:
            lm.close()

        assert result.choices[0].message.content == "private-ca-ok"
        assert len(bodies) == 1
        assert "ssl_verify" not in bodies[0]
        assert _ca_env() == before


def test_tls_transport_dependencies_are_declared_rather_than_inherited():
    """The implementation imports these directly, whatever dspy happens to install."""
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        import tomli as tomllib

    root = pathlib.Path(__file__).resolve().parents[2]
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    cia = data["project"]["optional-dependencies"]["cia"]
    for dependency in ("certifi", "httpx", "openai"):
        assert any(dep.split(";")[0].strip().startswith(dependency) for dep in cia), (
            dependency,
            cia,
        )
