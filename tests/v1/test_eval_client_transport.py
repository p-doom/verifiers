import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx

from verifiers.v1.clients.eval import EvalClient
from verifiers.v1.dialects import ChatDialect
from verifiers.v1.types import SamplingConfig


def _completion(content: str) -> dict:
    return {
        "id": content,
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _server(
    content: str, requests: list[str]
) -> tuple[ThreadingHTTPServer, threading.Thread]:
    payload = json.dumps(_completion(content)).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            requests.append(self.path)
            self.rfile.read(int(self.headers.get("content-length", "0")))
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def test_eval_client_ignores_a_malicious_ambient_proxy(monkeypatch) -> None:
    upstream_requests: list[str] = []
    proxy_requests: list[str] = []
    upstream, upstream_thread = _server("upstream", upstream_requests)
    proxy, proxy_thread = _server("proxy", proxy_requests)
    upstream_url = f"http://127.0.0.1:{upstream.server_port}/v1"
    proxy_url = f"http://127.0.0.1:{proxy.server_port}"
    for name in (
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
    ):
        monkeypatch.setenv(name, proxy_url)
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)

    async def exercise():
        async with httpx.AsyncClient(trust_env=True) as ambient_client:
            preflight = await ambient_client.post(f"{upstream_url}/chat/completions")
        assert preflight.json()["id"] == "proxy"
        proxy_requests.clear()

        client = EvalClient(upstream_url, "test-key")
        try:
            return await client.get_response(
                ChatDialect(),
                {"messages": [{"role": "user", "content": "hello"}]},
                "test-model",
                SamplingConfig(),
            )
        finally:
            await client.close()

    try:
        response = asyncio.run(exercise())
    finally:
        for server, thread in ((upstream, upstream_thread), (proxy, proxy_thread)):
            server.shutdown()
            server.server_close()
            thread.join()

    assert response.message.content == "upstream"
    assert upstream_requests == ["/v1/chat/completions"]
    assert proxy_requests == []
