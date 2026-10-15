"""
Claude Code Failover Proxy for OpenRouter
=========================================
Automatically intercepts 429 (Rate Limit Exceeded) errors from OpenRouter
and seamlessly switches between multiple accounts, API keys, and models.

Usage:
  1. Start this proxy:
       python claude_proxy.py
  2. Run Claude Code in your terminal:
       claude
"""

import http.server
import json
import os
import sys
import threading
import time
import urllib.request
from typing import Any, Dict, List

PORT = 8088

def load_env_file(filepath: str = ".env") -> None:
    """Load key-value pairs from .env without external dependencies."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filepath)
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass


load_env_file()

KEY_1 = os.environ.get("OPENROUTER_API_KEY_1", "")
KEY_2 = os.environ.get("OPENROUTER_API_KEY_2", "")

# Define accounts & fallback sequence (using verified active OpenRouter free models)
CONFIGS: List[Dict[str, str]] = []
if KEY_1:
    CONFIGS.append({
        "name": "Account 1 - Nemotron Ultra",
        "key": KEY_1,
        "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
    })
if KEY_2:
    CONFIGS.extend([
        {
            "name": "Account 2 - Nemotron Ultra",
            "key": KEY_2,
            "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
        },
        {
            "name": "Account 2 - Nemotron Super 120B",
            "key": KEY_2,
            "model": "nvidia/nemotron-3-super-120b-a12b:free",
        },
        {
            "name": "Account 2 - Gemma 4 31B",
            "key": KEY_2,
            "model": "google/gemma-4-31b-it:free",
        },
        {
            "name": "Account 2 - Nemotron Lightning",
            "key": KEY_2,
            "model": "nvidia/nemotron-3.5-lightning:free",
        },
    ])


class FailoverHandler(http.server.BaseHTTPRequestHandler):
    current_index = 0

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress default noisy access logs
        pass

    def do_GET(self) -> None:
        cfg = CONFIGS[self.current_index]
        target_path = self.path if self.path.startswith("/api") else "/api" + self.path
        target_url = "https://openrouter.ai" + target_path

        req = urllib.request.Request(
            target_url,
            headers={
                "Authorization": f"Bearer {cfg['key']}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req) as res:
                self.send_response(res.status)
                for k, v in res.getheaders():
                    if k.lower() not in ["transfer-encoding", "connection"]:
                        self.send_header(k, v)
                self.end_headers()
                self.wfile.write(res.read())
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self.end_headers()
            self.wfile.write(e.read())

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            payload = {}

        target_path = self.path if self.path.startswith("/api") else "/api" + self.path
        target_url = "https://openrouter.ai" + target_path

        # Try configurations in pool; if 429 or 5xx upstream error, auto-rotate through remaining configurations
        total_configs = len(CONFIGS)
        max_attempts = total_configs * 2
        for attempt in range(max_attempts):
            cfg = CONFIGS[self.current_index]
            if isinstance(payload, dict):
                # Apply model override for this configuration
                payload["model"] = cfg["model"]
                req_data = json.dumps(payload).encode("utf-8")
            else:
                req_data = body

            headers = {
                "Authorization": f"Bearer {cfg['key']}",
                "Content-Type": "application/json",
            }
            if "anthropic-version" in self.headers:
                headers["anthropic-version"] = self.headers["anthropic-version"]
            if "anthropic-beta" in self.headers:
                headers["anthropic-beta"] = self.headers["anthropic-beta"]

            req = urllib.request.Request(target_url, headers=headers, data=req_data)

            try:
                res = urllib.request.urlopen(req, timeout=60)
                print(f"[Proxy -> OK] Served via {cfg['name']} | Model: {cfg['model']}")

                self.send_response(res.status)
                for k, v in res.getheaders():
                    if k.lower() not in ["content-length", "transfer-encoding", "connection"]:
                        self.send_header(k, v)
                self.end_headers()

                # Stream response chunks back to Claude Code
                while True:
                    chunk = res.read(512)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
                return

            except urllib.error.HTTPError as e:
                err_data = e.read()
                err_text = err_data.decode("utf-8", errors="ignore")

                # If rate limited (429) or upstream error (502/503/524), switch immediately
                if e.code in [429, 502, 503, 504, 524] or "rate" in err_text.lower():
                    next_index = (self.current_index + 1) % total_configs
                    print(
                        f"\n[Proxy -> {e.code} FAILOVER] {cfg['name']} ({cfg['model']}) error: {err_text[:80]}...\n"
                        f"  -> Switching automatically to {CONFIGS[next_index]['name']} ({CONFIGS[next_index]['model']})...\n"
                    )
                    self.current_index = next_index
                    time.sleep(1)
                    continue
                else:
                    print(f"[Proxy -> Error {e.code}] {err_text[:160]}")
                    self.send_response(e.code)
                    self.end_headers()
                    self.wfile.write(err_data)
                    return
            except Exception as e:
                next_index = (self.current_index + 1) % total_configs
                print(f"\n[Proxy -> Connection Error: {e}] Switching to {CONFIGS[next_index]['name']}...\n")
                self.current_index = next_index
                time.sleep(1)
                continue

        # If all exhausted
        self.send_response(429)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(
            json.dumps({
                "type": "error",
                "error": {
                    "type": "rate_limit_error",
                    "message": "All configured OpenRouter accounts and models have exceeded their daily rate limits.",
                },
            }).encode("utf-8")
        )


def main() -> None:
    server_address = ("127.0.0.1", PORT)
    server = http.server.ThreadingHTTPServer(server_address, FailoverHandler)
    print("=" * 65)
    print("      CLAUDE CODE MULTI-ACCOUNT FAILOVER PROXY")
    print("=" * 65)
    print(f"Listening locally at: http://127.0.0.1:{PORT}")
    print("\nConfigured failover pool:")
    for idx, c in enumerate(CONFIGS, 1):
        masked_key = c["key"][:12] + "..." + c["key"][-6:]
        print(f"  {idx}. {c['name']:<28} | Model: {c['model']:<36} | Key: {masked_key}")
    print("\nAuto-failover is active: If 429 is encountered, requests")
    print("will instantly and silently route to the next available account.")
    print("Press Ctrl+C to stop.\n" + "=" * 65 + "\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down proxy...")
        server.server_close()


if __name__ == "__main__":
    main()
