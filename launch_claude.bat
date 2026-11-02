@echo off
title Claude Code with Auto-Failover Proxy
cd /d "%~dp0"

echo [1/2] Pointing Claude Code at the Local Failover Proxy for this window...
set ANTHROPIC_BASE_URL=http://127.0.0.1:8088/api
set ANTHROPIC_AUTH_TOKEN=local-proxy

echo [2/2] Starting Failover Proxy in background...
start /B python claude_proxy.py

timeout /t 2 /nobreak >nul

echo Starting Claude Code...
echo Press Ctrl+C in Claude to exit.
claude
