@echo off
title Claude Code with Auto-Failover Proxy
cd /d "%~dp0"

echo [1/2] Updating Claude Code settings for Local Failover Proxy...
python -c "import json; p = r'.claude/settings.local.json'; data = json.load(open(p)); data['env']['ANTHROPIC_BASE_URL'] = 'http://127.0.0.1:8088/api'; data['env']['ANTHROPIC_AUTH_TOKEN'] = 'local-proxy'; json.dump(data, open(p, 'w'), indent=2)"

echo [2/2] Starting Failover Proxy in background...
start /B python claude_proxy.py

timeout /t 2 /nobreak >nul

echo Starting Claude Code...
echo Press Ctrl+C in Claude to exit.
claude
