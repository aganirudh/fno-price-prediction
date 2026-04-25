#!/bin/bash

# 1. Start MCP Servers
python3 mcp_servers/market_data_server.py &
python3 mcp_servers/risk_server.py &
python3 mcp_servers/cost_server.py &

# 2. Start the Gradio Dashboard (on the port HF expects)
python3 app.py
