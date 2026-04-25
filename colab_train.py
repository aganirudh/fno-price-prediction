"""
Colab Orchestrator - Starts MCP servers and launches the NIFTY50 RL Pipeline.
"""
import subprocess
import time
import requests
import sys
import os
from pathlib import Path

# Fix Python Path for Colab
sys.path.insert(0, os.getcwd())
os.environ["PYTHONPATH"] = os.getcwd()

SERVERS = [
    {"name": "market_data", "path": "mcp_servers/market_data_server.py", "port": 8001},
    {"name": "risk", "path": "mcp_servers/risk_server.py", "port": 8002},
    {"name": "cost", "path": "mcp_servers/cost_server.py", "port": 8003},
]

def start_servers():
    processes = []
    print("🚀 Starting MCP Infrastructure Servers...")
    for server in SERVERS:
        print(f"  -> Launching {server['name']} on port {server['port']}...")
        proc = subprocess.Popen([sys.executable, server['path']], 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.STDOUT)
        processes.append(proc)
    
    time.sleep(5) # Wait for startup
    return processes

def run_pipeline():
    print("\n📦 Step 0: Ensuring NIFTY50 Data is present...")
    try:
        subprocess.run([sys.executable, "tools/download_data.py"], check=True)
    except Exception as e:
        print(f"⚠️ Data check failed: {e}")

    print("\n📈 Step 1: Training Portfolio Ensemble (PPO+A2C+DDPG)...")
    try:
        subprocess.run([sys.executable, "main.py", "--mode", "train-ensemble", "--timesteps", "50000"], check=True)
    except Exception as e:
        print(f"❌ Ensemble training failed: {e}")
        return

    print("\n🧠 Step 2: Training Hybrid Arbitrage Agent (GRPO)...")
    try:
        subprocess.run([sys.executable, "main.py", "--mode", "train-hybrid", "--steps", "1000"], check=True)
    except Exception as e:
        print(f"❌ Hybrid training failed: {e}")

if __name__ == "__main__":
    server_procs = start_servers()
    try:
        run_pipeline()
    finally:
        print("\n🧹 Cleaning up servers...")
        for p in server_procs:
            p.terminate()
        print("Done.")
