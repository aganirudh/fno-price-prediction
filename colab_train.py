"""
Colab/HF Orchestrator - Starts MCP servers and runs the NIFTY50 RL Pipeline.
Works in Colab, HF Spaces, and locally.
"""
import subprocess
import time
import sys
import os
from pathlib import Path

# Fix Python Path
sys.path.insert(0, os.getcwd())
os.environ["PYTHONPATH"] = os.getcwd()

SERVERS = [
    {"name": "market_data", "path": "mcp_servers/market_data_server.py", "port": 8001},
    {"name": "risk", "path": "mcp_servers/risk_server.py", "port": 8002},
    {"name": "cost", "path": "mcp_servers/cost_server.py", "port": 8003},
]

def start_servers():
    processes = []
    print("[Infra] Starting MCP servers...")
    for server in SERVERS:
        print(f"  -> {server['name']} on port {server['port']}")
        proc = subprocess.Popen(
            [sys.executable, server['path']],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        processes.append(proc)
    time.sleep(5)
    print("[Infra] Servers started.")
    return processes

def run_pipeline():
    # Step 0: Download Data
    print("\n[Step 0] Downloading NIFTY50 data...")
    result = subprocess.run([sys.executable, "tools/download_data.py"])
    if result.returncode != 0:
        print("[WARN] Data download had issues, continuing anyway...")

    # Step 1: Train Ensemble (PPO+A2C+DDPG) - CPU is fine for this
    print("\n[Step 1] Training Portfolio Ensemble (PPO+A2C+DDPG)...")
    result = subprocess.run([
        sys.executable, "main.py",
        "--mode", "train-ensemble",
        "--timesteps", "50000",
        "--no-wandb"
    ])
    if result.returncode != 0:
        print("[ERROR] Ensemble training failed.")
        return

    # Step 2: Train Hybrid (GRPO) - needs GPU
    print("\n[Step 2] Training Hybrid Arbitrage Agent (GRPO)...")
    try:
        import torch
        if not torch.cuda.is_available():
            print("[WARN] No GPU detected. Skipping GRPO hybrid training.")
            print("       The Ensemble RL training (Step 1) is complete.")
            return
    except ImportError:
        print("[WARN] torch not installed. Skipping GRPO step.")
        return

    result = subprocess.run([
        sys.executable, "main.py",
        "--mode", "train-hybrid",
        "--steps", "1000",
        "--no-wandb"
    ])
    if result.returncode != 0:
        print("[ERROR] Hybrid training failed.")

if __name__ == "__main__":
    server_procs = start_servers()
    try:
        run_pipeline()
    finally:
        print("\n[Cleanup] Stopping servers...")
        for p in server_procs:
            p.terminate()
        print("Done.")
