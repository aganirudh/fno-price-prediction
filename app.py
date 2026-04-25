import gradio as gr
import subprocess
import os
import time

# Pre-create data directory
os.makedirs("data/kaggle", exist_ok=True)

def run_command(command):
    try:
        # Runs the command and returns a snippet of output
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        output = ""
        start_time = time.time()
        # Capture for up to 5 seconds
        while time.time() - start_time < 5:
            line = process.stdout.readline()
            if not line: break
            output += line
            if len(output.split('\n')) > 50: break
        
        return output if output else "Command started (no immediate output)."
    except Exception as e:
        return str(e)

def get_latest_report():
    reports_dir = "reports"
    if not os.path.exists(reports_dir): return "No reports yet."
    reports = sorted([f for f in os.listdir(reports_dir) if f.endswith(".html")])
    if reports:
        with open(os.path.join(reports_dir, reports[-1]), "r") as f:
            return f.read()
    return "No HTML reports found."

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 PCP Arbitrage RL Command Center")
    
    with gr.Tab("Data & Training"):
        with gr.Row():
            btn_download = gr.Button("Step 0: Download Kaggle Data", variant="secondary")
            btn_ensemble = gr.Button("Step 1: Train Ensemble RL", variant="primary")
            btn_hybrid = gr.Button("Step 2: Train Hybrid (LLM)", variant="primary")
        
        output_log = gr.Textbox(label="Process Output", lines=15)
        
        btn_download.click(lambda: run_command("python3 tools/download_data.py"), outputs=output_log)
        btn_ensemble.click(lambda: run_command("python3 main.py --mode train-ensemble --timesteps 50000"), outputs=output_log)
        btn_hybrid.click(lambda: run_command("python3 main.py --mode train-hybrid --steps 1000"), outputs=output_log)

    with gr.Tab("Strategy Comparison"):
        btn_compare = gr.Button("Run Full 5-Strategy Backtest")
        report_view = gr.HTML(label="Latest Report", value=get_latest_report())
        btn_compare.click(lambda: run_command("python3 main.py --mode compare") or get_latest_report(), outputs=report_view)

    with gr.Tab("System Status"):
        btn_status = gr.Button("Check Status")
        status_out = gr.Textbox(label="System Status", lines=10)
        
        def check_status():
            # Using ps -ef for better compatibility
            mcp = subprocess.getoutput("ps -ef | grep _server.py | grep -v grep")
            # Recursive ls to see nested Kaggle files
            data = subprocess.getoutput("ls -Rlh data/kaggle | grep -v '^total 0' | head -n 20")
            return f"--- MCP Servers ---\n{mcp}\n\n--- Data Files (Recursive) ---\n{data}"
            
        btn_status.click(check_status, outputs=status_out)

demo.launch(server_name="0.0.0.0", server_port=7860)
