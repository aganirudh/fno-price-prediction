import gradio as gr
import subprocess
import os
import signal
import time

def run_command(command):
    try:
        # Runs the command and returns the output
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        output = ""
        # Capture first 50 lines or wait for 5 seconds to show initial progress
        start_time = time.time()
        while time.time() - start_time < 10:
            line = process.stdout.readline()
            if not line: break
            output += line
            if len(output.split('\n')) > 100: break
        
        return output + "\n... Command continues in background. Check logs for details."
    except Exception as e:
        return str(e)

def get_latest_report():
    reports_dir = "reports"
    if not os.path.exists(reports_dir):
        return "No reports directory found."
    reports = sorted([f for f in os.listdir(reports_dir) if f.endswith(".html")])
    if reports:
        with open(os.path.join(reports_dir, reports[-1]), "r") as f:
            return f.read()
    return "No HTML reports found yet. Run a comparison to generate one."

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 PCP Arbitrage RL Command Center")
    gr.Markdown("Management console for the Put-Call Parity Arbitrage Reinforcement Learning system.")
    
    with gr.Tab("Training & Execution"):
        with gr.Row():
            btn_ensemble = gr.Button("Step 1: Train Ensemble RL (PPO/A2C/DDPG)", variant="primary")
            btn_hybrid = gr.Button("Step 2: Train Hybrid (LLM/GRPO)", variant="secondary")
        
        with gr.Row():
            btn_paper = gr.Button("Start Paper Trading", variant="stop")
            btn_stop = gr.Button("Stop All Processes", variant="secondary")
            
        output_log = gr.Textbox(label="Process Output (Snippet)", lines=15)
        
        btn_ensemble.click(lambda: run_command("python3 main.py --mode train-ensemble --timesteps 50000"), outputs=output_log)
        btn_hybrid.click(lambda: run_command("python3 main.py --mode train-hybrid --steps 1000"), outputs=output_log)
        btn_paper.click(lambda: run_command("python3 main.py --mode paper --feed mock"), outputs=output_log)
        btn_stop.click(lambda: "All background processes signal sent (Mock)", outputs=output_log)

    with gr.Tab("Strategy Comparison"):
        btn_compare = gr.Button("Run Full 5-Strategy Backtest")
        report_view = gr.HTML(label="Latest Report", value=get_latest_report())
        
        def run_compare_and_show():
            run_command("python3 main.py --mode compare")
            return get_latest_report()
            
        btn_compare.click(run_compare_and_show, outputs=report_view)

    with gr.Tab("System Status"):
        with gr.Row():
            btn_status = gr.Button("Check MCP & Data Status")
        status_out = gr.Textbox(label="System Status", lines=10)
        
        def check_status():
            mcp = run_command("ps aux | grep _server.py")
            data = run_command("ls -R data/kaggle")
            return f"--- MCP Servers ---\n{mcp}\n\n--- Data Files ---\n{data}"
            
        btn_status.click(check_status, outputs=status_out)

demo.launch(server_name="0.0.0.0", server_port=7860)
