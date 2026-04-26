import os
from huggingface_hub import HfApi

def diag():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: Set HF_TOKEN environment variable.")
        return
    api = HfApi(token=token)
    print("Submitting diagnostic job...")
    job = api.run_job(
        image="python:3.10-slim",
        command=["bash", "-c", "echo 'SYSTEM MEMORY:'; free -h; echo 'CPU CORES:'; nproc; sleep 10"],
        flavor="a10g-small"
    )
    print(f"Job ID: {job.id}")

if __name__ == "__main__":
    diag()
