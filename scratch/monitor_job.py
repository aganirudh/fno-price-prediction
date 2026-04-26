"""Monitor HF job status."""
import os
import time
from huggingface_hub import HfApi

TOKEN = os.environ.get("HF_TOKEN")

if not TOKEN:
    print("ERROR: Set HF_TOKEN environment variable.")
    exit(1)

api = HfApi(token=TOKEN)

for i in range(20):
    try:
        jobs = list(api.list_jobs(token=TOKEN))
        if not jobs:
            print("No jobs found.")
            break
        j = jobs[0]
        print(f"[{i}] Job {j.id}: {j.status.stage} | {j.status.message or 'no message'}")
        if j.status.stage in ("COMPLETED", "ERROR"):
            break
    except Exception as e:
        print(f"Error: {e}")
    time.sleep(30)
