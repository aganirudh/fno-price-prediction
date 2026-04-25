# Use a CUDA-enabled base image
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git wget \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# Step 1: Install Torch first (the heaviest layer)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Step 2: Install Unsloth and Xformers
RUN pip install --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
RUN pip install --no-cache-dir xformers

# Step 3: Install RL and App dependencies
RUN pip install --no-cache-dir \
    trl peft transformers accelerate \
    stable-baselines3 gymnasium shimmy stockstats scikit-learn \
    fastapi uvicorn kaggle pandas numpy rich jinja2 gradio

COPY --chown=user . $HOME/app

EXPOSE 7860
EXPOSE 8001
EXPOSE 8002
EXPOSE 8003

CMD ["bash", "entrypoint.sh"]
