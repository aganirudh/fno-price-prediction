# Use the official PyTorch image (much more stable for HF)
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install only necessary system tools
RUN apt-get update && apt-get install -y git wget && rm -rf /var/lib/apt/lists/*

# Set up user
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# Step 1: Install Unsloth (using the pre-installed Torch)
RUN pip install --no-cache-dir --no-deps "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
RUN pip install --no-cache-dir xformers

# Step 2: Install RL and App dependencies (excluding torch)
RUN pip install --no-cache-dir \
    trl peft transformers accelerate \
    stable-baselines3 gymnasium shimmy stockstats scikit-learn \
    fastapi uvicorn kaggle pandas numpy rich jinja2 gradio wandb

COPY --chown=user . $HOME/app

EXPOSE 7860
EXPOSE 8001
EXPOSE 8002
EXPOSE 8003

CMD ["bash", "entrypoint.sh"]
