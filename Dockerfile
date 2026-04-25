# Use a CUDA-enabled base image for Unsloth
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Set up environment
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set up user
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# Copy project files
COPY --chown=user . $HOME/app

# Install Python dependencies
# We use a specific unsloth install for stable performance
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" \
    trl peft transformers accelerate \
    stable-baselines3 gymnasium shimmy stockstats scikit-learn \
    fastapi uvicorn kaggle pandas numpy rich jinja2 \
    gradio

# Expose ports for MCP servers and Gradio
EXPOSE 7860
EXPOSE 8001
EXPOSE 8002
EXPOSE 8003

# Run the entrypoint
CMD ["bash", "entrypoint.sh"]
