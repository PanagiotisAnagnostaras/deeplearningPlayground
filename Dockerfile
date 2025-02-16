FROM ubuntu:24.04

WORKDIR /deeplearningPlayground

RUN apt-get update && \
    apt-get install -y \
        software-properties-common \
        git \
        build-essential \
        gdb \
        python3 \
        vim \
        libasio-dev \
        libx11-dev \
        ffmpeg \
        imagemagick \
        x11-apps \
        python3.12-venv \
        xorg \
        openbox \
        python3-setuptools \
        libx11-6 \
        ca-certificates \
        curl \
        iputils-ping \
        cmake \
        python3-dev \
        python3-tk \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /workspace/.venv && \
    /workspace/.venv/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    /workspace/.venv/bin/pip install -r requirements.txt
# Copy source code
COPY . .