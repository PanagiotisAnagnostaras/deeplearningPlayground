FROM ubuntu:24.04

WORKDIR /workspace

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

# Copy source code
COPY . .