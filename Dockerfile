# 1. Base on Python 3.10.9 (slim Bullseye)
FROM python:3.10.9-slim-bullseye

# 2. Prevent .pyc writes & enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 3. Set work directory
WORKDIR /pixal

# 4. Install prerequisites (curl, gnupg, ca-certs, wget, tar) + inotify-tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      gnupg \
      wget \
      tar \
      inotify-tools \
      # runtime deps some libs may use:
      libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

# 5. Add NVIDIA's CUDA repo keyring
RUN curl -fsSL \
      https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/cuda-keyring_1.1-1_all.deb \
      -o cuda-keyring.deb && \
    dpkg -i cuda-keyring.deb && \
    rm cuda-keyring.deb

# 6. Install CUDA runtime libraries + NVVM (for libdevice) without driver deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      cuda-cudart-12-1 \
      cuda-libraries-12-1 \
      cuda-minimal-build-12-1 \
    && rm -rf /var/lib/apt/lists/*

# 7. Download & install cuDNN 9.3.0
RUN wget -q https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.3.0.75_cuda12-archive.tar.xz && \
    tar -xJf cudnn-linux-x86_64-9.3.0.75_cuda12-archive.tar.xz && \
    find . -type f -name 'cudnn*.h' -exec cp -P {} /usr/local/cuda/include/ \; && \
    find . -type f -name 'libcudnn*' -exec cp -P {} /usr/local/cuda/lib64/ \; && \
    chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn* && \
    rm -rf cudnn-linux-x86_64-9.3.0.75_cuda12-archive*

# 8. Copy project files and install Python dependencies
COPY . .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -e .[gpu]

# 9. Add the watcher script (component-ready logic)
COPY watch.sh /usr/local/bin/watch.sh
RUN chmod +x /usr/local/bin/watch.sh

# Defaults for the watcher (override at runtime if needed)
ENV WATCH_DIR=/mount/machine_learning/validation \
    OUTPUT_ROOT=/mount/machine_learning/results \
    PATTERN=jpg,jpeg,png,tif,tiff,bmp \
    CMD_TEMPLATE='pixal validate -i "{component}" -o "{output}"' \
    SENTINEL_NAME=.validated \
    RERUN_ON_CHANGE=false \
    REQUIRED_DIRS= \
    READY_TIMEOUT=60 \
    READY_POLL_INTERVAL=0.5


# 11. Default entrypoint is still the CLI (backwards compatible)
ENTRYPOINT ["pixal"]
CMD []