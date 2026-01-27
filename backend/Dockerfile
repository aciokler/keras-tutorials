FROM nvidia/cuda:13.1.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    KERAS_BACKEND=jax \
    CUDA_VISIBLE_DEVICES=0 \
    XLA_PYTHON_CLIENT_MEM_FRACTION=1.00 \
    LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# Install python3 and venv support
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev build-essential ca-certificates git curl \
 && rm -rf /var/lib/apt/lists/*

# Create a venv and upgrade pip inside it
RUN python3 -m venv $VIRTUAL_ENV \
 && $VIRTUAL_ENV/bin/python -m pip install --upgrade pip setuptools wheel

WORKDIR /workspace/
COPY minigpt /workspace
COPY requirements.txt /workspace
COPY testcuda.py /workspace

## Install other Python deps into the venv
#RUN $VIRTUAL_ENV/bin/pip install --no-cache-dir \
#    keras-core \
#    keras-hub \
#    tensorflow-cpu \
#    sentencepiece \
#    numpy \
#    matplotlib

#WORKDIR /workspace/
#COPY minigpt /workspac

RUN $VIRTUAL_ENV/bin/pip install --no-cache-dir -r requirements.txt

# Use the venv python explicitly
ENTRYPOINT ["/opt/venv/bin/python", "gpt.py"]