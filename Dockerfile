# KernelBench Docker Image
# nvcc 12.8 原生支持 Blackwell (sm_100/120/121)；与 PyTorch cu128 wheel 对齐
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

WORKDIR /workspace

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace/src

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# 【新增】备份原始源并替换为阿里云镜像源，解决连接超时问题
RUN cp /etc/apt/sources.list /etc/apt/sources.list.bak && \
    sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list

RUN apt-get update && apt-get install -y \
    software-properties-common \
    git gcc-10 g++-10 clang curl wget make cmake ninja-build build-essential \
    zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev \
    libreadline-dev libffi-dev libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y python3.12 python3.12-dev && \
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 100

RUN pip3 install \
    torch==2.9.0+cu128 torchvision==0.24.0+cu128 torchaudio==2.9.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# 验证 nvcc 版本（可选）
RUN nvcc --version

COPY . /workspace/

# RTX 5090 = sm_120 (Blackwell, GB202)，不是 sm_90 (Hopper)
# 同时保留向后兼容的常见 arch
ENV TORCH_CUDA_ARCH_LIST="12.0;9.0;8.9;8.6;8.0;7.5"

RUN python3.12 -m pip install pydra_config modal litellm ninja tomli tenacity
RUN python3.12 -m pip install .

CMD ["/bin/bash"]
