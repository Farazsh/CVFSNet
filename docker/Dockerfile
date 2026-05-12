FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# BUILD
# docker build . -t cvfsnet --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -f docker/Dockerfile
#
# RUN
# docker run --rm --gpus=all --shm-size 8G -v $(pwd):/opt/project -ti cvfsnet

SHELL ["/bin/bash", "--login", "-c"]

ARG USER_ID=1000
ARG GROUP_ID=1000

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH="/opt/venv/bin:/home/user/.local/bin:${PATH}"

WORKDIR /opt/project

# OS + Python tooling required by project dependencies
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
      python3 \
      python3-pip \
      python3-venv \
      git \
      ca-certificates \
      libgl1 \
      libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user (matching host IDs when provided)
RUN addgroup --gid ${GROUP_ID} user && \
    adduser --disabled-password --gecos '' --uid ${USER_ID} --gid ${GROUP_ID} user

USER user

# Install uv
RUN pip3 install --no-cache-dir --user uv

# Copy only dependency manifest first for better caching
COPY --chown=user:user pyproject.toml ./pyproject.toml

# Create virtual environment and install dependencies with uv
RUN uv venv /opt/venv && \
    uv sync --no-dev

# Copy full project source
COPY --chown=user:user . .

CMD ["bash"]
