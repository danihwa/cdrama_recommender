FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

# uv (manages Python + dependencies)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:/root/.cargo/bin:${PATH}"

# Python 3.12 via uv
RUN uv python install 3.12 && \
    ln -s /root/.local/bin/python3.12 /usr/local/bin/python3 && \
    ln -s /root/.local/bin/python3.12 /usr/local/bin/python

# Claude Code
RUN curl -fsSL https://claude.ai/install.sh | bash

RUN git config --global user.email "danichka.svobodova@gmail.com" && \
    git config --global user.name "danihwa"

WORKDIR /workspace
CMD ["bash"]
