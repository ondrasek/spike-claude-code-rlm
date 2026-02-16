# syntax=docker/dockerfile:1
FROM mcr.microsoft.com/devcontainers/base:bookworm

# Repository name passed from devcontainer.json build.args
ARG REPO_NAME
ENV REPO_NAME=${REPO_NAME}

# Install pyenv dependencies and zsh
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        libncursesw5-dev \
        xz-utils \
        tk-dev \
        libxml2-dev \
        libxmlsec1-dev \
        libffi-dev \
        liblzma-dev \
        jq \
        zsh && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create workspace directory (needed before postCreate clones the repo)
RUN mkdir -p /workspaces/${REPO_NAME} && \
    chown vscode:vscode /workspaces/${REPO_NAME}

# Switch to vscode user
USER vscode
WORKDIR /home/vscode

# Install pyenv
ENV PYENV_ROOT="/home/vscode/.pyenv"
ENV PATH="${PYENV_ROOT}/bin:${PYENV_ROOT}/shims:${PATH}"
RUN curl https://pyenv.run | bash

# Install Python 3.13 via pyenv
RUN pyenv install 3.13 && \
    pyenv global 3.13

# Install uv (Python package manager)
RUN pip install --user uv

# Add ~/.local/bin to PATH
ENV PATH="/home/vscode/.local/bin:${PATH}"

# Install dev tools that match pyproject.toml
RUN uv tool install ruff && \
    uv tool install pytest

# Install Claude CLI
RUN curl -fsSL https://claude.ai/install.sh | bash

# Install opencode (AI coding agent — curl|bash matches Claude CLI install pattern above)
RUN curl -fsSL https://opencode.ai/install | bash

# Install Oh My Zsh (skip if already present in base image)
RUN if [ ! -d /home/vscode/.oh-my-zsh ]; then \
        git clone --depth 1 https://github.com/ohmyzsh/ohmyzsh.git /home/vscode/.oh-my-zsh && \
        cp /home/vscode/.oh-my-zsh/templates/zshrc.zsh-template /home/vscode/.zshrc; \
    fi

# Configure shell PATH for pyenv and local bin
RUN echo 'export ZSH="$HOME/.oh-my-zsh"' >> /home/vscode/.zshrc && \
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> /home/vscode/.zshrc && \
    echo 'export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$HOME/.local/bin:$PATH"' >> /home/vscode/.zshrc && \
    echo 'eval "$(pyenv init -)"' >> /home/vscode/.zshrc && \
    echo 'source $ZSH/oh-my-zsh.sh' >> /home/vscode/.zshrc && \
    echo "alias repo=\"cd /workspaces/${REPO_NAME}\"" >> /home/vscode/.zshrc && \
    echo "export REPOSITORY_NAME=${REPO_NAME}" >> /home/vscode/.zshrc && \
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> /home/vscode/.bashrc && \
    echo 'export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$HOME/.local/bin:$PATH"' >> /home/vscode/.bashrc && \
    echo 'eval "$(pyenv init -)"' >> /home/vscode/.bashrc && \
    echo "alias repo=\"cd /workspaces/${REPO_NAME}\"" >> /home/vscode/.bashrc && \
    echo "export REPOSITORY_NAME=${REPO_NAME}" >> /home/vscode/.bashrc

# Configure static git settings
RUN git config --global init.defaultBranch main && \
    git config --global pull.rebase false && \
    git config --global credential.helper '!gh auth git-credential'

WORKDIR /workspaces/${REPO_NAME}
