FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# Build arguments for AWS CodeArtifact access
ARG TERRAY_PYPI_STORE
ARG AWS_AUTH_TOKEN

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install system dependencies including Python 3.10
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    nano \
    git \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    gfortran \
    libblas-dev \
    liblapack-dev \
    libopenblas-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install xTB 6.6.1 from GitHub releases
# Download pre-compiled binary for Linux x86_64
RUN wget https://github.com/grimme-lab/xtb/releases/download/v6.6.1/xtb-6.6.1-linux-x86_64.tar.xz \
    && tar -xf xtb-6.6.1-linux-x86_64.tar.xz \
    && cp xtb-6.6.1/bin/xtb /usr/local/bin/ \
    && chmod +x /usr/local/bin/xtb \
    && rm -rf xtb-6.6.1-linux-x86_64.tar.xz xtb-6.6.1

# Verify xTB installation
RUN xtb --version

# Create Python virtual environment
RUN python3.10 -m venv /opt/tpose-env

# Activate virtual environment by setting PATH
ENV PATH="/opt/tpose-env/bin:$PATH"

# Upgrade pip and install core dependencies
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.2 support
RUN pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# Install SO3LR and ASE
RUN pip install so3lr>=1.0.0 ase>=3.22.1

# Set working directory
WORKDIR /app

# Copy only requirements.txt first to cache pip installs
COPY services/pose-ranking/requirements.txt /app/requirements.txt

# Install service requirements (cached unless requirements.txt changes)
RUN pip install -r requirements.txt

# Install terray packages from CodeArtifact (token always changes, breaks cache)
# Put this BEFORE copying code so code changes don't force reinstall
RUN pip install \
    --index-url https://aws:${AWS_AUTH_TOKEN}@${TERRAY_PYPI_STORE} \
    --extra-index-url https://pypi.python.org/simple/ \
    tengine2>=0.6.1

# Now copy code files (only rebuilds these layers on code changes)
COPY shared/ /app/shared/
COPY services/pose-ranking/src/ /app/src/
COPY services/pose-ranking/tpose_ranking_entrypoint.py /app/

# Create non-root user and fix ownership
RUN useradd -m -u 1000 tposeuser && \
    chown -R tposeuser:tposeuser /app

USER tposeuser

# Verify installations
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')" || echo "PyTorch check skipped"
RUN python -c "import so3lr; print('SO3LR imported successfully')" || echo "SO3LR check skipped"
RUN python -c "from Bio import PDB; print('BioPython imported successfully')"
RUN python -c "from rdkit import Chem; print('RDKit imported successfully')"

# Set entrypoint for Tengine2/TDesign integration
# This entrypoint handles pickled I/O from S3 and runs pose ranking on GPU/CPU
ENTRYPOINT ["python", "tpose_ranking_entrypoint.py"]
