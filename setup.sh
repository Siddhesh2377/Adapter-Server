#!/usr/bin/env bash
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[+]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
error() { echo -e "${RED}[x]${NC} $1"; }

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo ""
echo "================================="
echo "  Adapter Training System Setup"
echo "================================="
echo ""

# ── Python check ──
PYTHON=""
for cmd in python3.12 python3.11 python3.10 python3; do
    if command -v "$cmd" &>/dev/null; then
        ver=$("$cmd" --version 2>&1 | grep -oP '\d+\.\d+')
        major=$(echo "$ver" | cut -d. -f1)
        minor=$(echo "$ver" | cut -d. -f2)
        if [ "$major" -eq 3 ] && [ "$minor" -ge 10 ]; then
            PYTHON="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    error "Python 3.10+ is required but not found."
    exit 1
fi
info "Using $($PYTHON --version)"

# ── Virtual environment ──
if [ ! -d ".venv" ]; then
    info "Creating virtual environment..."
    $PYTHON -m venv .venv
else
    info "Virtual environment already exists."
fi

source .venv/bin/activate
info "Activated .venv"

# ── Dependencies ──
info "Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
info "Dependencies installed."

# ── Environment file ──
if [ ! -f ".env" ]; then
    if [ -f "example.env" ]; then
        cp example.env .env
        warn ".env created from example.env -- edit it with your credentials:"
        echo ""
        echo "    SUPABASE_URL        - Your Supabase project URL"
        echo "    SUPABASE_DB_URL     - PostgreSQL connection string"
        echo "    SUPABASE_SERVICE_KEY - Service role key"
        echo "    HF_TOKEN            - HuggingFace token (optional, for auto-upload)"
        echo "    HF_REPO_ID          - HuggingFace repo (optional, for auto-upload)"
        echo ""
    else
        error "example.env not found. Create .env manually."
        exit 1
    fi
else
    info ".env already exists."
fi

# ── Verify vendored llama.cpp tools ──
INCLUDE_DIR="$PROJECT_DIR/include/llama"

if [ -f "$INCLUDE_DIR/convert_lora_to_gguf.py" ] && [ -f "$INCLUDE_DIR/convert_hf_to_gguf.py" ]; then
    info "Conversion scripts found in include/llama/"
else
    error "Missing conversion scripts in include/llama/"
    error "Expected: convert_lora_to_gguf.py, convert_hf_to_gguf.py, gguf-py/"
    exit 1
fi

if [ -f "$INCLUDE_DIR/bin/llama-quantize" ]; then
    info "llama-quantize binary found."
else
    warn "llama-quantize binary not found in include/llama/bin/"
    echo ""
    echo "  To build it from llama.cpp source:"
    echo "    git clone https://github.com/ggml-org/llama.cpp /tmp/llama.cpp"
    echo "    cd /tmp/llama.cpp && cmake -B build && cmake --build build --target llama-quantize"
    echo "    cp build/bin/llama-quantize $INCLUDE_DIR/bin/"
    echo "    cp build/bin/lib*.so* $INCLUDE_DIR/bin/"
    echo ""
fi

# ── Database migration ──
source .env 2>/dev/null || true
if [ -n "$SUPABASE_DB_URL" ]; then
    info "Running database migrations..."
    alembic upgrade head
    info "Migrations applied."
else
    warn "SUPABASE_DB_URL not set -- skipping migrations. Set it in .env and run: alembic upgrade head"
fi

# ── Done ──
echo ""
echo "================================="
echo -e "  ${GREEN}Setup complete.${NC}"
echo "================================="
echo ""
echo "  To start:"
echo "    source .venv/bin/activate"
echo "    python src/gradio_trainer.py"
echo ""
echo "  Opens at http://localhost:7860"
echo ""
