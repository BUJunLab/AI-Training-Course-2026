#!/usr/bin/env bash
# Create the conda environment for the Brain MRI CNN hands-on and launch Jupyter.
#
# Usage (run from anywhere — the script resolves its own paths):
#   bash env/setup.sh           # create/update the env AND launch Jupyter (default)
#   bash env/setup.sh setup     # only create/update the env, do not launch Jupyter
#   bash env/setup.sh run       # only launch Jupyter (env must already exist)
#
# After the env is created you can also use it manually:
#   conda activate cnn-handson
#   jupyter notebook notebooks/

set -euo pipefail

# Resolve the env/ directory (where this script lives) and the project root.
ENV_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "${ENV_DIR}/.." && pwd )"
cd "$PROJECT_DIR"

ENV_NAME="cnn-handson"
YAML_FILE="${ENV_DIR}/environment.yml"
MODE="${1:-all}"

# ---- sanity checks ----------------------------------------------------

if ! command -v conda >/dev/null 2>&1; then
    echo "Error: 'conda' not found on PATH."                                >&2
    echo "Install Miniconda or Anaconda first:"                             >&2
    echo "  https://docs.conda.io/en/latest/miniconda.html"                 >&2
    exit 1
fi

if [ ! -f "$YAML_FILE" ]; then
    echo "Error: ${YAML_FILE} not found."                                   >&2
    exit 1
fi

# Make `conda activate` usable from inside this script
# shellcheck disable=SC1091
eval "$(conda shell.bash hook)"

# ---- helpers ----------------------------------------------------------

env_exists() {
    conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"
}

setup_env() {
    if env_exists; then
        echo "==> Conda environment '${ENV_NAME}' already exists. Updating from ${YAML_FILE} ..."
        conda env update -f "$YAML_FILE" --prune
    else
        echo "==> Creating conda environment '${ENV_NAME}' from ${YAML_FILE} ..."
        conda env create -f "$YAML_FILE"
    fi
    echo "==> Environment ready."
}

run_jupyter() {
    if ! env_exists; then
        echo "Error: environment '${ENV_NAME}' does not exist yet."         >&2
        echo "Run:  bash env/setup.sh setup"                                >&2
        exit 1
    fi
    conda activate "$ENV_NAME"
    echo ""
    echo "==> Activated '${ENV_NAME}'."
    echo "==> Launching Jupyter Notebook in ${PROJECT_DIR}/notebooks/ ..."
    echo "    (Ctrl+C in this terminal to stop the server.)"
    echo ""
    exec jupyter notebook notebooks/
}

# ---- dispatch ---------------------------------------------------------

case "$MODE" in
    all)
        setup_env
        run_jupyter
        ;;
    setup)
        setup_env
        echo ""
        echo "Activate manually with:  conda activate ${ENV_NAME}"
        ;;
    run)
        run_jupyter
        ;;
    *)
        echo "Unknown mode: ${MODE}"                                        >&2
        echo "Usage: bash env/setup.sh [setup|run]"                         >&2
        exit 1
        ;;
esac
