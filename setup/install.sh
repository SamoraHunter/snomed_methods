#!/bin/bash
# setup/install.sh for snomed_methods
# Installs dependencies in a virtual environment and registers it as a Jupyter kernel

VENV_NAME="snomed_methods_env"
PROXY_MODE=false
CLONE_REPOS=true
FORCE_CLEAN=false
INSTALL_MODE="lite"  # Default to lite installation
DEV_MODE=false
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

show_help() {
    echo "Usage: ./setup/install.sh [OPTIONS]"
    echo "Options:"
    echo "  -h, --help                    Show this help message"
    echo "  -p, --proxy                   Install with proxy support"
    echo "  --no-clone                    Skip git clone operations"
    echo "  -f, --force                   Remove existing files and perform fresh install"
    echo "  -a, --all                     Install all components (overrides default lite installation)"
    echo "  --dev                         Install development dependencies"
    echo "  --name <name>                 Name for the virtual environment (default: snomed_methods_env)"
    echo ""
    echo "Dev container example:"
    echo "  bash setup/install.sh --dev --name snomed_devcontainer"
}

clone_repositories() {
    # Save current directory
    local current_dir=$(pwd)

    cd "$PROJECT_DIR" || { echo "Error: Could not change to project directory"; return 1; }

    # Clone any additional repos if needed in the future
    # Example:
    # local snomed_repo_url="https://github.com/SamoraHunter/snomed_methods.git"
    # local repos=("$snomed_repo_url")

    local repos=()

    # Save current proxy settings
    local saved_http_proxy="$http_proxy"
    local saved_https_proxy="$https_proxy"
    local saved_git_http_proxy=""
    local saved_git_https_proxy=""

    # Get current git proxy settings if they exist
    saved_git_http_proxy=$(git config --global --get http.proxy 2>/dev/null || echo "")
    saved_git_https_proxy=$(git config --global --get https.proxy 2>/dev/null || echo "")

    # Only disable the proxy if we are NOT in proxy mode.
    if [ "$PROXY_MODE" = false ]; then
        echo "Temporarily disabling proxy for Git operations..."

        # Unset environment proxy variables for git operations
        unset http_proxy
        unset https_proxy
        unset HTTP_PROXY
        unset HTTPS_PROXY

        # Unset git proxy configuration temporarily
        git config --global --unset http.proxy 2>/dev/null || true
        git config --global --unset https.proxy 2>/dev/null || true
    fi

    for repo in "${repos[@]}"; do
        local repo_name=$(basename "$repo" .git)
        if [ ! -d "$repo_name" ]; then
            echo "Cloning $repo..."
            if ! git clone "$repo"; then
                echo "WARNING: Failed to clone $repo_name. This might be due to a permission issue or network problem. Continuing installation..."
            else
                echo "Successfully cloned $repo_name."
            fi
        else
            echo "$repo_name already exists, skipping..."
        fi
    done

    # Restore proxy settings
    if [ "$PROXY_MODE" = false ]; then
        echo "Restoring proxy settings..."
        if [ -n "$saved_http_proxy" ]; then
            export http_proxy="$saved_http_proxy"
        fi
        if [ -n "$saved_https_proxy" ]; then
            export https_proxy="$saved_https_proxy"
        fi
        if [ -n "$saved_git_http_proxy" ]; then
            git config --global http.proxy "$saved_git_http_proxy"
        fi
        if [ -n "$saved_git_https_proxy" ]; then
            git config --global https.proxy "$saved_git_https_proxy"
        fi
    fi

    # Return to original directory
    cd "$current_dir" || { echo "Error: Could not return to original directory"; return 1; }
}

main() {

(
    set -e

if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--proxy) PROXY_MODE=true; shift;;
        --no-clone) CLONE_REPOS=false; shift;;
        -f|--force) FORCE_CLEAN=true; shift;;
        -a|--all) INSTALL_MODE="all"; shift;;
        --dev) DEV_MODE=true; shift;;
        --name)
            if [ -z "$2" ] || [[ "$2" == -* ]]; then
                echo "ERROR: --name requires a name argument." >&2
                exit 1
            fi
            VENV_NAME="$2"
            shift 2;;
        -h|--help) show_help; return 0;;
        *) echo "Unknown option: $1"; show_help; exit 1;;
    esac
done

# Validate proxy variables if proxy mode is enabled
if [ "$PROXY_MODE" = true ]; then
    if [ -z "$INTERNAL_PROXY_HOST" ] || [ -z "$INTERNAL_PYPI_MIRROR" ]; then
        echo "ERROR: Proxy mode (-p/--proxy) requires INTERNAL_PROXY_HOST and INTERNAL_PYPI_MIRROR environment variables to be set." >&2
        exit 1
    fi
fi

# Verify we're in the snomed_methods directory
if [[ ! "$(basename "$(pwd)")" == "snomed_methods" ]] && [[ ! "$(basename "$(pwd)")" == "setup" ]]; then
    echo "Warning: This script should ideally be run from the snomed_methods directory"
fi

# Clone additional repositories if needed
if [ "$CLONE_REPOS" = true ]; then
    clone_repositories
fi

# Handle force clean
if [ "$FORCE_CLEAN" = true ] && [ -d "$VENV_NAME" ]; then
    echo "Force clean enabled, removing existing virtual environment at $VENV_DIR..."
    rm -rf "$VENV_NAME"
fi

echo "Detecting Python interpreter..."
if command -v python3.10 >/dev/null; then
    PYTHON_CMD="python3.10"
elif command -v python3.11 >/dev/null; then
    PYTHON_CMD="python3.11"
elif command -v python3 >/dev/null; then
    PYTHON_CMD="python3"
elif command -v python >/dev/null; then
    PYTHON_CMD="python"
else
    echo "ERROR: Python is not installed." >&2
    exit 1
fi
echo "Using Python interpreter: $PYTHON_CMD"

echo "Creating virtual environment..."

$PYTHON_CMD -m venv "$VENV_NAME"

echo "Activating virtual environment..."
if [ ! -f "$VENV_NAME/bin/activate" ]; then
    echo "ERROR: Virtual environment activation script not found at $VENV_NAME/bin/activate"
    exit 1
fi
source "$VENV_NAME/bin/activate" || { echo "ERROR: Failed to activate virtual environment"; return 1; }

echo "Upgrading pip..."
pip_upgrade_args=("--upgrade" "pip")
if [ "$PROXY_MODE" = true ]; then
    pip_upgrade_args+=("--trusted-host" "$INTERNAL_PROXY_HOST" "-i" "$INTERNAL_PYPI_MIRROR")
fi
python -m pip install "${pip_upgrade_args[@]}"

echo "Installing project dependencies..."

extras=""
if [ "$INSTALL_MODE" = "all" ]; then
    extras="all"
fi
if [ "$DEV_MODE" = true ] && [ "$INSTALL_MODE" != "all" ]; then
    [ -n "$extras" ] && extras+=","
    extras+="dev,medcat,jupyter"
elif [ "$DEV_MODE" = false ] && [ -z "$extras" ]; then
    extras="jupyter"
fi

INSTALL_TARGET="."
[ -n "$extras" ] && INSTALL_TARGET=".[$extras]"

echo "DEV_MODE=$DEV_MODE"
echo "INSTALL_MODE=$INSTALL_MODE"
echo "EXTRAS=$extras"
echo "Running pip install -e \"$INSTALL_TARGET\""
pip_install_args=("-e" "$INSTALL_TARGET")
if [ "$PROXY_MODE" = true ]; then
    pip_install_args+=("--trusted-host" "$INTERNAL_PROXY_HOST" "-i" "$INTERNAL_PYPI_MIRROR" "--retries" "5" "--timeout" "60")
fi

pip install "${pip_install_args[@]}"

if [ "$DEV_MODE" = true ]; then
    echo "Installing pre-commit..."
    pip install --upgrade pre-commit
fi

echo "Installing SpaCy model..."
SPACY_MODEL_URL="https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.7.1/en_core_web_md-3.7.1-py3-none-any.whl"
pip_spacy_args=()
if [ "$PROXY_MODE" = true ]; then
    pip_spacy_args+=("en-core-web-md==3.7.1")
    pip_spacy_args+=("--trusted-host" "$INTERNAL_PROXY_HOST")
    pip_spacy_args+=("-i" "$INTERNAL_PYPI_MIRROR")
else
    pip_spacy_args+=("$SPACY_MODEL_URL")
fi

    pip install "${pip_spacy_args[@]}"

if [ "$DEV_MODE" = true ]; then
    echo "Installing pre-commit hooks..."
    git config --unset-all core.hooksPath 2>/dev/null || true
    python -m pip install --quiet pre-commit
    pre-commit install
fi

echo "Adding virtual environment to Jupyter kernelspec..."
python -m ipykernel install --user --name="$VENV_NAME"

echo "Deactivating virtual environment..."
deactivate

echo ""
echo "----------------------------------------------------"
echo "Installation completed successfully!"
echo "Virtual environment: $VENV_NAME"
echo "Kernel name: $VENV_NAME"
echo "To activate the environment, run: source $VENV_NAME/bin/activate"
echo "----------------------------------------------------"

)
local status=$?
return $status

}

# Pass all script arguments to the main function
main "$@"
