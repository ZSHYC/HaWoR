#!/usr/bin/env bash
# Source from Bash at any working directory: source scripts/server/activate.sh

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    printf 'Source this script to activate HaWoR: source scripts/server/activate.sh\n' >&2
    exit 2
fi

_hawor_activate() {
    local repo_root conda_root
    repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)" || return 1
    if [[ -f "$repo_root/.env.local" ]]; then
        source "$repo_root/.env.local" || return 1
    fi

    if ! declare -F conda >/dev/null; then
        conda_root="${HAWOR_CONDA_ROOT:-}"
        if [[ -z "$conda_root" ]] && command -v conda >/dev/null; then
            conda_root="$(conda info --base)" || return 1
        fi
        if [[ ! -f "$conda_root/etc/profile.d/conda.sh" ]]; then
            printf 'Initialize Conda first or set HAWOR_CONDA_ROOT in .env.local.\n' >&2
            return 1
        fi
        source "$conda_root/etc/profile.d/conda.sh" || return 1
    fi

    conda activate "${HAWOR_ENV_PREFIX:-hawor}" || return 1
    export HAWOR_CONDA_ROOT="$(conda info --base)"
    export HAWOR_ENV_PREFIX="$CONDA_PREFIX"
    export CUDA_HOME="$CONDA_PREFIX"
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export PYOPENGL_PLATFORM=egl
    export MPLBACKEND=Agg
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
    export MAX_JOBS="${MAX_JOBS:-8}"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
    export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/pip}"
}

if _hawor_activate; then
    unset -f _hawor_activate
else
    unset -f _hawor_activate
    return 1
fi
