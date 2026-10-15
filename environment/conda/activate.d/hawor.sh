#!/usr/bin/env bash
# Installed in $CONDA_PREFIX/etc/conda/activate.d; run by conda activate.
# Keep one backup per activation level, including unset/empty/export state.
_hawor_save_environment() {
    local name key declaration
    for name in CUDA_HOME LD_LIBRARY_PATH PYOPENGL_PLATFORM MPLBACKEND \
                TORCH_CUDA_ARCH_LIST MAX_JOBS OMP_NUM_THREADS; do
        key="_HAWOR_BACKUP_${CONDA_SHLVL}_${name}"
        # Returning from a stacked environment reruns this hook at the old level.
        # Retain that level's original backup until its matching deactivation.
        if [[ -v "${key}_SET" ]]; then
            continue
        fi
        if [[ -v "$name" ]]; then
            export "${key}_SET=1" "${key}_VALUE=${!name}"
            declaration="$(declare -p "$name")"
            if [[ "$declaration" =~ ^declare\ -[^[:space:]]*x ]]; then
                export "${key}_EXPORTED=1"
            else
                export "${key}_EXPORTED=0"
            fi
        else
            export "${key}_SET=0"
            unset "${key}_VALUE" "${key}_EXPORTED"
        fi
    done
}
_hawor_save_environment
unset -f _hawor_save_environment

export CUDA_HOME="$CONDA_PREFIX"
case "${LD_LIBRARY_PATH:-}" in
    "$CONDA_PREFIX/lib"|"$CONDA_PREFIX/lib:"*) ;;
    *) LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
esac
export LD_LIBRARY_PATH
export PYOPENGL_PLATFORM=egl
export MPLBACKEND=Agg
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export MAX_JOBS="${MAX_JOBS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
