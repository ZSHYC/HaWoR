#!/usr/bin/env bash
# Conda runs this before changing CONDA_SHLVL or switching environments.
_hawor_restore_environment() {
    local name key saved_set saved_value saved_exported
    for name in CUDA_HOME LD_LIBRARY_PATH PYOPENGL_PLATFORM MPLBACKEND \
                TORCH_CUDA_ARCH_LIST MAX_JOBS OMP_NUM_THREADS; do
        key="_HAWOR_BACKUP_${CONDA_SHLVL}_${name}"
        saved_set="${key}_SET"
        saved_value="${key}_VALUE"
        saved_exported="${key}_EXPORTED"
        # An unmatched hook must not erase a caller's environment.
        if [[ ! -v "$saved_set" ]]; then
            continue
        fi
        if [[ "${!saved_set}" == 1 ]]; then
            printf -v "$name" '%s' "${!saved_value}"
            if [[ "${!saved_exported}" == 1 ]]; then
                export "$name"
            else
                export -n "$name"
            fi
        else
            unset "$name"
        fi
        unset "$saved_set" "$saved_value" "$saved_exported"
    done
}
_hawor_restore_environment
unset -f _hawor_restore_environment
