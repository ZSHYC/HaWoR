"""Check real Conda activation, reactivation, switching and variable restoration."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


VARIABLES = [
    "CUDA_HOME", "LD_LIBRARY_PATH", "PYOPENGL_PLATFORM", "MPLBACKEND",
    "TORCH_CUDA_ARCH_LIST", "MAX_JOBS", "OMP_NUM_THREADS",
]

SNAPSHOT = """
import json, os, sys
from pathlib import Path
names = json.loads(sys.argv[2])
state = {name: os.environ.get(name) for name in names}
state['hook_backups'] = sorted(k for k in os.environ if k.startswith('_HAWOR_BACKUP_'))
Path(sys.argv[1]).write_text(json.dumps(state))
"""

CHECK_ACTIVE = """
import os
from pathlib import Path
prefix = os.environ['CONDA_PREFIX']
assert os.environ.get('CUDA_HOME') == prefix, 'CUDA_HOME was not set by activation'
assert os.environ.get('LD_LIBRARY_PATH', '').split(':')[0] == str(Path(prefix) / 'lib')
assert os.environ.get('PYOPENGL_PLATFORM') == 'egl'
assert os.environ.get('MPLBACKEND') == 'Agg'
assert all(os.environ.get(k) for k in ['TORCH_CUDA_ARCH_LIST', 'MAX_JOBS', 'OMP_NUM_THREADS'])
"""

SHELL = r"""
set -e
source "$1/etc/profile.d/conda.sh"
while [[ "${CONDA_SHLVL:-0}" -gt 0 ]]; do conda deactivate; done
test_python="$2"; snapshot_code="$3"; output_dir="$4"; target_env="$5"; names="$6"
snapshot() { "$test_python" -c "$snapshot_code" "$output_dir/$1.json" "$names"; }
unset CUDA_HOME LD_LIBRARY_PATH PYOPENGL_PLATFORM MPLBACKEND TORCH_CUDA_ARCH_LIST MAX_JOBS OMP_NUM_THREADS
snapshot before_unset
conda activate "$target_env"
python -c "$7"
conda deactivate
snapshot after_unset

export CUDA_HOME='/tmp/previous cuda'
export LD_LIBRARY_PATH='/tmp/previous libs:$literal'
export PYOPENGL_PLATFORM=''
export MPLBACKEND=svg
export -n MPLBACKEND
export TORCH_CUDA_ARCH_LIST=8.0
export MAX_JOBS=3
export OMP_NUM_THREADS=2
snapshot before_values
conda activate "$target_env"
python -c "$7"
conda activate "$target_env"
python -c "$7"
conda activate base
[[ "$MPLBACKEND" == svg ]]
snapshot after_switch
conda deactivate

conda activate base
conda activate --stack "$target_env"
python -c "$7"
conda deactivate
[[ "$MPLBACKEND" == svg ]]
snapshot after_stack_base
conda deactivate

conda activate "$target_env"
conda activate --stack base
conda deactivate
python -c "$7"
conda deactivate
[[ "$MPLBACKEND" == svg ]]
snapshot after_stack
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conda-root", type=Path)
    parser.add_argument("--env", default="hawor", help="Conda environment name or prefix")
    args = parser.parse_args()
    conda_root = args.conda_root
    if conda_root is None:
        conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
        if not conda:
            parser.error("Initialize Conda first, or pass --conda-root")
        conda_root = Path(subprocess.check_output([conda, "info", "--base"], text=True).strip())
    with tempfile.TemporaryDirectory(prefix="hawor-conda-check-") as output_dir:
        result = subprocess.run(
            ["bash", "--noprofile", "--norc", "-c", SHELL, "hawor-conda-check",
             str(conda_root), sys.executable, SNAPSHOT, output_dir, args.env,
             json.dumps(VARIABLES), CHECK_ACTIVE], check=False,
        )
        if result.returncode:
            raise SystemExit(result.returncode)
        states = {p.stem: json.loads(p.read_text()) for p in Path(output_dir).glob("*.json")}
        assert states["before_unset"] == states["after_unset"], "Unset variables were not restored"
        assert states["before_values"] == states["after_switch"], "Switching to base leaked variables"
        assert states["before_values"] == states["after_stack_base"], "Stacking on base leaked variables"
        assert states["before_values"] == states["after_stack"], "Stacked activation leaked variables"
        print("PASS: activation, reactivation, switching, stacking and exact variable restoration")


if __name__ == "__main__":
    main()
