#!/bin/bash
# ===========================================================================
# setup_venv.sh — Bootstrap the AERIS inference venv (one-time per filesystem)
#
# Idempotent: re-running is a no-op once .bootstrapped sentinel exists.
# Source this script from your PBS job (or run it directly to pre-build).
#
# Requirements:
#   module load frameworks   (provides PyTorch XPU, mpi4py, oneCCL)
# ===========================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
VENV_ROOT="${REPO_DIR}/data/cache/venvs/aeris"
SENTINEL="${VENV_ROOT}/.bootstrapped"

mkdir -p "${VENV_ROOT}"

if [[ -f "${SENTINEL}" ]]; then
    echo "[setup_venv] venv already built at ${VENV_ROOT}/.venv — skipping"
    # Works whether sourced (return) or executed as subprocess (exit)
    [[ "${BASH_SOURCE[0]}" != "${0}" ]] && return 0 || exit 0
fi

echo "[setup_venv] bootstrapping venv at ${VENV_ROOT}/.venv"

if ! command -v python3 >/dev/null 2>&1; then
    echo "[setup_venv] ERROR: python3 not found. Did you run 'module load frameworks'?" >&2
    return 1 2>/dev/null || exit 1
fi

FWK_PY=$(command -v python3)
FWK_VER=$(${FWK_PY} -c "import sys; print('%d.%d.%d' % sys.version_info[:3])")
echo "[setup_venv] frameworks python: ${FWK_PY} (${FWK_VER})"

# --system-site-packages inherits torch (XPU build), mpi4py, and oneCCL
# from the frameworks module — these are not available on PyPI for Aurora XPU.
${FWK_PY} -m venv "${VENV_ROOT}/.venv" --system-site-packages
source "${VENV_ROOT}/.venv/bin/activate"

# Constrain torch version so pip's dependency resolver doesn't pull a
# generic CUDA wheel from PyPI and shadow the XPU framework build.
cat > /tmp/aeris_constraints.txt <<EOF
torch==2.10.0a0
EOF

pip install --quiet --no-cache-dir \
    --constraint /tmp/aeris_constraints.txt \
    omegaconf einops h5py h5netcdf \
    xarray netCDF4 nest-asyncio \
    "PyYAML>=6.0"

rm -f /tmp/aeris_constraints.txt

touch "${SENTINEL}"
echo "[setup_venv] done"
