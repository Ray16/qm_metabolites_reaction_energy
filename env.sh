#!/usr/bin/env bash
# Source this to put ORCA 6.1.1 + matching OpenMPI 4.1.8 on PATH/LD_LIBRARY_PATH.
#   source thermodynamic_calc/env.sh
#
# ORCA must be invoked by ABSOLUTE path for parallel (MPI) runs to start.

export ORCA_ROOT="${ORCA_ROOT:-/nfs/lambda_stor_01/homes/rzhu/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg}"
export OPENMPI_ROOT="${OPENMPI_ROOT:-/nfs/lambda_stor_01/homes/rzhu/openmpi-4.1.8-install}"
export XTB_BIN="${XTB_BIN:-/nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/xtb/bin/xtb}"

export PATH="$OPENMPI_ROOT/bin:$ORCA_ROOT:$ORCA_ROOT/lib:$PATH"
export LD_LIBRARY_PATH="$OPENMPI_ROOT/lib:$ORCA_ROOT/lib:$ORCA_ROOT:$LD_LIBRARY_PATH"

# QM scratch on LOCAL disk (NFS home is ~full and scratch is large/IO-heavy).
export QM_SCRATCH="${QM_SCRATCH:-/tmp/qm_thermo_scratch}"
mkdir -p "$QM_SCRATCH"

echo "ORCA    : $ORCA_ROOT/orca"
echo "OpenMPI : $($OPENMPI_ROOT/bin/mpirun --version 2>/dev/null | head -1)"
echo "xtb     : $XTB_BIN"
echo "scratch : $QM_SCRATCH"
