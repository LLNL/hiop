
if [ ! -v BUILDDIR ]; then
  echo BUILDDIR is not set! Your paths may be misconfigured.
fi

export PROJ_DIR=/gpfs/wolf/proj-shared/csc359
source $PROJ_DIR/src/spack/share/spack/setup-env.sh
module purge
module load cuda/10.2.89
module use $PROJ_DIR/$MY_CLUSTER/Modulefiles/Core
module load exasgd-base
module load gcc-ext/7.4.0
module load spectrum-mpi-ext
module load openblas
module load cmake/3.18.2

ls $PROJ_DIR/src/spack/var/spack/environments/*

spack env activate exago-v0-99-2-hiop-v0-3-99-2

export MY_NVCC_ARCH="sm_70"
export EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DHIOP_NVCC_ARCH=$MY_NVCC_ARCH"

if [[ ! -f $BUILDDIR/nvblas.conf ]]; then
  cat > $BUILDDIR/nvblas.conf <<EOD
  NVBLAS_LOGFILE  nvblas.log
  NVBLAS_CPU_BLAS_LIB  /gpfs/wolf/proj-shared/csc359/ascent/Compiler/gcc-7.4.0/openblas/0.3.10/lib/libopenblas.so
  NVBLAS_GPU_LIST ALL
  NVBLAS_TILE_DIM 2048
  NVBLAS_AUTOPIN_MEM_ENABLED
EOD
fi
export NVBLAS_CONFIG_FILE=$BUILDDIR/nvblas.conf
EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DHIOP_TEST_WITH_BSUB=ON"
