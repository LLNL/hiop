
if [ ! -v BUILDDIR ]; then
  echo BUILDDIR is not set! Your paths may be misconfigured.
fi

module purge
module load cuda/11
module use /gpfs/wolf/proj-shared/csc359/ascent/Modulefiles/Core
module load exasgd-base
module load gcc-ext/7.4.0
module load spectrum-mpi-ext
module load openblas
module load magma/2.5.3-cuda11
module load metis
module load mpfr
module load suitesparse
module load cmake/3.18.2
module load raja
module load umpire
module load valgrind/3.14.0
export MY_RAJA_DIR=$RAJA_ROOT
export MY_UMPIRE_DIR=$UMPIRE_ROOT
export MY_METIS_DIR=$OLCF_METIS_ROOT
export MY_HIOP_MAGMA_DIR=$MAGMA_ROOT
export MY_UMFPACK_DIR=$SUITESPARSE_ROOT
export MY_NVCC_ARCH="sm_70"

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
