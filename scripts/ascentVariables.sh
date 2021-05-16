
if [ ! -v BUILDDIR ]; then
  echo BUILDDIR is not set! Your paths may be misconfigured.
fi

export MY_CLUSTER=ascent
export PROJ_DIR=/gpfs/wolf/proj-shared/csc359
source $PROJ_DIR/src/spack/share/spack/setup-env.sh
module purge
module load cuda/10.2.89
module use $PROJ_DIR/$MY_CLUSTER/Modulefiles/Core
module load exasgd-base
module load gcc/7.4.0
export CC=gcc CXX=g++ # For some reason this is not picked up by cmake with this module
module load spectrum-mpi/10.3.1.2-20200121
module load openblas/0.3.9-omp
module load cmake/3.18.2

ls $PROJ_DIR/src/spack/var/spack/environments/*

spack env activate hiop-v0-4-3-deps

export MY_NVCC_ARCH="sm_70"
export EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DHIOP_NVCC_ARCH=$MY_NVCC_ARCH"

if [[ ! -f $BUILDDIR/nvblas.conf ]]; then
  cat > $BUILDDIR/nvblas.conf <<-EOD
  NVBLAS_LOGFILE  nvblas.log
  NVBLAS_CPU_BLAS_LIB /autofs/nccsopen-svm1_sw/ascent/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/gcc-7.4.0/openblas-0.3.9-cjxfkk67xpigoo4qo77tzvigloabwuvr/lib/libopenblas.so
  NVBLAS_GPU_LIST ALL
  NVBLAS_TILE_DIM 2048
  NVBLAS_AUTOPIN_MEM_ENABLED
EOD
fi
export NVBLAS_CONFIG_FILE=$BUILDDIR/nvblas.conf
EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DHIOP_TEST_WITH_BSUB=ON -DCMAKE_CUDA_ARCHITECTURES=70"
