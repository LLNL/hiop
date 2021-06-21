
if [ ! -v BUILDDIR ]; then
  echo BUILDDIR is not set! Your paths may be misconfigured.
fi

export MY_CLUSTER=ascent
export PROJ_DIR=/gpfs/wolf/proj-shared/csc359
source $PROJ_DIR/src/spack/share/spack/setup-env.sh

ls $PROJ_DIR/src/spack/var/spack/environments/*

spack env activate hiop-v0-4-3-deps

module purge
module load cuda/11.0.2
module load gcc/7.4.0
module load spectrum-mpi/10.3.1.2-20200121
module load cmake/3.18.2

#These are not picked up by the module for some reason
export CC=/sw/ascent/gcc/7.4.0/bin/gcc
export CXX=/sw/ascent/gcc/7.4.0/bin/g++
export FC=/sw/ascent/gcc/7.4.0/bin/gfortran

export EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_TOOLKIT_ROOT_DIR \
  "

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
EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DHIOP_TEST_WITH_BSUB=ON"
module load cuda/11.0.2
