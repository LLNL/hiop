#!/bin/bash

cleanup() {
  echo
  echo Exit code $1 caught in build script.
  echo
  if [[ "$1" == "0" ]]; then
    echo BUILD_STATUS:0
  else
    echo
    echo Failure found on line $2 in build script.
    echo
    echo BUILD_STATUS:1
  fi
}

trap 'cleanup $? $LINENO' EXIT

export BUILD=1
export TEST=1
while [[ $# -gt 0 ]]
do
  case $1 in
    --build-only|-B)
      echo
      echo Building only
      echo
      export BUILD=1
      export TEST=0
      shift
      ;;
    --test-only|-T)
      echo
      echo Testing only
      echo
      export BUILD=0
      export TEST=1
      shift
      ;;
    *)
      cat <<EOD
    Argument not found!
      
    usage: $0 [ --test-only|-T ] [ --build-only|-B ]
EOD
      exit 1
      ;;
  esac
done

set -x
x="unset"

if [[ ! -v MY_CLUSTER ]]
then
  export MY_CLUSTER=`uname -n | sed -e 's/[0-9]//g' -e 's/\..*//'`
fi
BUILDDIR="$(pwd)/build"
extra_cmake_args=""

module purge
case "$MY_CLUSTER" in
newell|newell_shared)
    export PROJ_DIR=/qfs/projects/exasgd
    export APPS_DIR=/share/apps
    #  NOTE: The following is required when running from Gitlab CI via slurm job
    source /etc/profile.d/modules.sh
    module use -a /usr/share/Modules/modulefiles
    module use -a /share/apps/modules/tools
    module use -a /share/apps/modules/compilers
    module use -a /share/apps/modules/mpi
    module use -a /etc/modulefiles
    export MY_GCC_VERSION=7.4.0
    export MY_CUDA_VERSION=10.2
    export MY_OPENMPI_VERSION=3.1.5
    export MY_CMAKE_VERSION=3.16.4
    export MY_MAGMA_VERSION=2.5.2_cuda10.2
    export MY_METIS_VERSION=5.1.0
    export MY_NVCC_ARCH="sm_70"

    module load gcc/$MY_GCC_VERSION
    module load cuda/$MY_CUDA_VERSION
    module load openmpi/$MY_OPENMPI_VERSION
    module load cmake/$MY_CMAKE_VERSION
    module load magma/$MY_MAGMA_VERSION

    export NVBLAS_CONFIG_FILE=$PROJ_DIR/$MY_CLUSTER/nvblas.conf
    export MY_RAJA_DIR=$PROJ_DIR/$MY_CLUSTER/raja
    export MY_UMPIRE_DIR=$PROJ_DIR/$MY_CLUSTER/umpire
    export MY_UMFPACK_DIR=$PROJ_DIR/$MY_CLUSTER/suitesparse
    export MY_METIS_DIR=$APPS_DIR/metis/$MY_METIS_VERSION
    export MY_HIOP_MAGMA_DIR=$APPS_DIR/magma/2.5.2/cuda10.2
    ;;
ascent)
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
    extra_cmake_args="$extra_cmake_args -DHIOP_TEST_WITH_BSUB=ON"
    ;;
marianas|dl*)
    export MY_CLUSTER="marianas"
    export PROJ_DIR=/qfs/projects/exasgd
    export APPS_DIR=/share/apps
    #  NOTE: The following is required when running from Gitlab CI via slurm job
    source /etc/profile.d/modules.sh
    module use -a /share/apps/modules/Modules/versions
    module use -a $MODULESHOME/modulefiles/environment
    module use -a $MODULESHOME/modulefiles/development/mpi
    module use -a $MODULESHOME/modulefiles/development/mlib
    module use -a $MODULESHOME/modulefiles/development/compilers
    module use -a $MODULESHOME/modulefiles/development/tools
    module use -a $MODULESHOME/modulefiles/apps
    module use -a $MODULESHOME/modulefiles/libs
    export MY_GCC_VERSION=7.3.0
    export MY_CUDA_VERSION=10.2.89
    export MY_OPENMPI_VERSION=3.1.3
    export MY_CMAKE_VERSION=3.15.3
    export MY_MAGMA_VERSION=2.5.2_cuda10.2
    export MY_METIS_VERSION=5.1.0
    export MY_NVCC_ARCH="sm_60"

    export NVBLAS_CONFIG_FILE=$PROJ_DIR/$MY_CLUSTER/nvblas.conf
    module load gcc/$MY_GCC_VERSION
    module load cuda/$MY_CUDA_VERSION
    module load openmpi/$MY_OPENMPI_VERSION
    module load cmake/$MY_CMAKE_VERSION
    module load magma/$MY_MAGMA_VERSION

    export MY_RAJA_DIR=$PROJ_DIR/$MY_CLUSTER/raja
    export MY_UMPIRE_DIR=$PROJ_DIR/$MY_CLUSTER/umpire
    export MY_UMFPACK_DIR=$PROJ_DIR/$MY_CLUSTER/suitesparse
    export MY_METIS_DIR=$APPS_DIR/metis/$MY_METIS_VERSION
    export MY_HIOP_MAGMA_DIR=$APPS_DIR/magma/2.5.2/cuda10.2
    ;;
*)
    echo
    echo Cluster not detected.
    echo
    export NVBLAS_CONFIG_FILE=$BUILDDIR/nvblas.conf
    ;;
esac

base_path=`dirname $0`
#  NOTE: The following is required when running from Gitlab CI via slurm job
if [ -z "$SLURM_SUBMIT_DIR" ]; then
    cd $base_path          || exit 1
fi

if [[ ! -v NVBLAS_CONFIG_FILE ]] || [[ ! -f "$NVBLAS_CONFIG_FILE" ]]
then
  echo "Please provide file 'nvblas.conf' in $BUILDDIR or set variable to desired location."
  exit 1
fi

module list

export CMAKE_OPTIONS="\
    -DCMAKE_BUILD_TYPE=Debug \
    -DENABLE_TESTS=ON \
    -DHIOP_USE_MPI=On \
    -DHIOP_DEEPCHECKS=ON \
    -DRAJA_DIR=$MY_RAJA_DIR \
    -DHIOP_USE_RAJA=On \
    -Dumpire_DIR=$MY_UMPIRE_DIR \
    -DHIOP_USE_UMPIRE=On \
    -DHIOP_WITH_KRON_REDUCTION=ON \
    -DHIOP_UMFPACK_DIR=$MY_UMFPACK_DIR \
    -DHIOP_METIS_DIR=$MY_METIS_DIR \
    -DHIOP_USE_GPU=ON \
    -DHIOP_MAGMA_DIR=$MY_HIOP_MAGMA_DIR \
    -DHIOP_NVCC_ARCH=$MY_NVCC_ARCH \
    $extra_cmake_args"

if [[ "$BUILD" == "1" ]]; then
  if [[ -d $BUILDDIR ]]; then
    rm -rf $BUILDDIR || exit 1
  fi
  mkdir -p $BUILDDIR || exit 1
  echo
  echo Build step
  echo
  pushd $BUILDDIR                             || exit 1
  cmake $CMAKE_OPTIONS ..                     || exit 1
  make -j || exit 1
  popd
fi

if [[ "$TEST" == "1" ]]; then
  echo
  echo Testing step
  echo

  pushd $BUILDDIR || exit 1
  ctest -VV --timeout 1800 || exit 1
  popd
fi

exit 0
