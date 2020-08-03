#!/bin/bash

set -x

#  NOTE: The following is required when running from Gitlab CI via slurm job
source /etc/profile.d/modules.sh

export MY_CLUSTER=`uname -n | sed -e 's/[0-9]//g' -e 's/\..*//'`
export NVBLAS_CONFIG_FILE=$(pwd)/nvblas.conf

module purge
if [ "$MY_CLUSTER" == "newell" ]; then
    export MY_GCC_VERSION=7.4.0
    export MY_CUDA_VERSION=10.2
    export MY_OPENMPI_VERSION=3.1.5
    export MY_CMAKE_VERSION=3.16.4
    export MY_MAGMA_VERSION=2.5.2_cuda10.2
    export MY_HIOP_MAGMA_DIR=/share/apps/magma/2.5.2/cuda10.2
    module load magma/$MY_MAGMA_VERSION
    export MY_NVCC_ARCH="sm_70"
else
    #  NOTE: The following is required when running from Gitlab CI via slurm job
    export MY_CLUSTER="marianas"
    export MY_GCC_VERSION=7.3.0
    export MY_CUDA_VERSION=10.1.243
    export MY_OPENMPI_VERSION=3.1.3
    export MY_CMAKE_VERSION=3.15.3
    # export MY_MAGMA_VERSION=2.5.2_cuda10.2
    # export MY_HIOP_MAGMA_DIR=/share/apps/magma/2.5.2/cuda10.2
    export MY_HIOP_MAGMA_DIR=/qfs/projects/exasgd/marianas/magma
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MY_HIOP_MAGMA_DIR/lib
    export MY_NVCC_ARCH="sm_60"
fi

module load gcc/$MY_GCC_VERSION
module load cuda/$MY_CUDA_VERSION
module load openmpi/$MY_OPENMPI_VERSION
module load cmake/$MY_CMAKE_VERSION

export MY_RAJA_DIR=/qfs/projects/exasgd/$MY_CLUSTER/raja
export MY_UMPIRE_DIR=/qfs/projects/exasgd/$MY_CLUSTER/umpire

base_path=`dirname $0`
#  NOTE: The following is required when running from Gitlab CI via slurm job
if [ -z "$SLURM_SUBMIT_DIR" ]; then
    cd $base_path          || exit 1
fi

export CMAKE_OPTIONS="\
    -DCMAKE_BUILD_TYPE=Debug \
    -DENABLE_TESTS=ON \
    -DHIOP_USE_MPI=Off \
    -DHIOP_DEEPCHECKS=ON \
    -DRAJA_DIR=$MY_RAJA_DIR \
    -DHIOP_USE_RAJA=On \
    -Dumpire_DIR=$MY_UMPIRE_DIR \
    -DHIOP_USE_UMPIRE=On \
    -DHIOP_USE_GPU=On \
    -DHIOP_MAGMA_DIR=$MY_HIOP_MAGMA_DIR \
    -DHIOP_NVCC_ARCH=$MY_NVCC_ARCH"

BUILDDIR="build"
rm -rf $BUILDDIR                            || exit 1
mkdir -p $BUILDDIR                          || exit 1
cd $BUILDDIR                                || exit 1
cmake $CMAKE_OPTIONS ..                     || exit 1
cmake --build .                             || exit 1
ctest                                       || cat Testing/Temporary/LastTest.log
exit 0
