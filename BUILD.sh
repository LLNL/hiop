#!/bin/bash

set -x

#  NOTE: The following is required when running from Gitlab CI via slurm job
source /etc/profile.d/modules.sh

export MY_CLUSTER=`uname -n | sed -e 's/[0-9]//g' -e 's/\..*//'`

if [ "$MY_CLUSTER" == "newell" ]; then
    export MY_GCC_VERSION=7.4.0
    export MY_CUDA_VERSION=9.2
    export MY_OPENMPI_VERSION=3.1.5
    export MY_CMAKE_VERSION=3.16.4
else
    #  NOTE: The following is required when running from Gitlab CI via slurm job
    export MY_CLUSTER="marianas"
    export MY_GCC_VERSION=7.3.0
    export MY_CUDA_VERSION=9.2.148
    export MY_OPENMPI_VERSION=3.1.3
    export MY_CMAKE_VERSION=3.15.3
fi

module purge
module load gcc/$MY_GCC_VERSION
module load cuda/$MY_CUDA_VERSION
module load openmpi/$MY_OPENMPI_VERSION
module load cmake/$MY_CMAKE_VERSION

base_path=`dirname $0`
#  NOTE: The following is required when running from Gitlab CI via slurm job
if [ -z "$SLURM_SUBMIT_DIR" ]; then
    cd $base_path          || exit 1
fi

#export MAKEFLAGS="-j 8"
#export CMAKE_OPTIONS="-GNinja -DCMAKE_BUILD_TYPE=Debug -DENABLE_TESTS=ON"
export CMAKE_OPTIONS="-DCMAKE_BUILD_TYPE=Debug -DHIOP_WITH_MAKETEST=ON -DHIOP_USE_MPI=ON -DHIOP_DEEPCHECKS=ON"

BUILDDIR="build"
rm -rf $BUILDDIR                || exit 1
mkdir -p $BUILDDIR              || exit 1
cd $BUILDDIR                    || exit 1
cmake $CMAKE_OPTIONS ..         || exit 1
cmake --build .                 || exit 1
ctest                           || cat Testing/Temporary/LastTest.log
exit 0
