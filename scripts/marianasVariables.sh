if [ ! -v BUILDDIR ]; then
  echo BUILDDIR is not set! Your paths may be misconfigured.
fi
export MY_CLUSTER="marianas"
export PROJ_DIR=/qfs/projects/exasgd
export APPS_DIR=/share/apps
export SPACK_ARCH=linux-centos7-broadwell
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
module use -a $PROJ_DIR/src/spack/share/spack/modules/$SPACK_ARCH/
export MY_GCC_VERSION=7.3.0
export MY_CUDA_VERSION=10.2.89
export MY_OPENMPI_VERSION=3.1.3
export MY_CMAKE_VERSION=3.15.3
export MY_MAGMA_VERSION=2.5.4-gcc-7.3.0-bwwsayw
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
export MY_COINHSL_DIR=$PROJ_DIR/$MY_CLUSTER/ipopt
EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DBLAS_LIBRARIES=/usr/lib64/libopenblas.so"
