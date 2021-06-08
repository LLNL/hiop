if [ ! -v BUILDDIR ]; then
  echo BUILDDIR is not set! Your paths may be misconfigured.
fi
export PROJ_DIR=/qfs/projects/exasgd
export APPS_DIR=/share/apps
export SPACK_ARCH=linux-rhel7-power9le
#  NOTE: The following is required when running from Gitlab CI via slurm job
source /etc/profile.d/modules.sh
module use -a /usr/share/Modules/modulefiles
module use -a /share/apps/modules/tools
module use -a /share/apps/modules/compilers
module use -a /share/apps/modules/mpi
module use -a /etc/modulefiles
module use -a $PROJ_DIR/src/spack/share/spack/modules/$SPACK_ARCH/
export MY_GCC_VERSION=7.4.0
export MY_CUDA_VERSION=10.2
export MY_OPENMPI_VERSION=3.1.5
export MY_CMAKE_VERSION=3.19.6
export MY_MAGMA_VERSION=2.5.4-gcc-7.4.0-jss5aen
export MY_METIS_VERSION=5.1.0
export MY_NVCC_ARCH="70"

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
export MY_COINHSL_DIR=$PROJ_DIR/$MY_CLUSTER/ipopt
