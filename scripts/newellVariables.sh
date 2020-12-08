if [ ! -v BUILDDIR ]; then
  echo BUILDDIR is not set! Your paths may be misconfigured.
fi
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
export MY_COINHSL_DIR=$PROJ_DIR/$MY_CLUSTER/ipopt
