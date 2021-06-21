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

# Activate spack environment
source $PROJ_DIR/src/spack/share/spack/setup-env.sh
spack env activate hiop-v0-4-2-deps-newell

# Load system modules
module load gcc/7.4.0
module load cuda/10.2
module load openmpi/3.1.5
module load cmake/3.19.6

export MY_NVCC_ARCH="70"
export NVBLAS_CONFIG_FILE=$PROJ_DIR/$MY_CLUSTER/nvblas.conf
