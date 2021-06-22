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

# Load spack-built modules
module load exasgd-arpack-ng/3.8.0/openmpi-3.1.5/gcc-7.4.0-yl7lsxq
module load exasgd-autoconf-archive/2019.01.06/gcc-7.4.0-zr3h7p2
module load exasgd-blt/0.3.6/gcc-7.4.0-z4yhwmy
module load exasgd-butterflypack/1.2.1/openmpi-3.1.5/gcc-7.4.0-iaw77e6
module load exasgd-camp/0.1.0/cuda-10.2.89-system/gcc-7.4.0-fvkaniz
module load exasgd-coinhsl/2015.06.23/gcc-7.4.0-qhwhtdc
module load exasgd-cub/1.12.0-rc0/gcc-7.4.0-5lcegaf
module load exasgd-gmp/6.2.1/gcc-7.4.0-6svtfvr
module load exasgd-ipopt/3.12.10/gcc-7.4.0-sfzycsu
module load exasgd-magma/2.5.4/cuda-10.2.89-system/gcc-7.4.0-bikffe2
module load exasgd-metis/5.1.0/gcc-7.4.0-7cjo5kb
module load exasgd-mpfr/4.1.0/gcc-7.4.0-bkusb67
module load exasgd-netlib-scalapack/2.1.0/openmpi-3.1.5/gcc-7.4.0-3uycg6f
module load exasgd-openblas/0.3.5/gcc-7.4.0-72rh6zx
module load exasgd-parmetis/4.0.3/openmpi-3.1.5/gcc-7.4.0-ym6nkdq
module load exasgd-perl/5.26.0/gcc-7.4.0-j742qcl
module load exasgd-pkgconf/1.7.4/gcc-7.4.0-5ios5aw
module load exasgd-raja/0.13.0/cuda-10.2.89-system/gcc-7.4.0-uj7zop4
module load exasgd-strumpack/5.1.1/cuda-10.2.89-system/openmpi-3.1.5/gcc-7.4.0-x55sdhv
module load exasgd-suite-sparse/5.8.1/gcc-7.4.0-ij4zlyb
module load exasgd-texinfo/6.5/gcc-7.4.0-r24oezq
module load exasgd-umpire/4.1.1/cuda-10.2.89-system/gcc-7.4.0-njlsagd
module load exasgd-zfp/0.5.5/gcc-7.4.0-av4beua

# Load system modules
module load gcc/7.4.0
module load cuda/10.2
module load openmpi/3.1.5
module load cmake/3.19.6

export EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DCMAKE_CUDA_ARCHITECTURES=70"
export NVBLAS_CONFIG_FILE=$PROJ_DIR/$MY_CLUSTER/nvblas.conf
