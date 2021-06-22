if [ ! -v BUILDDIR ]; then
  echo BUILDDIR is not set! Your paths may be misconfigured.
fi
export MY_CLUSTER="marianas"
export PROJ_DIR=/qfs/projects/exasgd
export APPS_DIR=/share/apps
export SPACK_ARCH=linux-centos7-broadwell

#  NOTE: The following is required when running from Gitlab CI via slurm job
source /etc/profile.d/modules.sh
(
module use -a /share/apps/modules/Modules/versions
module use -a $MODULESHOME/modulefiles/environment
module use -a $MODULESHOME/modulefiles/development/mpi
module use -a $MODULESHOME/modulefiles/development/mlib
module use -a $MODULESHOME/modulefiles/development/compilers
module use -a $MODULESHOME/modulefiles/development/tools
module use -a $MODULESHOME/modulefiles/apps
module use -a $MODULESHOME/modulefiles/libs
module use -a $PROJ_DIR/src/spack/share/spack/modules/$SPACK_ARCH/
) 2>&1 1>&/dev/null

export NVBLAS_CONFIG_FILE=$PROJ_DIR/$MY_CLUSTER/nvblas.conf

# Load spack-built modules
module load exasgd-arpack-ng/3.8.0/openmpi-3.1.3/gcc-7.3.0-zirdafa
module load exasgd-autoconf-archive/2019.01.06/gcc-7.3.0-bzoccad
module load exasgd-blt/0.3.6/gcc-7.3.0-yvfnprq
module load exasgd-butterflypack/1.2.1/openmpi-3.1.3/gcc-7.3.0-2syic2u
module load exasgd-camp/0.1.0/cuda-10.2.89/gcc-7.3.0-bvsl36m
module load exasgd-cmake/3.20.2/gcc-7.3.0-ymp2go7
module load exasgd-coinhsl/2015.06.23/gcc-7.3.0-rkvwgny
module load exasgd-cub/1.12.0-rc0/gcc-7.3.0-o5omfqk
module load exasgd-cuda/10.2.89/gcc-7.3.0-fytuxuh
module load exasgd-gmp/6.2.1/gcc-7.3.0-uimke6b
module load exasgd-ipopt/3.12.10/gcc-7.3.0-kyxesp7
module load exasgd-magma/2.5.4/cuda-10.2.89/gcc-7.3.0-25cv4mc
module load exasgd-metis/5.1.0/gcc-7.3.0-232rotu
module load exasgd-mpfr/4.1.0/gcc-7.3.0-3rrse4o
module load exasgd-ncurses/6.2/gcc-7.3.0-h2zctno
module load exasgd-netlib-scalapack/2.1.0/openmpi-3.1.3/gcc-7.3.0-kcsvzxm
module load exasgd-openblas/0.3.5/gcc-7.3.0-ug3gdtf
module load exasgd-openmpi/3.1.3/gcc-7.3.0-etzhtfg
module load exasgd-parmetis/4.0.3/openmpi-3.1.3/gcc-7.3.0-4atunu3
module load exasgd-perl/5.26.0/gcc-7.3.0-wog3ysl
module load exasgd-pkgconf/1.7.4/gcc-7.3.0-n2z4txg
module load exasgd-raja/0.13.0/cuda-10.2.89/gcc-7.3.0-jemkyui
module load exasgd-strumpack/5.1.1/cuda-10.2.89/openmpi-3.1.3/gcc-7.3.0-4fqr6cn
module load exasgd-suite-sparse/5.10.1/gcc-7.3.0-kj6x5bx
module load exasgd-texinfo/6.5/gcc-7.3.0-l44ipkv
module load exasgd-umpire/4.1.2/cuda-10.2.89/gcc-7.3.0-3uphncx
module load exasgd-zfp/0.5.5/gcc-7.3.0-h3p6smb

# Load system modules
module load gcc/7.3.0
module load cuda/10.2.89
module load openmpi/3.1.3

export EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DCMAKE_CUDA_ARCHITECTURES=60"
