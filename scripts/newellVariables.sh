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
# arpack-ng@3.8.0%gcc@7.4.0+mpi+shared arch=linux-rhel7-power9le
module load exasgd-arpack-ng/3.8.0/openmpi-3.1.5/gcc-7.4.0-yl7lsxq
# autoconf-archive@2019.01.06%gcc@7.4.0 arch=linux-rhel7-power9le
module load exasgd-autoconf-archive/2019.01.06/gcc-7.4.0-zr3h7p2
# blt@0.3.6%gcc@7.4.0 arch=linux-rhel7-power9le
module load exasgd-blt/0.3.6/gcc-7.4.0-z4yhwmy
# butterflypack@1.2.1%gcc@7.4.0~ipo+shared build_type=RelWithDebInfo arch=linux-rhel7-power9le
module load exasgd-butterflypack/1.2.1/openmpi-3.1.5/gcc-7.4.0-iaw77e6
# camp@0.1.0%gcc@7.4.0+cuda~ipo~rocm~tests amdgpu_target=none build_type=RelWithDebInfo cuda_arch=none arch=linux-rhel7-power9le
module load exasgd-camp/0.1.0/cuda-10.2.89-system/gcc-7.4.0-fvkaniz
# coinhsl@2015.06.23%gcc@7.4.0+blas arch=linux-rhel7-power9le
module load exasgd-coinhsl/2015.06.23/gcc-7.4.0-qhwhtdc
# cub@1.12.0-rc0%gcc@7.4.0 arch=linux-rhel7-power9le
module load exasgd-cub/1.12.0-rc0/gcc-7.4.0-5lcegaf
# gmp@6.2.1%gcc@7.4.0 arch=linux-rhel7-power9le
module load exasgd-gmp/6.2.1/gcc-7.4.0-6svtfvr
# ipopt@3.12.10%gcc@7.4.0+coinhsl~debug~metis~mumps arch=linux-rhel7-power9le
module load exasgd-ipopt/3.12.10/gcc-7.4.0-sfzycsu
# magma@2.5.4%gcc@7.4.0+cuda+fortran~ipo+shared build_type=RelWithDebInfo cuda_arch=70 arch=linux-rhel7-power9le
module load exasgd-magma/2.5.4/cuda-10.2.89-system/gcc-7.4.0-bikffe2
# metis@5.1.0%gcc@7.4.0~gdb~int64~real64+shared build_type=Release patches=4991da938c1d3a1d3dea78e49bbebecba00273f98df2a656e38b83d55b281da1,b1225da886605ea558db7ac08dd8054742ea5afe5ed61ad4d0fe7a495b1270d2 arch=linux-rhel7-power9le
module load exasgd-metis/5.1.0/gcc-7.4.0-7cjo5kb
# mpfr@4.1.0%gcc@7.4.0 arch=linux-rhel7-power9le
module load exasgd-mpfr/4.1.0/gcc-7.4.0-bkusb67
# netlib-scalapack@2.1.0%gcc@7.4.0~ipo~pic+shared build_type=Release patches=1c9ce5fee1451a08c2de3cc87f446aeda0b818ebbce4ad0d980ddf2f2a0b2dc4,f2baedde688ffe4c20943c334f580eb298e04d6f35c86b90a1f4e8cb7ae344a2 arch=linux-rhel7-power9le
module load exasgd-netlib-scalapack/2.1.0/openmpi-3.1.5/gcc-7.4.0-3uycg6f
# openblas@0.3.5%gcc@7.4.0~bignuma~consistent_fpcsr~ilp64+locking+pic+shared patches=865703b4f405543bbd583413fdeff2226dfda908be33639276c06e5aa7ae2cae threads=none arch=linux-rhel7-power9le
module load exasgd-openblas/0.3.5/gcc-7.4.0-72rh6zx
# parmetis@4.0.3%gcc@7.4.0~gdb~int64~ipo+shared build_type=RelWithDebInfo patches=4f892531eb0a807eb1b82e683a416d3e35154a455274cf9b162fb02054d11a5b,50ed2081bc939269689789942067c58b3e522c269269a430d5d34c00edbc5870,704b84f7c7444d4372cb59cca6e1209df4ef3b033bc4ee3cf50f369bce972a9d arch=linux-rhel7-power9le
module load exasgd-parmetis/4.0.3/openmpi-3.1.5/gcc-7.4.0-ym6nkdq
# perl@5.26.0%gcc@7.4.0+cpanm~shared~threads patches=0eac10ed90aeb0459ad8851f88081d439a4e41978e586ec743069e8b059370ac arch=linux-rhel7-power9le
module load exasgd-perl/5.26.0/gcc-7.4.0-j742qcl
# pkgconf@1.7.4%gcc@7.4.0 arch=linux-rhel7-power9le
module load exasgd-pkgconf/1.7.4/gcc-7.4.0-5ios5aw
# raja@0.13.0%gcc@7.4.0+cuda~examples~exercises~ipo+openmp~rocm+shared~tests amdgpu_target=none build_type=RelWithDebInfo cuda_arch=none arch=linux-rhel7-power9le
module load exasgd-raja/0.13.0/cuda-10.2.89-system/gcc-7.4.0-uj7zop4
# strumpack@5.1.1%gcc@7.4.0~build_dev_tests~build_tests+butterflypack+c_interface~count_flops+cuda~ipo+mpi+openmp+parmetis~rocm~scotch~shared~slate~task_timers+zfp amdgpu_target=none build_type=RelWithDebInfo cuda_arch=none arch=linux-rhel7-power9le
module load exasgd-strumpack/5.1.1/cuda-10.2.89-system/openmpi-3.1.5/gcc-7.4.0-x55sdhv
# suite-sparse@5.8.1%gcc@7.4.0~cuda~openmp+pic~tbb arch=linux-rhel7-power9le
module load exasgd-suite-sparse/5.8.1/gcc-7.4.0-ij4zlyb
# texinfo@6.5%gcc@7.4.0 patches=12f6edb0c6b270b8c8dba2ce17998c580db01182d871ee32b7b6e4129bd1d23a,1732115f651cff98989cb0215d8f64da5e0f7911ebf0c13b064920f088f2ffe1 arch=linux-rhel7-power9le
module load exasgd-texinfo/6.5/gcc-7.4.0-r24oezq
# umpire@4.1.1%gcc@7.4.0+c+cuda~deviceconst~examples~fortran~ipo~numa+openmp~rocm~shared amdgpu_target=none build_type=RelWithDebInfo cuda_arch=none patches=135bbc7d2f371531f432672b115ac0a407968aabfffc5b8a941db9b493dbf81f,7d912d31cd293df005ba74cb96c6f3e32dc3d84afff49b14509714283693db08 tests=none arch=linux-rhel7-power9le
module load exasgd-umpire/4.1.1/cuda-10.2.89-system/gcc-7.4.0-njlsagd
# zfp@0.5.5%gcc@7.4.0~aligned~c~cuda~fasthash~fortran~ipo~openmp~profile~python+shared~strided~twoway bsws=64 build_type=RelWithDebInfo cuda_arch=none arch=linux-rhel7-power9le
module load exasgd-zfp/0.5.5/gcc-7.4.0-av4beua

# Load system modules
module load gcc/7.4.0
module load cuda/10.2
module load openmpi/3.1.5
module load cmake/3.19.6

export EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DCMAKE_CUDA_ARCHITECTURES=70"
export NVBLAS_CONFIG_FILE=$PROJ_DIR/$MY_CLUSTER/nvblas.conf
