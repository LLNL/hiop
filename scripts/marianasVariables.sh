#  NOTE: The following is required when running from Gitlab CI via slurm job
source /etc/profile.d/modules.sh
module use -a /qfs/projects/exasgd/src/spack/share/spack/modules/linux-centos7-broadwell/
module use -a /qfs/projects/exasgd/src/spack/share/spack/modules/linux-centos7-x86_64/

# Load spack-built modules
# arpack-ng@3.8.0%gcc@7.3.0+mpi+shared arch=linux-centos7-broadwell
module load exasgd-arpack-ng/3.8.0/openmpi-3.1.3/gcc-7.3.0-zirdafa
# autoconf-archive@2019.01.06%gcc@7.3.0 arch=linux-centos7-broadwell
module load exasgd-autoconf-archive/2019.01.06/gcc-7.3.0-bzoccad
# blt@0.3.6%gcc@7.3.0 arch=linux-centos7-broadwell
module load exasgd-blt/0.3.6/gcc-7.3.0-yvfnprq
# butterflypack@1.2.1%gcc@7.3.0~ipo+shared build_type=RelWithDebInfo arch=linux-centos7-broadwell
module load exasgd-butterflypack/1.2.1/openmpi-3.1.3/gcc-7.3.0-2syic2u
# camp@0.1.0%gcc@7.3.0+cuda~ipo~rocm~tests amdgpu_target=none build_type=RelWithDebInfo cuda_arch=none arch=linux-centos7-broadwell
module load exasgd-camp/0.1.0/cuda-10.2.89/gcc-7.3.0-bvsl36m
# cmake@3.20.2%gcc@7.3.0~doc+ncurses~openssl+ownlibs~qt build_type=Release arch=linux-centos7-broadwell
module load exasgd-cmake/3.20.2/gcc-7.3.0-ymp2go7
# coinhsl@2015.06.23%gcc@7.3.0+blas arch=linux-centos7-broadwell
module load exasgd-coinhsl/2015.06.23/gcc-7.3.0-rkvwgny
# cub@1.12.0-rc0%gcc@7.3.0 arch=linux-centos7-broadwell
module load exasgd-cub/1.12.0-rc0/gcc-7.3.0-o5omfqk
# cuda@10.2.89%gcc@7.3.0~dev arch=linux-centos7-broadwell
module load exasgd-cuda/10.2.89/gcc-7.3.0-fytuxuh
# gmp@6.2.1%gcc@7.3.0 arch=linux-centos7-broadwell
module load exasgd-gmp/6.2.1/gcc-7.3.0-uimke6b
# ipopt@3.12.10%gcc@7.3.0+coinhsl~debug~metis~mumps arch=linux-centos7-broadwell
module load exasgd-ipopt/3.12.10/gcc-7.3.0-kyxesp7
# magma@2.5.4%gcc@7.3.0+cuda+fortran~ipo+shared build_type=RelWithDebInfo cuda_arch=60 arch=linux-centos7-broadwell
module load exasgd-magma/2.5.4/cuda-10.2.89/gcc-7.3.0-25cv4mc
# metis@5.1.0%gcc@7.3.0~gdb~int64~real64+shared build_type=Release patches=4991da938c1d3a1d3dea78e49bbebecba00273f98df2a656e38b83d55b281da1,b1225da886605ea558db7ac08dd8054742ea5afe5ed61ad4d0fe7a495b1270d2 arch=linux-centos7-broadwell
module load exasgd-metis/5.1.0/gcc-7.3.0-232rotu
# mpfr@4.1.0%gcc@7.3.0 arch=linux-centos7-broadwell
module load exasgd-mpfr/4.1.0/gcc-7.3.0-3rrse4o
# ncurses@6.2%gcc@7.3.0~symlinks+termlib abi=none arch=linux-centos7-broadwell
module load exasgd-ncurses/6.2/gcc-7.3.0-h2zctno
# netlib-scalapack@2.1.0%gcc@7.3.0~ipo~pic+shared build_type=Release patches=1c9ce5fee1451a08c2de3cc87f446aeda0b818ebbce4ad0d980ddf2f2a0b2dc4,f2baedde688ffe4c20943c334f580eb298e04d6f35c86b90a1f4e8cb7ae344a2 arch=linux-centos7-broadwell
module load exasgd-netlib-scalapack/2.1.0/openmpi-3.1.3/gcc-7.3.0-kcsvzxm
# openblas@0.3.5%gcc@7.3.0~bignuma~consistent_fpcsr~ilp64+locking+pic+shared patches=865703b4f405543bbd583413fdeff2226dfda908be33639276c06e5aa7ae2cae threads=none arch=linux-centos7-broadwell
module load exasgd-openblas/0.3.5/gcc-7.3.0-ug3gdtf
# openmpi@3.1.3%gcc@7.3.0~atomics~cuda~cxx~cxx_exceptions+gpfs~internal-hwloc~java~legacylaunchers~lustre~memchecker~pmi~singularity~sqlite3+static~thread_multiple+vt+wrapper-rpath fabrics=none schedulers=none arch=linux-centos7-broadwell
module load exasgd-openmpi/3.1.3/gcc-7.3.0-etzhtfg
# parmetis@4.0.3%gcc@7.3.0~gdb~int64~ipo+shared build_type=RelWithDebInfo patches=4f892531eb0a807eb1b82e683a416d3e35154a455274cf9b162fb02054d11a5b,50ed2081bc939269689789942067c58b3e522c269269a430d5d34c00edbc5870,704b84f7c7444d4372cb59cca6e1209df4ef3b033bc4ee3cf50f369bce972a9d arch=linux-centos7-broadwell
module load exasgd-parmetis/4.0.3/openmpi-3.1.3/gcc-7.3.0-4atunu3
# perl@5.26.0%gcc@7.3.0+cpanm~shared~threads patches=0eac10ed90aeb0459ad8851f88081d439a4e41978e586ec743069e8b059370ac arch=linux-centos7-broadwell
module load exasgd-perl/5.26.0/gcc-7.3.0-wog3ysl
# pkgconf@1.7.4%gcc@7.3.0 arch=linux-centos7-broadwell
module load exasgd-pkgconf/1.7.4/gcc-7.3.0-n2z4txg
# raja@0.13.0%gcc@7.3.0+cuda~examples~exercises~ipo+openmp~rocm+shared~tests amdgpu_target=none build_type=RelWithDebInfo cuda_arch=none arch=linux-centos7-broadwell
module load exasgd-raja/0.13.0/cuda-10.2.89/gcc-7.3.0-jemkyui
# strumpack@5.1.1%gcc@7.3.0~build_dev_tests~build_tests+butterflypack+c_interface~count_flops+cuda~ipo+mpi+openmp+parmetis~rocm~scotch+shared~slate~task_timers+zfp amdgpu_target=none build_type=RelWithDebInfo cuda_arch=none arch=linux-centos7-broadwell
module load exasgd-strumpack/5.1.1/cuda-10.2.89/openmpi-3.1.3/gcc-7.3.0-4fqr6cn
# suite-sparse@5.10.1%gcc@7.3.0~cuda~openmp+pic~tbb arch=linux-centos7-broadwell
module load exasgd-suite-sparse/5.10.1/gcc-7.3.0-kj6x5bx
# texinfo@6.5%gcc@7.3.0 patches=12f6edb0c6b270b8c8dba2ce17998c580db01182d871ee32b7b6e4129bd1d23a,1732115f651cff98989cb0215d8f64da5e0f7911ebf0c13b064920f088f2ffe1 arch=linux-centos7-broadwell
module load exasgd-texinfo/6.5/gcc-7.3.0-l44ipkv
# umpire@4.1.2%gcc@7.3.0+c+cuda~deviceconst~examples~fortran~ipo~numa+openmp~rocm~shared amdgpu_target=none build_type=RelWithDebInfo cuda_arch=none patches=135bbc7d2f371531f432672b115ac0a407968aabfffc5b8a941db9b493dbf81f,7d912d31cd293df005ba74cb96c6f3e32dc3d84afff49b14509714283693db08 tests=none arch=linux-centos7-broadwell
module load exasgd-umpire/4.1.2/cuda-10.2.89/gcc-7.3.0-3uphncx
# zfp@0.5.5%gcc@7.3.0~aligned~c~cuda~fasthash~fortran~ipo~openmp~profile~python+shared~strided~twoway bsws=64 build_type=RelWithDebInfo cuda_arch=none arch=linux-centos7-broadwell
module load exasgd-zfp/0.5.5/gcc-7.3.0-h3p6smb

# Load system modules
module load gcc/7.3.0
module load cuda/10.2.89
module load openmpi/3.1.3

[ -f $PWD/nvblas.conf ] && rm $PWD/nvblas.conf
cat > $PWD/nvblas.conf <<-EOD
NVBLAS_LOGFILE  nvblas.log
NVBLAS_CPU_BLAS_LIB $OPENBLAS_LIBRARY_DIR/libopenblas.so
NVBLAS_GPU_LIST ALL
NVBLAS_TILE_DIM 2048
NVBLAS_AUTOPIN_MEM_ENABLED
EOD
export NVBLAS_CONFIG_FILE=$PWD/nvblas.conf
echo "Generated $PWD/nvblas.conf"

export EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DCMAKE_CUDA_ARCHITECTURES=60 -DHIOP_USE_STRUMPACK=ON"
