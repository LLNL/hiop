#  NOTE: The following is required when running from Gitlab CI via slurm job
source /etc/profile.d/modules.sh
module use -a /qfs/projects/exasgd/src/spack/share/spack/modules/linux-centos7-broadwell/
module use -a /qfs/projects/exasgd/src/spack/share/spack/modules/linux-centos7-x86_64/
module use -a /qfs/projects/exasgd/src/spack/share/spack/modules/linux-centos7-zen2/

# Load spack-built modules
# autoconf@2.69%gcc@9.1.0 patches=35c4492,7793209,a49dd5b arch=linux-centos7-x86_64
module load exasgd-autoconf/2.69/gcc-9.1.0-xaal7xw
# autoconf-archive@2022.02.11%gcc@9.1.0 arch=linux-centos7-x86_64
module load exasgd-autoconf-archive/2022.02.11/gcc-9.1.0-uzu4cl4
# automake@1.16.5%gcc@9.1.0 arch=linux-centos7-x86_64
module load exasgd-automake/1.16.5/gcc-9.1.0-je647tg
# blt@0.4.1%gcc@9.1.0 arch=linux-centos7-x86_64
module load exasgd-blt/0.4.1/gcc-9.1.0-atd2nxb
# camp@0.2.2%gcc@9.1.0+cuda~ipo~rocm~tests build_type=RelWithDebInfo cuda_arch=60 arch=linux-centos7-x86_64
module load exasgd-camp/0.2.2/cuda-11.4.0/gcc-9.1.0-g277fwl
# cmake@3.23.1%gcc@9.1.0~doc+ncurses+ownlibs~qt build_type=Release arch=linux-centos7-x86_64
module load exasgd-cmake/3.23.1/gcc-9.1.0-tfqiuui
# coinhsl@2015.06.23%gcc@9.1.0+blas arch=linux-centos7-x86_64
module load exasgd-coinhsl/2015.06.23/gcc-9.1.0-frpha6o
# cub@1.16.0%gcc@9.1.0 arch=linux-centos7-x86_64
module load exasgd-cub/1.16.0/gcc-9.1.0-mzolmhz
# cuda@11.4.0%gcc@9.1.0~allow-unsupported-compilers~dev arch=linux-centos7-x86_64
module load exasgd-cuda/11.4.0/gcc-9.1.0-vvyc6ru
# ginkgo@glu%gcc@9.1.0+cuda~develtools~full_optimizations~hwloc~ipo~oneapi+openmp~rocm+shared build_type=Release cuda_arch=60 arch=linux-centos7-x86_64
module load exasgd-ginkgo/glu/cuda-11.4.0/gcc-9.1.0-drwioxw
# gmp@6.2.1%gcc@9.1.0 libs=shared,static arch=linux-centos7-x86_64
module load exasgd-gmp/6.2.1/gcc-9.1.0-xynizoz
# libsigsegv@2.13%gcc@9.1.0 arch=linux-centos7-x86_64
module load exasgd-libsigsegv/2.13/gcc-9.1.0-3hpa5yq
# libtool@2.4.7%gcc@9.1.0 arch=linux-centos7-x86_64
module load exasgd-libtool/2.4.7/gcc-9.1.0-vgq3gy7
# m4@1.4.19%gcc@9.1.0+sigsegv patches=9dc5fbd,bfdffa7 arch=linux-centos7-x86_64
module load exasgd-m4/1.4.19/gcc-9.1.0-edbcgvf
# magma@2.6.2%gcc@9.1.0+cuda+fortran~ipo~rocm+shared build_type=RelWithDebInfo cuda_arch=60 arch=linux-centos7-x86_64
module load exasgd-magma/2.6.2/cuda-11.4.0/gcc-9.1.0-yhyz4ex
# metis@5.1.0%gcc@9.1.0~gdb~int64~real64+shared build_type=Release patches=4991da9,b1225da arch=linux-centos7-x86_64
module load exasgd-metis/5.1.0/gcc-9.1.0-pvcxfl2
# mpfr@4.1.0%gcc@9.1.0 libs=shared,static arch=linux-centos7-x86_64
module load exasgd-mpfr/4.1.0/gcc-9.1.0-4ohy64q
# ncurses@6.2%gcc@9.1.0~symlinks+termlib abi=none arch=linux-centos7-x86_64
module load exasgd-ncurses/6.2/gcc-9.1.0-ht75mcf
# openblas@0.3.20%gcc@9.1.0~bignuma~consistent_fpcsr~ilp64+locking+pic+shared symbol_suffix=none threads=none arch=linux-centos7-x86_64
module load exasgd-openblas/0.3.20/gcc-9.1.0-4cgoaop
# openmpi@4.1.0%gcc@9.1.0~atomics~cuda~cxx~cxx_exceptions~gpfs~internal-hwloc~java~legacylaunchers~lustre~memchecker~pmi+pmix+romio+rsh~singularity+static+vt+wrapper-rpath fabrics=none patches=60ce20b schedulers=none arch=linux-centos7-x86_64
module load exasgd-openmpi/4.1.0/gcc-9.1.0-oemcjd2
# openssl@1.0.2k-fips%gcc@9.1.0~docs~shared certs=system arch=linux-centos7-x86_64
module load exasgd-openssl/1.0.2k-fips/gcc-9.1.0-j35p7cg
# perl@5.26.0%gcc@9.1.0+cpanm~shared~threads patches=0eac10e,8cf4302 arch=linux-centos7-x86_64
module load exasgd-perl/5.26.0/gcc-9.1.0-hkdmvuv
# pkgconf@1.8.0%gcc@9.1.0 arch=linux-centos7-x86_64
module load exasgd-pkgconf/1.8.0/gcc-9.1.0-gf4npoa
# raja@0.14.0%gcc@9.1.0+cuda+examples+exercises~ipo+openmp~rocm+shared~tests build_type=RelWithDebInfo cuda_arch=60 arch=linux-centos7-x86_64
module load exasgd-raja/0.14.0/cuda-11.4.0/gcc-9.1.0-ocbi5z6
# suite-sparse@5.10.1%gcc@9.1.0~cuda~graphblas~openmp+pic~tbb arch=linux-centos7-x86_64
module load exasgd-suite-sparse/5.10.1/gcc-9.1.0-iht7c6l
# texinfo@6.5%gcc@9.1.0 patches=12f6edb,1732115 arch=linux-centos7-x86_64
module load exasgd-texinfo/6.5/gcc-9.1.0-74ugzwk
# umpire@6.0.0%gcc@9.1.0+c+cuda~deviceconst+examples~fortran~ipo~numa~openmp~rocm~shared build_type=RelWithDebInfo cuda_arch=60 tests=none arch=linux-centos7-x86_64
module load exasgd-umpire/6.0.0/cuda-11.4.0/gcc-9.1.0-prpuw5c

# Load system modules
module load gcc/9.1.0

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

export EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DCMAKE_CUDA_ARCHITECTURES=60"
export CMAKE_CACHE_SCRIPT=gcc-cuda.cmake
