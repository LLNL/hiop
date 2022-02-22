#  NOTE: The following is required when running from Gitlab CI via slurm job
source /etc/profile.d/modules.sh
module use -a /qfs/projects/exasgd/src/spack/share/spack/modules/linux-centos7-broadwell/
module use -a /qfs/projects/exasgd/src/spack/share/spack/modules/linux-centos7-x86_64/
module use -a /qfs/projects/exasgd/src/spack/share/spack/modules/linux-centos7-zen2/

# Load spack-built modules
# autoconf@2.69%gcc@9.1.0 patches=35c4492,7793209,a49dd5b arch=linux-centos7-zen2
module load autoconf-2.69-gcc-9.1.0-iu4rxsv
# autoconf-archive@2019.01.06%gcc@9.1.0 arch=linux-centos7-zen2
module load autoconf-archive-2019.01.06-gcc-9.1.0-5iu7qn2
# automake@1.16.5%gcc@9.1.0 arch=linux-centos7-zen2
module load automake-1.16.5-gcc-9.1.0-jlsc4m7
# blt@0.4.1%gcc@9.1.0 arch=linux-centos7-zen2
module load blt-0.4.1-gcc-9.1.0-46rfisg
# camp@0.2.2%gcc@9.1.0+cuda~ipo~rocm~tests build_type=RelWithDebInfo cuda_arch=60 arch=linux-centos7-zen2
module load camp-0.2.2-gcc-9.1.0-bigamlu
# cmake@3.21.4%gcc@9.1.0~doc+ncurses+openssl+ownlibs~qt build_type=Release arch=linux-centos7-zen2
module load cmake-3.21.4-gcc-9.1.0-apvb74z
# coinhsl@2015.06.23%gcc@9.1.0+blas arch=linux-centos7-zen2
module load coinhsl-2015.06.23-gcc-9.1.0-3a5axd6
# cub@1.12.0-rc0%gcc@9.1.0 arch=linux-centos7-zen2
module load cub-1.12.0-rc0-gcc-9.1.0-uoardkf
# cuda@11.4.0%gcc@9.1.0~allow-unsupported-compilers~dev arch=linux-centos7-zen2
module load cuda-11.4.0-gcc-9.1.0-nxnv3xb
# gmp@6.2.1%gcc@9.1.0 arch=linux-centos7-zen2
module load gmp-6.2.1-gcc-9.1.0-djrhu5p
# libsigsegv@2.13%gcc@9.1.0 arch=linux-centos7-zen2
module load libsigsegv-2.13-gcc-9.1.0-tmbulqw
# libtool@2.4.6%gcc@9.1.0 arch=linux-centos7-zen2
module load libtool-2.4.6-gcc-9.1.0-d3fcfrm
# m4@1.4.19%gcc@9.1.0+sigsegv patches=9dc5fbd,bfdffa7 arch=linux-centos7-zen2
module load m4-1.4.19-gcc-9.1.0-7lc4ecb
# magma@master%gcc@9.1.0+cuda+fortran~ipo~rocm+shared build_type=RelWithDebInfo cuda_arch=60 arch=linux-centos7-zen2
module load magma-master-gcc-9.1.0-wbaf5gc
# metis@5.1.0%gcc@9.1.0~gdb~int64~real64+shared build_type=Release patches=4991da9,b1225da arch=linux-centos7-zen2
module load metis-5.1.0-gcc-9.1.0-qimvzjz
# mpfr@4.1.0%gcc@9.1.0 arch=linux-centos7-zen2
module load mpfr-4.1.0-gcc-9.1.0-zpdg5z5
# openblas@0.3.19%gcc@9.1.0~bignuma~consistent_fpcsr~ilp64+locking+pic+shared symbol_suffix=none threads=none arch=linux-centos7-zen2
module load openblas-0.3.19-gcc-9.1.0-huvnbt3
# openmpi@4.1.0%gcc@9.1.0~atomics~cuda~cxx~cxx_exceptions+gpfs~internal-hwloc~java~legacylaunchers~lustre~memchecker~pmi~pmix+romio~singularity~sqlite3+static~thread_multiple+vt+wrapper-rpath fabrics=none patches=60ce20b schedulers=none arch=linux-centos7-zen2
module load openmpi-4.1.0-gcc-9.1.0-lpezrwd
# perl@5.26.0%gcc@9.1.0+cpanm+shared+threads patches=0eac10e,8cf4302 arch=linux-centos7-zen2
module load perl-5.26.0-gcc-9.1.0-mdtxew7
# raja@0.14.0%gcc@9.1.0+cuda+examples+exercises~ipo+openmp~rocm+shared~tests build_type=RelWithDebInfo cuda_arch=60 arch=linux-centos7-zen2
module load raja-0.14.0-gcc-9.1.0-t6fitor
# suite-sparse@5.10.1%gcc@9.1.0~cuda~graphblas~openmp+pic~tbb arch=linux-centos7-zen2
module load suite-sparse-5.10.1-gcc-9.1.0-k5vuvd4
# texinfo@6.5%gcc@9.1.0 patches=12f6edb,1732115 arch=linux-centos7-zen2
module load texinfo-6.5-gcc-9.1.0-rpk3pcg
# umpire@6.0.0%gcc@9.1.0+c+cuda~deviceconst+examples~fortran~ipo~numa~openmp~rocm~shared build_type=RelWithDebInfo cuda_arch=60 tests=none arch=linux-centos7-zen2
module load umpire-6.0.0-gcc-9.1.0-e2a7iik

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
