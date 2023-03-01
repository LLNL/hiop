#  NOTE: The following is required when running from Gitlab CI via slurm job
source /etc/profile.d/modules.sh
module use -a /qfs/projects/exasgd/src/ci-deception/ci-modules/linux-centos7-zen2

# Load spack-built modules
# pkgconf@1.8.0%gcc@10.2.0 arch=linux-centos7-zen2
module load pkgconf-1.8.0-gcc-10.2.0-fuflwbl
# ncurses@6.3%gcc@10.2.0~symlinks+termlib abi=none arch=linux-centos7-zen2
module load ncurses-6.3-gcc-10.2.0-4wlnxto
# ca-certificates-mozilla@2022-07-19%gcc@10.2.0 arch=linux-centos7-zen2
module load ca-certificates-mozilla-2022-07-19-gcc-10.2.0-h2opehw
# berkeley-db@18.1.40%gcc@10.2.0+cxx~docs+stl patches=b231fcc arch=linux-centos7-zen2
module load berkeley-db-18.1.40-gcc-10.2.0-hltd4j3
# libiconv@1.16%gcc@10.2.0 libs=shared,static arch=linux-centos7-zen2
module load libiconv-1.16-gcc-10.2.0-gbg7l5p
# diffutils@3.8%gcc@10.2.0 arch=linux-centos7-zen2
module load diffutils-3.8-gcc-10.2.0-mjfwces
# bzip2@1.0.8%gcc@10.2.0~debug~pic+shared arch=linux-centos7-zen2
module load bzip2-1.0.8-gcc-10.2.0-bxh46iv
# readline@8.1.2%gcc@10.2.0 arch=linux-centos7-zen2
module load readline-8.1.2-gcc-10.2.0-vtya5ay
# gdbm@1.19%gcc@10.2.0 arch=linux-centos7-zen2
module load gdbm-1.19-gcc-10.2.0-efj5agg
# zlib@1.2.12%gcc@10.2.0+optimize+pic+shared patches=0d38234 arch=linux-centos7-zen2
module load zlib-1.2.12-gcc-10.2.0-gnkqokp
# perl@5.34.1%gcc@10.2.0+cpanm+shared+threads arch=linux-centos7-zen2
module load perl-5.34.1-gcc-10.2.0-xp4fpdr
# openssl@1.1.1q%gcc@10.2.0~docs~shared certs=mozilla patches=3fdcf2d arch=linux-centos7-zen2
## module load openssl-1.1.1q-gcc-10.2.0-xhxspos
# cmake@3.23.3%gcc@10.2.0~doc+ncurses+ownlibs~qt build_type=Release arch=linux-centos7-zen2
module load cmake-3.23.3-gcc-10.2.0-ggyj7bs
# blt@0.4.1%gcc@10.2.0 arch=linux-centos7-zen2
module load blt-0.4.1-gcc-10.2.0-oabae2w
# cub@1.16.0%gcc@10.2.0 arch=linux-centos7-zen2
module load cub-1.16.0-gcc-10.2.0-ovgrtom
# cuda@11.4%gcc@10.2.0~allow-unsupported-compilers~dev arch=linux-centos7-zen2
module load cuda-11.4-gcc-10.2.0-ewurpsv
# camp@0.2.3%gcc@10.2.0+cuda~ipo+openmp~rocm~tests build_type=RelWithDebInfo cuda_arch=60,70,75,80 arch=linux-centos7-zen2
module load camp-0.2.3-gcc-10.2.0-36lcy72
# openblas@0.3.20%gcc@10.2.0~bignuma~consistent_fpcsr~ilp64+locking+pic+shared patches=9f12903 symbol_suffix=none threads=none arch=linux-centos7-zen2
module load openblas-0.3.20-gcc-10.2.0-x6v3mwm
# coinhsl@2019.05.21%gcc@10.2.0+blas arch=linux-centos7-zen2
module load coinhsl-2019.05.21-gcc-10.2.0-gkzkws6
# ginkgo@1.5.0.glu_experimental%gcc@10.2.0+cuda~develtools~full_optimizations~hwloc~ipo~mpi~oneapi+openmp~rocm+shared build_system=cmake build_type=Debug cuda_arch=60,70,75,80 arch=linux-centos7-zen2
module load ginkgo-1.5.0.glu_experimental-gcc-10.2.0-3o5dw4r
# magma@2.6.2%gcc@10.2.0+cuda+fortran~ipo~rocm+shared build_type=RelWithDebInfo cuda_arch=60,70,75,80 arch=linux-centos7-zen2
module load magma-2.6.2-gcc-10.2.0-caockkq
# metis@5.1.0%gcc@10.2.0~gdb~int64~real64+shared build_type=Release patches=4991da9,b1225da arch=linux-centos7-zen2
module load metis-5.1.0-gcc-10.2.0-k4z4v6l
# openmpi@4.1.0mlx5.0%gcc@10.2.0~atomics~cuda~cxx~cxx_exceptions~gpfs~internal-hwloc~java~legacylaunchers~lustre~memchecker+romio+rsh~singularity+static+vt+wrapper-rpath fabrics=none patches=60ce20b schedulers=none arch=linux-centos7-zen2
module load openmpi-4.1.0mlx5.0-gcc-10.2.0-ytj7jxb
# raja@0.14.0%gcc@10.2.0+cuda~examples~exercises~ipo+openmp~rocm+shared~tests build_type=RelWithDebInfo cuda_arch=60,70,75,80arch=linux-centos7-zen2
module load raja-0.14.0-gcc-10.2.0-tyzamiy
# libsigsegv@2.13%gcc@10.2.0 arch=linux-centos7-zen2
module load libsigsegv-2.13-gcc-10.2.0-aj5goyi
# m4@1.4.19%gcc@10.2.0+sigsegv patches=9dc5fbd,bfdffa7 arch=linux-centos7-zen2
module load m4-1.4.19-gcc-10.2.0-k5kkyx6
# autoconf@2.69%gcc@10.2.0 patches=35c4492,7793209,a49dd5b arch=linux-centos7-zen2
module load autoconf-2.69-gcc-10.2.0-jnh4mbw
# automake@1.16.5%gcc@10.2.0 arch=linux-centos7-zen2
module load automake-1.16.5-gcc-10.2.0-pgpzgqq
# libtool@2.4.7%gcc@10.2.0 arch=linux-centos7-zen2
module load libtool-2.4.7-gcc-10.2.0-mzc2mvw
# gmp@6.2.1%gcc@10.2.0 libs=shared,static arch=linux-centos7-zen2
module load gmp-6.2.1-gcc-10.2.0-tpo7i4x
# autoconf-archive@2022.02.11%gcc@10.2.0 patches=139214f arch=linux-centos7-zen2
module load autoconf-archive-2022.02.11-gcc-10.2.0-tirhdzr
# texinfo@6.5%gcc@10.2.0 patches=12f6edb,1732115 arch=linux-centos7-zen2
module load texinfo-6.5-gcc-10.2.0-mcrbwnj
# mpfr@4.1.0%gcc@10.2.0 libs=shared,static arch=linux-centos7-zen2
module load mpfr-4.1.0-gcc-10.2.0-3yutkz3
# suite-sparse@5.10.1%gcc@10.2.0~cuda~graphblas~openmp+pic~tbb arch=linux-centos7-zen2
module load suite-sparse-5.10.1-gcc-10.2.0-add65sb
# umpire@6.0.0%gcc@10.2.0+c+cuda~device_alloc~deviceconst~examples~fortran~ipo~numa~openmp~rocm~shared build_type=RelWithDebInfo cuda_arch=60,70,75,80 tests=none arch=linux-centos7-zen2
module load umpire-6.0.0-gcc-10.2.0-lrjkuun
# hiop@develop%gcc@10.2.0+cuda~cusolver+deepchecking~full_optimizations+ginkgo~ipo~jsrun+kron+mpi+raja~rocm~shared+sparse build_type=RelWithDebInfo cuda_arch=60,70,75,80 arch=linux-centos7-zen2
## module load hiop-develop-gcc-10.2.0-bgzxttu

# Load system modules
module load gcc/10.2.0
module load cuda/11.4

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

export CMAKE_CACHE_SCRIPT=gcc-cuda.cmake
