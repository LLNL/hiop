module use -a /usr/workspace/hiop/software/spack_modules_20240124/linux-rhel7-power9le

module purge

module load gcc/8.3.1
module load cmake/3.20.2 
module load cuda/11.7.0 
module load python/3.8.2

# cmake@=3.20.2%gcc@=8.3.1~doc+ncurses+ownlibs build_system=generic build_type=Release arch=linux-rhel7-power9le
#module load cmake/3.20.2-gcc-8.3.1-module-lpmoh3j
# gcc-runtime@=8.3.1%gcc@=8.3.1 build_system=generic arch=linux-rhel7-power9le
module load gcc-runtime/8.3.1-gcc-8.3.1-module-6vlkybe
# blt@=0.4.1%gcc@=8.3.1 build_system=generic arch=linux-rhel7-power9le
module load blt/0.4.1-gcc-8.3.1-module-4qz27hh
# cub@=2.1.0%gcc@=8.3.1 build_system=generic arch=linux-rhel7-power9le
module load cub/2.1.0-gcc-8.3.1-module-y3txey4
# cuda@=11.7.0%gcc@=8.3.1~allow-unsupported-compilers~dev build_system=generic arch=linux-rhel7-power9le
#module load cuda/11.7.0-gcc-8.3.1-module-at7dzwx
# gmake@=4.4.1%gcc@=8.3.1~guile build_system=generic arch=linux-rhel7-power9le
module load gmake/4.4.1-gcc-8.3.1-module-ydj27bx
# camp@=0.2.3%gcc@=8.3.1+cuda~ipo+openmp~rocm~tests build_system=cmake build_type=Release cuda_arch=70 generator=make arch=linux-rhel7-power9le
module load camp/0.2.3-gcc-8.3.1-module-7emdy7o
# gnuconfig@=2022-09-17%gcc@=8.3.1 build_system=generic arch=linux-rhel7-power9le
module load gnuconfig/2022-09-17-gcc-8.3.1-module-tebfisj
# berkeley-db@=18.1.40%gcc@=8.3.1+cxx~docs+stl build_system=autotools patches=26090f4,b231fcc arch=linux-rhel7-power9le
module load berkeley-db/18.1.40-gcc-8.3.1-module-42f44ve
# libiconv@=1.17%gcc@=8.3.1 build_system=autotools libs=shared,static arch=linux-rhel7-power9le
module load libiconv/1.17-gcc-8.3.1-module-ytdlppt
# diffutils@=3.9%gcc@=8.3.1 build_system=autotools arch=linux-rhel7-power9le
module load diffutils/3.9-gcc-8.3.1-module-roteu43
# bzip2@=1.0.8%gcc@=8.3.1~debug~pic+shared build_system=generic arch=linux-rhel7-power9le
module load bzip2/1.0.8-gcc-8.3.1-module-r3iw45a
# pkgconf@=1.9.5%gcc@=8.3.1 build_system=autotools arch=linux-rhel7-power9le
module load pkgconf/1.9.5-gcc-8.3.1-module-lqdmmz3
# ncurses@=6.4%gcc@=8.3.1~symlinks+termlib abi=none build_system=autotools arch=linux-rhel7-power9le
module load ncurses/6.4-gcc-8.3.1-module-r4jf2fc
# readline@=8.2%gcc@=8.3.1 build_system=autotools patches=bbf97f1 arch=linux-rhel7-power9le
module load readline/8.2-gcc-8.3.1-module-squ6psq
# gdbm@=1.23%gcc@=8.3.1 build_system=autotools arch=linux-rhel7-power9le
module load gdbm/1.23-gcc-8.3.1-module-bkpen7q
# zlib-ng@=2.1.5%gcc@=8.3.1+compat+opt build_system=autotools arch=linux-rhel7-power9le
module load zlib-ng/2.1.5-gcc-8.3.1-module-5mjtwml
# perl@=5.38.0%gcc@=8.3.1+cpanm+opcode+open+shared+threads build_system=generic patches=714e4d1 arch=linux-rhel7-power9le
module load perl/5.38.0-gcc-8.3.1-module-aurq6wi
# openblas@=0.3.24%gcc@=8.3.1~bignuma~consistent_fpcsr+fortran~ilp64+locking+pic+shared build_system=makefile symbol_suffix=none threads=none arch=linux-rhel7-power9le
module load openblas/0.3.24-gcc-8.3.1-module-lpmxy3n
# coinhsl@=2015.06.23%gcc@=8.3.1+blas build_system=autotools arch=linux-rhel7-power9le
module load coinhsl/2015.06.23-gcc-8.3.1-module-7mkgb2d
# ginkgo@=1.5.0.glu_experimental%gcc@=8.3.1+cuda~develtools~full_optimizations~hwloc~ipo~mpi+openmp~rocm+shared~sycl build_system=cmake build_type=Release cuda_arch=70 generator=make arch=linux-rhel7-power9le
module load ginkgo/1.5.0.glu_experimental-gcc-8.3.1-module-ql5jego
# magma@=2.6.2%gcc@=8.3.1+cuda+fortran~ipo~rocm+shared build_system=cmake build_type=Release cuda_arch=70 generator=make arch=linux-rhel7-power9le
module load magma/2.6.2-gcc-8.3.1-module-hok7ges
# metis@=5.1.0%gcc@=8.3.1~gdb~int64~ipo~real64+shared build_system=cmake build_type=Release generator=make patches=4991da9,93a7903,b1225da arch=linux-rhel7-power9le
module load metis/5.1.0-gcc-8.3.1-module-alz2jts
# raja@=0.14.0%gcc@=8.3.1+cuda~examples~exercises~ipo+openmp~plugins~rocm+shared~tests build_system=cmake build_type=Release cuda_arch=70 generator=make arch=linux-rhel7-power9le
module load raja/0.14.0-gcc-8.3.1-module-rs3jofo
# spectrum-mpi@=rolling-release%gcc@=8.3.1 build_system=bundle arch=linux-rhel7-power9le
module load spectrum-mpi/rolling-release-gcc-8.3.1-module-62ppinp
# libsigsegv@=2.14%gcc@=8.3.1 build_system=autotools arch=linux-rhel7-power9le
module load libsigsegv/2.14-gcc-8.3.1-module-edsrfng
# m4@=1.4.19%gcc@=8.3.1+sigsegv build_system=autotools patches=9dc5fbd,bfdffa7 arch=linux-rhel7-power9le
module load m4/1.4.19-gcc-8.3.1-module-nypsjmv
# autoconf@=2.72%gcc@=8.3.1 build_system=autotools arch=linux-rhel7-power9le
module load autoconf/2.72-gcc-8.3.1-module-nzuoeaa
# automake@=1.16.5%gcc@=8.3.1 build_system=autotools arch=linux-rhel7-power9le
module load automake/1.16.5-gcc-8.3.1-module-zodgmoo
# findutils@=4.9.0%gcc@=8.3.1 build_system=autotools patches=440b954 arch=linux-rhel7-power9le
module load findutils/4.9.0-gcc-8.3.1-module-5gwitka
# libtool@=2.4.7%gcc@=8.3.1 build_system=autotools arch=linux-rhel7-power9le
module load libtool/2.4.7-gcc-8.3.1-module-ulk4aiy
# gmp@=6.2.1%gcc@=8.3.1+cxx build_system=autotools libs=shared,static patches=69ad2e2 arch=linux-rhel7-power9le
module load gmp/6.2.1-gcc-8.3.1-module-ixxhi6l
# autoconf-archive@=2023.02.20%gcc@=8.3.1 build_system=autotools arch=linux-rhel7-power9le
module load autoconf-archive/2023.02.20-gcc-8.3.1-module-zkvzpdn
# xz@=5.4.1%gcc@=8.3.1~pic build_system=autotools libs=shared,static arch=linux-rhel7-power9le
module load xz/5.4.1-gcc-8.3.1-module-qqsbjls
# libxml2@=2.10.3%gcc@=8.3.1+pic~python+shared build_system=autotools arch=linux-rhel7-power9le
module load libxml2/2.10.3-gcc-8.3.1-module-rszbsuw
# pigz@=2.8%gcc@=8.3.1 build_system=makefile arch=linux-rhel7-power9le
module load pigz/2.8-gcc-8.3.1-module-s3c6uns
# zstd@=1.5.5%gcc@=8.3.1+programs build_system=makefile compression=none libs=shared,static arch=linux-rhel7-power9le
module load zstd/1.5.5-gcc-8.3.1-module-eafslmc
# tar@=1.34%gcc@=8.3.1 build_system=autotools zip=pigz arch=linux-rhel7-power9le
module load tar/1.34-gcc-8.3.1-module-rm6jve4
# gettext@=0.22.4%gcc@=8.3.1+bzip2+curses+git~libunistring+libxml2+pic+shared+tar+xz build_system=autotools arch=linux-rhel7-power9le
module load gettext/0.22.4-gcc-8.3.1-module-w7gkgbj
# texinfo@=7.0.3%gcc@=8.3.1 build_system=autotools arch=linux-rhel7-power9le
module load texinfo/7.0.3-gcc-8.3.1-module-eoofajf
# mpfr@=4.2.0%gcc@=8.3.1 build_system=autotools libs=shared,static arch=linux-rhel7-power9le
module load mpfr/4.2.0-gcc-8.3.1-module-pkiqrkc
# suite-sparse@=5.13.0%gcc@=8.3.1~cuda~graphblas~openmp+pic build_system=generic arch=linux-rhel7-power9le
module load suite-sparse/5.13.0-gcc-8.3.1-module-yoyxdkr
# umpire@=6.0.0%gcc@=8.3.1~c+cuda~device_alloc~deviceconst~examples~fortran~ipo~numa~openmp~rocm~shared build_system=cmake build_type=Release cuda_arch=70 generator=make tests=none arch=linux-rhel7-power9le
module load umpire/6.0.0-gcc-8.3.1-module-wdgpdbs

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

export EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DHIOP_USE_GINKGO=OFF -DHIOP_TEST_WITH_BSUB=ON -DCMAKE_CUDA_ARCHITECTURES=70"
export EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DHIOP_CTEST_LAUNCH_COMMAND:STRING='jsrun -n 2 -a 1 -c 1 -g 1'"
export CMAKE_CACHE_SCRIPT=gcc-cuda.cmake

