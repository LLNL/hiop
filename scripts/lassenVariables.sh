module use -a /usr/workspace/hiop/software/spack_modules_202408/linux-rhel7-power9le

module purge

module load gcc/8.3.1
module load cmake/3.20.2 
module load python/3.8.2


# cmake@=3.20.2%gcc@=8.3.1~doc+ncurses+ownlibs build_system=generic build_type=Release arch=linux-rhel7-power9le
module load cmake/3.20.2-linux-rhel7-power9le-sgbbk2e
# glibc@=2.17%gcc@=8.3.1 build_system=autotools patches=be65fec,e179c43 arch=linux-rhel7-power9le
module load glibc/2.17-linux-rhel7-power9le-ltqhcqm
# gcc-runtime@=8.3.1%gcc@=8.3.1 build_system=generic arch=linux-rhel7-power9le
module load gcc-runtime/8.3.1-linux-rhel7-power9le-hvpgryd
# blt@=0.4.1%gcc@=8.3.1 build_system=generic arch=linux-rhel7-power9le
module load blt/0.4.1-linux-rhel7-power9le-yq3ifkk
# cub@=1.16.0%gcc@=8.3.1 build_system=generic arch=linux-rhel7-power9le
module load cub/1.16.0-linux-rhel7-power9le-mbi6tgn
# gmake@=4.4.1%gcc@=8.3.1~guile build_system=generic arch=linux-rhel7-power9le
module load gmake/4.4.1-linux-rhel7-power9le-76tj7qq
# gnuconfig@=2022-09-17%gcc@=8.3.1 build_system=generic arch=linux-rhel7-power9le
module load gnuconfig/2022-09-17-linux-rhel7-power9le-33h26h4
# libiconv@=1.17%gcc@=8.3.1 build_system=autotools libs=shared,static arch=linux-rhel7-power9le
module load libiconv/1.17-linux-rhel7-power9le-vomriir
# pkgconf@=2.2.0%gcc@=8.3.1 build_system=autotools arch=linux-rhel7-power9le
module load pkgconf/2.2.0-linux-rhel7-power9le-w5eyts5
# xz@=5.4.6%gcc@=8.3.1~pic build_system=autotools libs=shared,static arch=linux-rhel7-power9le
module load xz/5.4.6-linux-rhel7-power9le-wy2yvqt
# zlib-ng@=2.2.1%gcc@=8.3.1+compat+new_strategies+opt+pic+shared build_system=autotools arch=linux-rhel7-power9le
module load zlib-ng/2.2.1-linux-rhel7-power9le-zfirv2c
# libxml2@=2.10.3%gcc@=8.3.1+pic~python+shared build_system=autotools arch=linux-rhel7-power9le
module load libxml2/2.10.3-linux-rhel7-power9le-a2cuzya
# cuda@=11.4.2%gcc@=8.3.1~allow-unsupported-compilers~dev build_system=generic arch=linux-rhel7-power9le
module load cuda/11.4.2-linux-rhel7-power9le-rpeosz6
# camp@=0.2.3%gcc@=8.3.1+cuda~ipo+openmp~rocm~tests build_system=cmake build_type=Release cuda_arch=70 generator=make patches=cb9e25b arch=linux-rhel7-power9le
module load camp/0.2.3-linux-rhel7-power9le-seoxg6w
# berkeley-db@=18.1.40%gcc@=8.3.1+cxx~docs+stl build_system=autotools patches=26090f4,b231fcc arch=linux-rhel7-power9le
module load berkeley-db/18.1.40-linux-rhel7-power9le-xeq7mjg
# diffutils@=3.10%gcc@=8.3.1 build_system=autotools arch=linux-rhel7-power9le
module load diffutils/3.10-linux-rhel7-power9le-gg26vck
# bzip2@=1.0.8%gcc@=8.3.1~debug~pic+shared build_system=generic arch=linux-rhel7-power9le
module load bzip2/1.0.8-linux-rhel7-power9le-kzyaip2
# ncurses@=6.5%gcc@=8.3.1~symlinks+termlib abi=none build_system=autotools patches=7a351bc arch=linux-rhel7-power9le
module load ncurses/6.5-linux-rhel7-power9le-h3en26s
# readline@=8.2%gcc@=8.3.1 build_system=autotools patches=bbf97f1 arch=linux-rhel7-power9le
module load readline/8.2-linux-rhel7-power9le-dhcjafy
# gdbm@=1.23%gcc@=8.3.1 build_system=autotools arch=linux-rhel7-power9le
module load gdbm/1.23-linux-rhel7-power9le-eizs5lo
# perl@=5.40.0%gcc@=8.3.1+cpanm+opcode+open+shared+threads build_system=generic arch=linux-rhel7-power9le
module load perl/5.40.0-linux-rhel7-power9le-cmrz6t7
# openblas@=0.3.24%gcc@=8.3.1~bignuma~consistent_fpcsr+dynamic_dispatch+fortran~ilp64+locking+pic+shared build_system=makefile symbol_suffix=none threads=none arch=linux-rhel7-power9le
module load openblas/0.3.24-linux-rhel7-power9le-6ek5q6o
# coinhsl@=2015.06.23%gcc@=8.3.1+blas build_system=autotools arch=linux-rhel7-power9le
module load coinhsl/2015.06.23-linux-rhel7-power9le-7usp2us
# ginkgo@=1.5.0.glu_experimental%gcc@=8.3.1+cuda~develtools~full_optimizations~hwloc~ipo~mpi+openmp~rocm+shared~sycl build_system=cmake build_type=Release cuda_arch=70 generator=make arch=linux-rhel7-power9le
module load ginkgo/1.5.0.glu_experimental-linux-rhel7-power9le-ibgwveo
# magma@=2.6.2%gcc@=8.3.1+cuda+fortran~ipo~rocm+shared build_system=cmake build_type=Release cuda_arch=70 generator=make arch=linux-rhel7-power9le
module load magma/2.6.2-linux-rhel7-power9le-qdoblh3
# metis@=5.1.0%gcc@=8.3.1~gdb~int64~ipo~real64+shared build_system=cmake build_type=Release generator=make patches=4991da9,93a7903,b1225da arch=linux-rhel7-power9le
module load metis/5.1.0-linux-rhel7-power9le-pq37727
# raja@=0.14.0%gcc@=8.3.1+cuda~desul~examples~exercises~ipo~omptask+openmp~plugins~rocm~run-all-tests+shared~tests~vectorization build_system=cmake build_type=Release cuda_arch=70 generator=make arch=linux-rhel7-power9le
module load raja/0.14.0-linux-rhel7-power9le-i3do7mn
# spectrum-mpi@=rolling-release%gcc@=8.3.1 build_system=bundle arch=linux-rhel7-power9le
module load spectrum-mpi/rolling-release-linux-rhel7-power9le-cycs4kt
# libsigsegv@=2.14%gcc@=8.3.1 build_system=autotools arch=linux-rhel7-power9le
module load libsigsegv/2.14-linux-rhel7-power9le-fl37xzk
# m4@=1.4.19%gcc@=8.3.1+sigsegv build_system=autotools patches=9dc5fbd,bfdffa7 arch=linux-rhel7-power9le
module load m4/1.4.19-linux-rhel7-power9le-gwetdjs
# autoconf@=2.72%gcc@=8.3.1 build_system=autotools arch=linux-rhel7-power9le
module load autoconf/2.72-linux-rhel7-power9le-nr3otal
# automake@=1.16.5%gcc@=8.3.1 build_system=autotools arch=linux-rhel7-power9le
module load automake/1.16.5-linux-rhel7-power9le-4tpk52n
# findutils@=4.9.0%gcc@=8.3.1 build_system=autotools patches=440b954 arch=linux-rhel7-power9le
module load findutils/4.9.0-linux-rhel7-power9le-7lhqpqk
# libtool@=2.4.7%gcc@=8.3.1 build_system=autotools arch=linux-rhel7-power9le
module load libtool/2.4.7-linux-rhel7-power9le-fo55ddx
# gmp@=6.3.0%gcc@=8.3.1+cxx build_system=autotools libs=shared,static arch=linux-rhel7-power9le
module load gmp/6.3.0-linux-rhel7-power9le-wtffv4t
# autoconf-archive@=2023.02.20%gcc@=8.3.1 build_system=autotools arch=linux-rhel7-power9le
module load autoconf-archive/2023.02.20-linux-rhel7-power9le-nlgst5g
# pigz@=2.8%gcc@=8.3.1 build_system=makefile arch=linux-rhel7-power9le
module load pigz/2.8-linux-rhel7-power9le-du7lszg
# zstd@=1.5.6%gcc@=8.3.1+programs build_system=makefile compression=none libs=shared,static arch=linux-rhel7-power9le
module load zstd/1.5.6-linux-rhel7-power9le-rlza3tv
# tar@=1.34%gcc@=8.3.1 build_system=autotools zip=pigz arch=linux-rhel7-power9le
module load tar/1.34-linux-rhel7-power9le-66m3wvh
# gettext@=0.22.5%gcc@=8.3.1+bzip2+curses+git~libunistring+libxml2+pic+shared+tar+xz build_system=autotools arch=linux-rhel7-power9le
module load gettext/0.22.5-linux-rhel7-power9le-je7e7cy
# texinfo@=7.1%gcc@=8.3.1 build_system=autotools arch=linux-rhel7-power9le
module load texinfo/7.1-linux-rhel7-power9le-oss2b3r
# mpfr@=4.2.1%gcc@=8.3.1 build_system=autotools libs=shared,static arch=linux-rhel7-power9le
module load mpfr/4.2.1-linux-rhel7-power9le-tqg7cbt
# suite-sparse@=5.13.0%gcc@=8.3.1~cuda~graphblas~openmp+pic build_system=generic arch=linux-rhel7-power9le
module load suite-sparse/5.13.0-linux-rhel7-power9le-nhqdwpc
# umpire@=6.0.0%gcc@=8.3.1~asan~backtrace~c+cuda~dev_benchmarks~device_alloc~deviceconst~examples~fortran~ipc_shmem~ipo~mpi~numa~openmp~openmp_target~rocm~sanitizer_tests~shared~sqlite_experimental~tools~werror build_system=cmake build_type=Release cuda_arch=70 generator=make tests=none arch=linux-rhel7-power9le
module load umpire/6.0.0-linux-rhel7-power9le-qndtsb2

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

