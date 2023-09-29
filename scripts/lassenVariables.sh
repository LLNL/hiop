module use -a /usr/workspace/hiop/software/spack_modules_20230927/linux-rhel7-power9le

module purge

module load gcc/8.3.1
module load cmake/3.20.2 
module load cuda/11.7.0 
module load python/3.8.2

# cmake@=3.20.2%gcc@=8.3.1~doc+ncurses+ownlibs build_system=generic build_type=Release arch=linux-rhel7-power9le
module load cmake/3.20.2-gcc-8.3.1-lpmoh3j
# blt@=0.4.1%gcc@=8.3.1 build_system=generic arch=linux-rhel7-power9le
module load blt/0.4.1-gcc-8.3.1-yukzfby
# cub@=2.1.0%gcc@=8.3.1 build_system=generic arch=linux-rhel7-power9le
module load cub/2.1.0-gcc-8.3.1-s7pvqou
# cuda@=11.7.0%gcc@=8.3.1~allow-unsupported-compilers~dev build_system=generic arch=linux-rhel7-power9le
module load cuda/11.7.0-gcc-8.3.1-x3xhhxp
# gnuconfig@=2022-09-17%gcc@=8.3.1 build_system=generic arch=linux-rhel7-power9le
module load gnuconfig/2022-09-17-gcc-8.3.1-ubaow5e
# gmake@=4.4.1%gcc@=8.3.1~guile build_system=autotools arch=linux-rhel7-power9le
module load gmake/4.4.1-gcc-8.3.1-b265ipg
# camp@=0.2.3%gcc@=8.3.1+cuda~ipo+openmp~rocm~tests build_system=cmake build_type=Release cuda_arch=70 generator=make arch=linux-rhel7-power9le
module load camp/0.2.3-gcc-8.3.1-2x7ny3c
# berkeley-db@=18.1.40%gcc@=8.3.1+cxx~docs+stl build_system=autotools patches=26090f4,b231fcc arch=linux-rhel7-power9le
module load berkeley-db/18.1.40-gcc-8.3.1-xpt3q2c
# libiconv@=1.17%gcc@=8.3.1 build_system=autotools libs=shared,static arch=linux-rhel7-power9le
module load libiconv/1.17-gcc-8.3.1-i4bdklt
# diffutils@=3.9%gcc@=8.3.1 build_system=autotools arch=linux-rhel7-power9le
module load diffutils/3.9-gcc-8.3.1-qtfterp
# bzip2@=1.0.8%gcc@=8.3.1~debug~pic+shared build_system=generic arch=linux-rhel7-power9le
module load bzip2/1.0.8-gcc-8.3.1-73nkzwe
# pkgconf@=1.9.5%gcc@=8.3.1 build_system=autotools arch=linux-rhel7-power9le
module load pkgconf/1.9.5-gcc-8.3.1-2c2v5nm
# ncurses@=6.4%gcc@=8.3.1~symlinks+termlib abi=none build_system=autotools arch=linux-rhel7-power9le
module load ncurses/6.4-gcc-8.3.1-fsvemkj
# readline@=8.2%gcc@=8.3.1 build_system=autotools patches=bbf97f1 arch=linux-rhel7-power9le
module load readline/8.2-gcc-8.3.1-3c4bn2u
# gdbm@=1.23%gcc@=8.3.1 build_system=autotools arch=linux-rhel7-power9le
module load gdbm/1.23-gcc-8.3.1-toaffdo
# zlib@=1.3%gcc@=8.3.1+optimize+pic+shared build_system=makefile arch=linux-rhel7-power9le
module load zlib/1.3-gcc-8.3.1-kcgmx7q
# perl@=5.38.0%gcc@=8.3.1+cpanm+opcode+open+shared+threads build_system=generic arch=linux-rhel7-power9le
module load perl/5.38.0-gcc-8.3.1-2q5cjcp
# openblas@=0.3.24%gcc@=8.3.1~bignuma~consistent_fpcsr+fortran~ilp64+locking+pic+shared build_system=makefile symbol_suffix=none threads=none arch=linux-rhel7-power9le
module load openblas/0.3.24-gcc-8.3.1-3d7wabl
# coinhsl@=2015.06.23%gcc@=8.3.1+blas build_system=autotools arch=linux-rhel7-power9le
module load coinhsl/2015.06.23-gcc-8.3.1-7dqtp5j
# ginkgo@=1.5.0.glu_experimental%gcc@=8.3.1+cuda~develtools~full_optimizations~hwloc~ipo~mpi~oneapi+openmp~rocm+shared build_system=cmake build_type=Release cuda_arch=70 generator=make arch=linux-rhel7-power9le
module load ginkgo/1.5.0.glu_experimental-gcc-8.3.1-bgvmliy
# magma@=2.6.2%gcc@=8.3.1+cuda+fortran~ipo~rocm+shared build_system=cmake build_type=Release cuda_arch=70 generator=make arch=linux-rhel7-power9le
module load magma/2.6.2-gcc-8.3.1-7m4gjwg
# metis@=5.1.0%gcc@=8.3.1~gdb~int64~ipo~real64+shared build_system=cmake build_type=Release generator=make patches=4991da9,93a7903,b1225da arch=linux-rhel7-power9le
module load metis/5.1.0-gcc-8.3.1-n3azkkr
# raja@=0.14.0%gcc@=8.3.1+cuda~examples~exercises~ipo+openmp~rocm+shared~tests build_system=cmake build_type=Release cuda_arch=70 generator=make arch=linux-rhel7-power9le
module load raja/0.14.0-gcc-8.3.1-dyzbglo
# spectrum-mpi@=rolling-release%gcc@=8.3.1 build_system=bundle arch=linux-rhel7-power9le
module load spectrum-mpi/rolling-release-gcc-8.3.1-kvpxd6q
# libsigsegv@=2.14%gcc@=8.3.1 build_system=autotools arch=linux-rhel7-power9le
module load libsigsegv/2.14-gcc-8.3.1-6snmsyn
# m4@=1.4.19%gcc@=8.3.1+sigsegv build_system=autotools patches=9dc5fbd,bfdffa7 arch=linux-rhel7-power9le
module load m4/1.4.19-gcc-8.3.1-gsjhfhn
# autoconf@=2.69%gcc@=8.3.1 build_system=autotools patches=35c4492,7793209,a49dd5b arch=linux-rhel7-power9le
module load autoconf/2.69-gcc-8.3.1-dlao2fj
# automake@=1.16.5%gcc@=8.3.1 build_system=autotools arch=linux-rhel7-power9le
module load automake/1.16.5-gcc-8.3.1-27dhugr
# libtool@=2.4.7%gcc@=8.3.1 build_system=autotools arch=linux-rhel7-power9le
module load libtool/2.4.7-gcc-8.3.1-4i2eord
# gmp@=6.2.1%gcc@=8.3.1+cxx build_system=autotools libs=shared,static patches=69ad2e2 arch=linux-rhel7-power9le
module load gmp/6.2.1-gcc-8.3.1-yphfqvg
# autoconf-archive@=2023.02.20%gcc@=8.3.1 build_system=autotools arch=linux-rhel7-power9le
module load autoconf-archive/2023.02.20-gcc-8.3.1-rplcy5b
# xz@=5.4.1%gcc@=8.3.1~pic build_system=autotools libs=shared,static arch=linux-rhel7-power9le
module load xz/5.4.1-gcc-8.3.1-rnv4z5d
# libxml2@=2.10.3%gcc@=8.3.1+pic~python+shared build_system=autotools arch=linux-rhel7-power9le
module load libxml2/2.10.3-gcc-8.3.1-fs62rg5
# pigz@=2.7%gcc@=8.3.1 build_system=makefile arch=linux-rhel7-power9le
module load pigz/2.7-gcc-8.3.1-zbsrkb7
# zstd@=1.5.5%gcc@=8.3.1+programs build_system=makefile compression=none libs=shared,static arch=linux-rhel7-power9le
module load zstd/1.5.5-gcc-8.3.1-4prtcsn
# tar@=1.34%gcc@=8.3.1 build_system=autotools zip=pigz arch=linux-rhel7-power9le
module load tar/1.34-gcc-8.3.1-2jday4m
# gettext@=0.21.1%gcc@=8.3.1+bzip2+curses+git~libunistring+libxml2+tar+xz build_system=autotools arch=linux-rhel7-power9le
module load gettext/0.21.1-gcc-8.3.1-ktcyrwa
# texinfo@=7.0.3%gcc@=8.3.1 build_system=autotools arch=linux-rhel7-power9le
module load texinfo/7.0.3-gcc-8.3.1-5hajhsb
# mpfr@=4.2.0%gcc@=8.3.1 build_system=autotools libs=shared,static arch=linux-rhel7-power9le
module load mpfr/4.2.0-gcc-8.3.1-xuqovag
# suite-sparse@=5.13.0%gcc@=8.3.1~cuda~graphblas~openmp+pic build_system=generic arch=linux-rhel7-power9le
module load suite-sparse/5.13.0-gcc-8.3.1-6goqhuf
# umpire@=6.0.0%gcc@=8.3.1~c+cuda~device_alloc~deviceconst~examples~fortran~ipo~numa~openmp~rocm~shared build_system=cmake build_type=Release cuda_arch=70 generator=make tests=none arch=linux-rhel7-power9le
module load umpire/6.0.0-gcc-8.3.1-g3n426t


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

