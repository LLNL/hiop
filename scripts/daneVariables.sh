module use -a /usr/workspace/hiop/software/spack_modules_202408/linux-rhel8-icelake

module purge

module load python/3.10.8
module load cmake/3.23.1
module load gcc/10.3.1

# cmake@=3.23.1%gcc@=10.3.1~doc+ncurses+ownlibs build_system=generic build_type=Release patches=dbc3892 arch=linux-rhel8-icelake
module load cmake/3.23.1-linux-rhel8-icelake-7c47exg
# glibc@=2.28%gcc@=10.3.1 build_system=autotools arch=linux-rhel8-icelake
module load glibc/2.28-linux-rhel8-icelake-cdykmru
# gcc-runtime@=10.3.1%gcc@=10.3.1 build_system=generic arch=linux-rhel8-icelake
module load gcc-runtime/10.3.1-linux-rhel8-icelake-4en2kx5
# blt@=0.4.1%gcc@=10.3.1 build_system=generic arch=linux-rhel8-icelake
module load blt/0.4.1-linux-rhel8-icelake-2bquhhy
# gmake@=4.4.1%gcc@=10.3.1~guile build_system=generic arch=linux-rhel8-icelake
module load gmake/4.4.1-linux-rhel8-icelake-pn7rcew
# camp@=0.2.3%gcc@=10.3.1~cuda~ipo+openmp~rocm~tests build_system=cmake build_type=Release generator=make patches=cb9e25b arch=linux-rhel8-icelake
module load camp/0.2.3-linux-rhel8-icelake-eazmpee
# berkeley-db@=18.1.40%gcc@=10.3.1+cxx~docs+stl build_system=autotools patches=26090f4,b231fcc arch=linux-rhel8-icelake
module load berkeley-db/18.1.40-linux-rhel8-icelake-nsdhyqb
# libiconv@=1.17%gcc@=10.3.1 build_system=autotools libs=shared,static arch=linux-rhel8-icelake
module load libiconv/1.17-linux-rhel8-icelake-rttf7cf
# diffutils@=3.10%gcc@=10.3.1 build_system=autotools arch=linux-rhel8-icelake
module load diffutils/3.10-linux-rhel8-icelake-kbt6ef4
# bzip2@=1.0.8%gcc@=10.3.1~debug~pic+shared build_system=generic arch=linux-rhel8-icelake
module load bzip2/1.0.8-linux-rhel8-icelake-v6xpk32
# pkgconf@=2.2.0%gcc@=10.3.1 build_system=autotools arch=linux-rhel8-icelake
module load pkgconf/2.2.0-linux-rhel8-icelake-mwiy3yh
# ncurses@=6.5%gcc@=10.3.1~symlinks+termlib abi=none build_system=autotools patches=7a351bc arch=linux-rhel8-icelake
module load ncurses/6.5-linux-rhel8-icelake-ynsc6ow
# readline@=8.2%gcc@=10.3.1 build_system=autotools patches=bbf97f1 arch=linux-rhel8-icelake
module load readline/8.2-linux-rhel8-icelake-wzrrzmb
# gdbm@=1.23%gcc@=10.3.1 build_system=autotools arch=linux-rhel8-icelake
module load gdbm/1.23-linux-rhel8-icelake-ckimnlc
# zlib-ng@=2.2.1%gcc@=10.3.1+compat+new_strategies+opt+pic+shared build_system=autotools arch=linux-rhel8-icelake
module load zlib-ng/2.2.1-linux-rhel8-icelake-awsj4js
# perl@=5.40.0%gcc@=10.3.1+cpanm+opcode+open+shared+threads build_system=generic arch=linux-rhel8-icelake
module load perl/5.40.0-linux-rhel8-icelake-bheybro
# openblas@=0.3.27%gcc@=10.3.1~bignuma~consistent_fpcsr+dynamic_dispatch+fortran~ilp64+locking+pic+shared build_system=makefile symbol_suffix=none threads=none arch=linux-rhel8-icelake
module load openblas/0.3.27-linux-rhel8-icelake-tvnt3p7
# coinhsl@=2015.06.23%gcc@=10.3.1+blas build_system=autotools arch=linux-rhel8-icelake
module load coinhsl/2015.06.23-linux-rhel8-icelake-nszs3vc
# metis@=5.1.0%gcc@=10.3.1~gdb~int64~ipo~real64+shared build_system=cmake build_type=Release generator=make patches=4991da9,93a7903,b1225da arch=linux-rhel8-icelake
module load metis/5.1.0-linux-rhel8-icelake-nalw554
# mvapich2@=2.3.7%gcc@=10.3.1~alloca~cuda~debug~hwloc_graphics~hwlocv2+regcache+wrapperrpath build_system=autotools ch3_rank_bits=32 fabrics=mrail file_systems=auto patches=d98d8e7 process_managers=auto threads=multiple arch=linux-rhel8-icelake
module load mvapich2/2.3.7-linux-rhel8-icelake-ewsxyd4
# raja@=0.14.0%gcc@=10.3.1~cuda~desul~examples~exercises~ipo~omptask+openmp~plugins~rocm~run-all-tests+shared~tests~vectorization build_system=cmake build_type=Release generator=make arch=linux-rhel8-icelake
module load raja/0.14.0-linux-rhel8-icelake-tvdflyy
# umpire@=6.0.0%gcc@=10.3.1~asan~backtrace~c~cuda~dev_benchmarks~device_alloc~deviceconst~examples~fortran~ipc_shmem~ipo~mpi~numa+openmp~openmp_target~rocm~sanitizer_tests~shared~sqlite_experimental~tools~werror build_system=cmake build_type=Release generator=make tests=none arch=linux-rhel8-icelake
module load umpire/6.0.0-linux-rhel8-icelake-dlnf5u3

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

export CMAKE_CACHE_SCRIPT=gcc-cpu.cmake
