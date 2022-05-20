#  NOTE: The following is required when running from Gitlab CI via slurm job
source /etc/profile.d/modules.sh
module use -a /usr/share/Modules/modulefiles
module use -a /share/apps/modules/tools
module use -a /share/apps/modules/compilers
module use -a /share/apps/modules/mpi
module use -a /etc/modulefiles
module use -a /qfs/projects/exasgd/src/cameron-spack/share/spack/modules/linux-rhel7-power9le

# Load spack-built modules
# blt@0.4.1%gcc@7.4.0 arch=linux-rhel7-power9le
module load blt-0.4.1-gcc-7.4.0-2th7jgq
# camp@0.2.2%gcc@7.4.0+cuda~ipo~rocm~tests build_type=RelWithDebInfo cuda_arch=70 arch=linux-rhel7-power9le
module load camp-0.2.2-gcc-7.4.0-vsu2jwh
# cmake@3.23.1%gcc@7.4.0~doc+ncurses+ownlibs~qt build_type=Release arch=linux-rhel7-power9le
module load cmake-3.23.1-gcc-7.4.0-ckfugtf
# coinhsl@2015.06.23%gcc@7.4.0+blas arch=linux-rhel7-power9le
module load coinhsl-2015.06.23-gcc-7.4.0-ts5vjfq
# cub@1.12.0%gcc@7.4.0 arch=linux-rhel7-power9le
module load cub-1.12.0-gcc-7.4.0-4qyvoqn
# ginkgo@glu%gcc@7.4.0+cuda~develtools~full_optimizations~hwloc~ipo~oneapi+openmp~rocm+shared build_type=Release cuda_arch=70 arch=linux-rhel7-power9le
module load ginkgo-glu-gcc-7.4.0-r5wjmju
# gmp@6.2.1%gcc@7.4.0 libs=shared,static arch=linux-rhel7-power9le
module load gmp-6.2.1-gcc-7.4.0-oea2aet
# magma@2.6.2%gcc@7.4.0+cuda+fortran~ipo~rocm+shared build_type=RelWithDebInfo cuda_arch=70 arch=linux-rhel7-power9le
module load magma-2.6.2-gcc-7.4.0-6yuqfpm
# metis@5.1.0%gcc@7.4.0~gdb~int64~real64+shared build_type=Release patches=4991da9,b1225da arch=linux-rhel7-power9le
module load metis-5.1.0-gcc-7.4.0-shhhyku
# mpfr@4.1.0%gcc@7.4.0 libs=shared,static arch=linux-rhel7-power9le
module load mpfr-4.1.0-gcc-7.4.0-tz5esun
# ncurses@6.2%gcc@7.4.0~symlinks+termlib abi=none arch=linux-rhel7-power9le
module load ncurses-6.2-gcc-7.4.0-kqhmmpv
# openblas@0.3.20%gcc@7.4.0~bignuma~consistent_fpcsr~ilp64+locking+pic+shared symbol_suffix=none threads=none arch=linux-rhel7-power9le
module load openblas-0.3.20-gcc-7.4.0-3zdqw2i
# openssl@1.1.1n%gcc@7.4.0~docs~shared certs=system arch=linux-rhel7-power9le
module load openssl-1.1.1n-gcc-7.4.0-mxyibws
# raja@0.14.0%gcc@7.4.0+cuda~examples~exercises~ipo+openmp~rocm+shared~tests build_type=RelWithDebInfo cuda_arch=70 arch=linux-rhel7-power9le
module load raja-0.14.0-gcc-7.4.0-sew5thv
# suite-sparse@5.10.1%gcc@7.4.0~cuda~graphblas~openmp+pic~tbb arch=linux-rhel7-power9le
module load suite-sparse-5.10.1-gcc-7.4.0-e5qockg
# umpire@6.0.0%gcc@7.4.0+c+cuda~deviceconst~examples~fortran~ipo~numa~openmp~rocm~shared build_type=RelWithDebInfo cuda_arch=70 tests=none arch=linux-rhel7-power9le
module load umpire-6.0.0-gcc-7.4.0-rpwrj4p
# zlib@1.2.12%gcc@7.4.0+optimize+pic+shared patches=0d38234 arch=linux-rhel7-power9le
module load zlib-1.2.12-gcc-7.4.0-d6xlzc6

# Load system modules
module load gcc/7.4.0
module load cuda/10.2
module load openmpi/3.1.5

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

export EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DCMAKE_CUDA_ARCHITECTURES=70"
export CMAKE_CACHE_SCRIPT=gcc-cuda.cmake
