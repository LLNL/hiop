#  NOTE: The following is required when running from Gitlab CI via slurm job
source /etc/profile.d/modules.sh
module use -a /qfs/projects/exasgd/src/cameron-spack/share/spack/modules/linux-centos7-haswell

# Load spack-built modules
# blt@0.4.1%gcc@7.3.0 arch=linux-centos7-broadwell
module load blt-0.4.1-gcc-7.3.0-twuxein
# camp@0.2.2%gcc@7.3.0+cuda~ipo~rocm~tests build_type=RelWithDebInfo cuda_arch=60 arch=linux-centos7-broadwell
module load camp-0.2.2-gcc-7.3.0-u3kctft
# cmake@3.21.4%gcc@7.3.0~doc+ncurses+openssl+ownlibs~qt build_type=Release arch=linux-centos7-broadwell
module load cmake-3.21.4-gcc-7.3.0-amjh4ql
# coinhsl@2015.06.23%gcc@7.3.0+blas arch=linux-centos7-haswell
module load coinhsl-2015.06.23-gcc-7.3.0-rp3qhnu
# cub@1.12.0-rc0%gcc@7.3.0 arch=linux-centos7-broadwell
module load cub-1.12.0-rc0-gcc-7.3.0-4zav6ns
# ginkgo@glu%gcc@7.3.0+cuda~develtools~full_optimizations~hwloc~ipo~oneapi+openmp~rocm+shared build_type=Release cuda_arch=60 arch=linux-centos7-broadwell
module load ginkgo-glu-gcc-7.3.0-vozd2so
# magma@2.6.2%gcc@7.3.0+cuda+fortran~ipo~rocm+shared build_type=RelWithDebInfo cuda_arch=60 arch=linux-centos7-broadwell
module load magma-2.6.2-gcc-7.3.0-6g2we7v
# metis@5.1.0%gcc@7.3.0~gdb~int64~real64+shared build_type=Release patches=4991da9,b1225da arch=linux-centos7-haswell
module load metis-5.1.0-gcc-7.3.0-mp3iiiv
# openblas@0.3.18%gcc@4.8.5~bignuma~consistent_fpcsr~ilp64+locking+pic+shared symbol_suffix=none threads=none arch=linux-centos7-haswell
module load openblas-0.3.18-gcc-4.8.5-k6y2hl7
# raja@0.14.0%gcc@7.3.0+cuda~examples~exercises~ipo+openmp~rocm~shared~tests build_type=RelWithDebInfo cuda_arch=60 arch=linux-centos7-broadwell
module load raja-0.14.0-gcc-7.3.0-ui2yfv6
# suite-sparse@5.7.2%gcc@7.3.0~cuda~graphblas~openmp+pic~tbb arch=linux-centos7-haswell
module load suite-sparse-5.7.2-gcc-7.3.0-lkj5mue
# umpire@6.0.0%gcc@7.3.0+c+cuda~deviceconst~examples~fortran~ipo~numa~openmp~rocm~shared build_type=RelWithDebInfo cuda_arch=60 tests=none arch=linux-centos7-broadwell
module load umpire-6.0.0-gcc-7.3.0-pq7d6om

# Load system modules
module load gcc/7.3.0
module load openmpi/4.1.0
module load cuda/10.2.89

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
