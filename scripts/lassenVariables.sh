module use -a /usr/WS1/opt_llnl/nai/software/spack/share/spack/modules/linux-rhel7-power9le

module purge

module load camp-0.2.2-gcc-8.3.1-7pw25ce
module load magma-2.6.2-gcc-8.3.1-7eouwik
module load coinhsl-2015.06.23-gcc-8.3.1-65kik3g
module load metis-5.1.0-gcc-8.3.1-mlpk4jh
module load perl-5.34.1-gcc-8.3.1-q7xpwxc
module load openblas-0.3.20-gcc-8.3.1-rm2c5ic
module load raja-0.14.0-gcc-8.3.1-ylug63w
module load umpire-6.0.0-gcc-8.3.1-tdfycms
module load zlib-1.2.12-gcc-8.3.1-i2ruf5x

module load cmake/3.20.2 
module load gcc/8.3.1 
module load cuda/11.7.0 
module load python/3.8.2

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

export EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DHIOP_USE_GINKGO=OFF -DCMAKE_CUDA_ARCHITECTURES=70"
export CMAKE_CACHE_SCRIPT=gcc-cuda.cmake
