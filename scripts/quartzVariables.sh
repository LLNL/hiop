module load gcc/8.3.1 cmake/3.18 mkl/2019.0 mvapich2/2.3
#export LD_LIBRARY_PATH=/g/g92/chiang7/workspace/software/quartz/COIN-OR/build/lib:$LD_LIBRARY_PATH
#export PATH=/g/g92/chiang7/workspace/software/quartz/COIN-OR/build/bin:$PATH

EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DHIOP_COINHSL_DIR:STRING=/usr/workspace/chiang7/software/quartz/COIN-OR/build"
EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DHIOP_METIS_DIR:STRING=/usr/workspace/chiang7/software/quartz/COIN-OR/build"
export CMAKE_CACHE_SCRIPT=gcc-cpu.cmake



