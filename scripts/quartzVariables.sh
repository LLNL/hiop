module load gcc/9.3.1 
module load mkl/2020.0
module load mvapich2/2.3
module load cmake/3.22.4

EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DHIOP_COINHSL_DIR:STRING=/usr/workspace/chiang7/software/quartz/COIN-OR/build"
EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DHIOP_METIS_DIR:STRING=/usr/workspace/chiang7/software/quartz/COIN-OR/build"
EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DHIOP_USE_RAJA:STRING=ON -DRAJA_DIR:STRING=/g/g92/chiang7/workspaces/chiang7/software/quartz/LLNL/RAJA/build_opt/_install -Dumpire_DIR:STRING=/g/g92/chiang7/workspaces/chiang7/software/quartz/LLNL/Umpire/build_opt/_install

export CMAKE_CACHE_SCRIPT=gcc-cpu.cmake


