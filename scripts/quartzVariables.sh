module load gcc/8.3.1 cmake/3.18 mkl/2019.0 mvapich2/2.3

EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DHIOP_COINHSL_DIR:STRING=/usr/workspace/chiang7/software/quartz/COIN-OR/build"
EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DHIOP_METIS_DIR:STRING=/usr/workspace/chiang7/software/quartz/COIN-OR/build"
EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DHIOP_USE_RAJA:STRING=ON -DRAJA_DIR:STRING=/g/g92/chiang7/workspaces/chiang7/software/quartz/LLNL/RAJA/build_opt/_install -Dumpire_DIR:STRING=/g/g92/chiang7/workspaces/chiang7/software/quartz/LLNL/Umpire/build_opt/_install"

export CMAKE_CACHE_SCRIPT=gcc-cpu.cmake


