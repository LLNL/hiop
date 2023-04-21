
module use -a /usr/workspace/hiop/quartz/software/spack_modules/linux-rhel8-broadwell

module purge

module load coinhsl-2015.06.23-gcc-10.3.1-laebmhu
module load metis-5.1.0-gcc-10.3.1-jaquufw
module load openblas-0.3.20-gcc-10.3.1-5ahntf5

module load raja-0.14.0-gcc-10.3.1-pjdruyn 
module load umpire-6.0.0-gcc-10.3.1-sq7yi4q
module load zlib-1.2.12-gcc-10.3.1-q4d3dyj

module load gcc/10.3.1
module load mvapich2/2.3.6
module load cmake/3.22.4
module load python/3.9.12

#export EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DHIOP_CTEST_LAUNCH_COMMAND:STRING='jsrun -n 2 -a 1 -c 1 -g 1'"
export EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DHIOP_USE_RAJA:STRING=ON"
export CMAKE_CACHE_SCRIPT=gcc-cpu.cmake





