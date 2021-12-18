#  NOTE: The following is required when running from Gitlab CI via slurm job
source /etc/profile.d/modules.sh
module use -a /usr/share/Modules/modulefiles
module use -a /share/apps/modules/tools
module use -a /share/apps/modules/compilers
module use -a /share/apps/modules/mpi
module use -a /etc/modulefiles

module load rocm/4.5.1
module load umpire/6.0.0
module load raja/0.14.0
module load gcc/8.4.0
module load openblas/0.3.18
module load cmake/3.19.6
module load magma/2.6.1

export LD_LIBRARY_PATH=/share/apps/openmpi/4.1.1/gcc/8.1.0/lib
export PATH=/share/apps/openmpi/4.1.1/gcc/8.1.0/bin:$PATH

# For some reason, OS libstdc++ keeps being found before GCC 8.4.0, so we have
# to force this link directory. GCC 4.8.5 is far too old...
export EXTRA_CMAKE_ARGS="-DHIOP_EXTRA_LINK_FLAGS=-L/share/apps/gcc/8.4.0/lib64;-Wl,-rpath,/share/apps/gcc/8.4.0/lib64"
export CMAKE_CACHE_SCRIPT=clang-hip.cmake
