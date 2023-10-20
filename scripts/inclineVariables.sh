#!/bin/bash

. /etc/profile.d/modules.sh

module purge

# MPI module is finnicky on incline
modules=$(module list 2>&1)
if echo $modules | grep -q 'openmpi'; then
  module load gcc/8.4.0
  module rm openmpi
fi

# System modules
module load gcc/8.4.0
module load openmpi/4.1.4
module load rocm/5.3.0
module load cmake/3.21.4

# Spack modules
module use -a /vast/projects/exasgd/spack/install/modules/linux-centos7-zen3
# umpire@6.0.0%clang@15.0.0-rocm5.3.0 cxxflags="--gcc-toolchain=/share/apps/gcc/8.4.0/" +c~cuda~device_alloc~deviceconst~examples~fortran~ipo~numa~openmp+rocm+shared amdgpu_target=gfx908 build_system=cmake build_type=RelWithDebInfo generator=make tests=none arch=linux-centos7-zen
module load umpire-6.0.0-clang-15.0.0-rocm5.3.0-hsuiw34
# magma@2.6.2%clang@15.0.0-rocm5.3.0 cxxflags="--gcc-toolchain=/share/apps/gcc/8.4.0/" ~cuda+fortran~ipo+rocm+shared amdgpu_target=gfx908 build_system=cmake build_type=Release generator=make arch=linux-centos7-zen
module load magma-2.6.2-clang-15.0.0-rocm5.3.0-failpgu
# raja@0.14.0%clang@15.0.0-rocm5.3.0 cxxflags="--gcc-toolchain=/share/apps/gcc/8.4.0/" ~cuda~examples~exercises~ipo~openmp+rocm+shared~tests amdgpu_target=gfx908 build_system=cmake build_type=Release generator=make arch=linux-centos7-zen
module load raja-0.14.0-clang-15.0.0-rocm5.3.0-x4u3jfh

export CC=$(which clang)
export CXX=$(which clang++)
export FC=$(which gfortran)

export EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DAMDGPU_TARGETS='gfx908'"
export CMAKE_CACHE_SCRIPT=clang-hip.cmake
