#!/bin/bash

[[ -f /etc/prfile.d/modules.sh ]] && \
  source /etc/profile.d/modules.sh

declare -a cmake_args
builddir=build
[ -d $builddir ] && rm -rf $builddir; mkdir $builddir

if type module
then
  proj_dir=/qfs/projects/exasgd
  mod_dir=/share/apps
  cluster=$(uname -n | cut -d'.' -f1 | sed -E 's/[[:digit:]]//g')
  module purge
  if [ "$cluster" == "newell" ]
  then
    gcc=7.4.0
    cmake=3.16.4
    openmpi=3.1.5
    cuda=10.2
    magma=2.5.2_cuda10.2
    metis=5.1.0
    cmake_args+=("-DHIOP_USE_GPU=ON -DHIOP_MAGMA_DIR=$mod_dir/magma/2.5.2/cuda10.2/ \
      -DHIOP_USE_MPI=ON -DHIOP_WITH_KRON_REDUCTION=ON \
      -DHIOP_UMFPACK_DIR=$proj_dir/$cluster/suitesparse \
      -DHIOP_METIS_DIR=$mod_dir/metis/5.1.0")
    export NVBLAS_CONFIG_FILE=$proj_dir/newell/nvblas.conf
  elif [ "$cluster" == "marianas" ]
  then
    gcc=7.3.0
    cmake=3.15.3
    openmpi=3.1.3
    cuda=10.1.243
    magma=2.5.2_cuda10.2
    metis=5.1.0
    cmake_args+=("-DHIOP_USE_GPU=ON -DHIOP_MAGMA_DIR=$mod_dir/magma/2.5.2/cuda10.2/ \
      -DHIOP_USE_MPI=ON -DHIOP_WITH_KRON_REDUCTION=ON \
      -DHIOP_UMFPACK_DIR=$proj_dir/$cluster/suitesparse \
      -DHIOP_METIS_DIR=$mod_dir/metis/5.1.0")
    export NVBLAS_CONFIG_FILE=$proj_dir/newell/nvblas.conf
  else
    echo Generic Build
    echo Note: NVBLAS_CONFIG_FILE will not be set.
  fi

  module load gcc/$gcc
  module load cmake/$cmake
  module load openmpi/$openmpi
  module load cuda/$cuda
  module load magma/$magma
  module load metis/$metis
fi

make_flags="-j 8"

cmake_args+=(
  "-DHIOP_USE_MPI=ON -DHIOP_DEEPCHECKS=OFF -DCMAKE_BUILD_TYPE=RELEASE"
  "-DHIOP_USE_MPI=OFF -DHIOP_DEEPCHECKS=OFF -DCMAKE_BUILD_TYPE=RELEASE"
  "-DHIOP_USE_MPI=ON -DHIOP_DEEPCHECKS=ON -DCMAKE_BUILD_TYPE=RELEASE"
  "-DHIOP_USE_MPI=OFF -DHIOP_DEEPCHECKS=ON -DCMAKE_BUILD_TYPE=RELEASE"
  "-DHIOP_USE_MPI=ON -DHIOP_DEEPCHECKS=OFF -DCMAKE_BUILD_TYPE=DEBUG"
  "-DHIOP_USE_MPI=OFF -DHIOP_DEEPCHECKS=OFF -DCMAKE_BUILD_TYPE=DEBUG"
  "-DHIOP_USE_MPI=ON -DHIOP_DEEPCHECKS=ON -DCMAKE_BUILD_TYPE=DEBUG"
)

for i in $(seq 0 ${#cmake_args[@]})
do
  build=${cmake_args[i]}
  echo
  echo "Building with cmake args: $build"
  echo
  [ -d $builddir ] && rm -rf $builddir; mkdir $builddir
  cd $builddir

  set -x
  pwd
  cmake $build .. || exit 1
  make $make_flags || exit 1
  ctest || exit 1
  popd
  set +x

  echo
  echo - - - - - - - - - - - - - - -
  echo "Build $[1 + i] / ${#cmake_args[@]} successful."
  echo - - - - - - - - - - - - - - -
  echo

  cd ..
done

echo
echo All major builds were successfull.
echo

if [[ $OSTYPE =~ darwin* ]]
then
  echo
  echo Found OSX
  echo Running with clang sanitize
  echo

  [ -d $builddir ] && rm -rf $builddir; mkdir $builddir
  cd $builddir
  CC=clang CXX=clang++ cmake \
    -DCMAKE_CXX_FLAGS="-fsanitize=nullability,undefined,integer,alignment" \
    -DHIOP_USE_MPI=ON -DHIOP_DEEPCHECKS=ON -DCMAKE_BUILD_TYPE=DEBUG .. || exit 1
  make $make_flags || exit 1
  ctest || exit 1
  mpiexec -np 2 ./src/Drivers/nlpDenseCons_ex1.exe 
  mpiexec -np 2 ./src/Drivers/nlpDenseCons_ex2.exe 
  mpiexec -np 2 ./src/Drivers/nlpDenseCons_ex3.exe 
fi
