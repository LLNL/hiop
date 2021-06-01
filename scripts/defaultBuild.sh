# Attempts to build with all options enabled. This is the default build for
# continuous integration

defaultBuild()
{
  export CMAKE_OPTIONS="\
      -DCMAKE_BUILD_TYPE=Debug \
      -DHIOP_BUILD_SHARED=ON \
      -DHIOP_BUILD_STATIC=ON \
      -DENABLE_TESTS=ON \
      -DHIOP_USE_MPI=On \
      -DHIOP_SPARSE=On \
      -DHIOP_DEEPCHECKS=ON \
      -DRAJA_DIR=$MY_RAJA_DIR \
      -DHIOP_USE_RAJA=On \
      -Dumpire_DIR=$MY_UMPIRE_DIR \
      -DHIOP_USE_UMPIRE=On \
      -DHIOP_WITH_KRON_REDUCTION=ON \
      -DHIOP_UMFPACK_DIR=$MY_UMFPACK_DIR \
      -DHIOP_METIS_DIR=$MY_METIS_DIR \
      -DHIOP_USE_GPU=ON \
      -DHIOP_USE_CUDA=ON \
      -DHIOP_MAGMA_DIR=$MY_HIOP_MAGMA_DIR \
      -DHIOP_NVCC_ARCH=$MY_NVCC_ARCH \
      -DHIOP_COINHSL_DIR=$MY_COINHSL_DIR \
      $EXTRA_CMAKE_ARGS"

  if [[ "$BUILD" == "1" ]]; then
    if [[ -f $BUILDDIR/CMakeCache.txt ]]; then
      rm -f $BUILDDIR/CMakeCache.txt || exit 1
    fi
    mkdir -p $BUILDDIR || exit 1
    echo
    echo Build step
    echo
    pushd $BUILDDIR                             || exit 1
    cmake $CMAKE_OPTIONS ..                     || exit 1
    $MAKE_CMD || exit 1
    popd
  fi

  if [[ "$TEST" == "1" ]]; then
    echo
    echo Testing step
    echo

    pushd $BUILDDIR || exit 1
    $CTEST_CMD || exit 1
    popd
  fi
}
