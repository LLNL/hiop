# Attempts to build with all options enabled. This is the default build for
# continuous integration

defaultBuild()
{
  if [[ "$BUILD" == "1" ]]; then
    if [[ -f $BUILDDIR/CMakeCache.txt ]]; then
      rm -f $BUILDDIR/CMakeCache.txt || exit 1
    fi
    mkdir -p $BUILDDIR || exit 1
    echo
    echo Build step
    echo
    local SRCDIR=$PWD
    pushd $BUILDDIR || exit 1
    eval "cmake -C $SRCDIR/scripts/$CMAKE_CACHE_SCRIPT $EXTRA_CMAKE_ARGS .." || exit 1
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
