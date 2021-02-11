# Contains functions to run all reasonable configurations of HiOp
# This script will be sourced by the top-level BUILD.sh script, and the
# appropriate functions will be called from there.

# We don't want verbose output when running all possible builds, the output
# becomes very difficult to read.
set +xv

# If any build configurations fail to build, they will be written here
# and reported after all build configurations have run
export logFile=${BUILDMATRIX_LOGFILE:-"$BUILDDIR/../buildmatrix.log"}

# Error codes for specific cases
export success=0
export buildError=1
export testError=2
export cmakeError=3
export strerr=(Success BuildError TestError CmakeError)

# Options used to configure CMake in current build; set in buildMatrix function
export cmakeOptions=""

# Global build ID
export buildNo=0

# Names of each column in the logfile
export columnNames="buildID;buildStatus;cmakeOptions"

# If the tests are verbose, we overrun the log output limit
export CTEST_CMD=$(echo $CTEST_CMD | sed 's/ -VV//')

# Logs the output of a given run to the logfile
logRun()
{
  echo "Build status:"
  echo "$buildNo;${strerr[$1]};$cmakeOptions" \
    | sed -e 's/\s\+/ /g' \
    | tee --append "$logFile"
  ((buildNo++))
}

reportRuns()
{
  local numFailures=$(grep -v 'Success' $logFile | wc -l)
  ((numFailures--)) # Don't count the header row
  echo "Found $numFailures failures. Full report:"
  cat $logFile
  return $numFailures
}

# When running build matrix from CI, it's nicer to have each config run
# as a seperate job. The cmake configurations are passed 
CIBuildMatrix()
{
  local baseCmakeOptions=" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DENABLE_TESTING=ON \
    -DHIOP_DEEPCHECKS=ON \
    "
  
  eval 'rajaOp=$RAJAOP'
  eval 'gpuOp=$GPUOP'
  eval 'kronRedOp=$KRONREDOP'
  eval 'mpiOp=$MPIOP'
  eval 'sparseOp=$SPARSEOP'
  export cmakeOptions="$baseCmakeOptions $rajaOp $gpuOp $kronRedOp $mpiOp $sparseOp"
  buildAndTest $doTest
  logRun $?
  reportRuns
}

# Iterates through every configuration of CMake variables and call the 
# _buildAndTest_ function to ensure the configuration is functional
buildMatrix()
{
  [[ -f $logFile ]] && rm $logFile
  touch $logFile
  echo "$columnNames" >> $logFile

  local doTest=${1:-1}

  local baseCmakeOptions=" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DENABLE_TESTING=ON \
    -DHIOP_DEEPCHECKS=ON \
    "

  rajaOpts=(
    '-DHIOP_USE_RAJA=OFF'
    "-DHIOP_USE_RAJA=ON \
      -DRAJA_DIR=$MY_RAJA_DIR \
      -DHIOP_USE_UMPIRE=On \
      -Dumpire_DIR=$MY_UMPIRE_DIR"
    )
  gpuOpts=(
    '-DHIOP_USE_GPU=OFF'
    "-DHIOP_USE_GPU=ON \
      -DHIOP_NVCC_ARCH=$MY_NVCC_ARCH \
      -DHIOP_MAGMA_DIR=$MY_HIOP_MAGMA_DIR"
    )
  kronRedOpts=(
    "-DHIOP_WITH_KRON_REDUCTION=ON \
      -DHIOP_METIS_DIR=$MY_METIS_DIR \
      -DHIOP_UMFPACK_DIR=$MY_UMFPACK_DIR"
    "-DHIOP_WITH_KRON_REDUCTION=OFF"
    )
  mpiOpts=(
    '-DHIOP_USE_MPI=ON'
    '-DHIOP_USE_MPI=OFF'
    )
  sparseOpts=(
    '-DHIOP_SPARSE=OFF'
    "-DHIOP_SPARSE=ON \
      -DHIOP_METIS_DIR=$MY_METIS_DIR \
      -DHIOP_COINHSL_DIR=$MY_COINHSL_DIR"
    )

  # Are we running parallel jobs in ci?
  if [[ $FULL_BUILD_MATRIX_PARALLEL -eq 1 ]]; then
    rajaOp="${rajaOpts[CI_RAJAOP]}"
    gpuOp="${gpuOpts[CI_GPUOP]}"
    kronRedOp="${kronRedOpts[CI_KRONREDOP]}"
    mpiOp="${mpiOpts[CI_MPIOP]}"
    sparseOp="${sparseOpts[CI_SPARSEOP]}"

    echo "Using:"
    echo "rajaOp: $rajaOp"
    echo "gpuOp: $gpuOp"
    echo "kronRedOp: $kronRedOp"
    echo "mpiOp: $mpiOp"
    echo "sparseOp: $sparseOp"
    echo
  
    export cmakeOptions="$baseCmakeOptions $rajaOp $gpuOp $kronRedOp $mpiOp $sparseOp"
    buildAndTest $doTest
    logRun $?
    reportRuns
  else # Are we running a single long-running job to test all configs?
    for rajaOp in "${rajaOpts[@]}"; do
      for gpuOp in "${gpuOpts[@]}"; do
        for kronRedOp in "${kronRedOpts[@]}"; do
          for mpiOp in "${mpiOpts[@]}"; do
            for sparseOp in "${sparseOpts[@]}"; do
              export cmakeOptions="$baseCmakeOptions $rajaOp $gpuOp $kronRedOp $mpiOp $sparseOp"
              buildAndTest $doTest
              logRun $?
              reportRuns
            done
          done
        done
      done
    done
  fi

  reportRuns
  return $?
}

buildAndTest()
{
  echo
  echo CMake Options:
  echo
  echo $cmakeOptions
  echo

  local doTest=${1:-1}
  local rc=0

  echo
  echo Configuring
  echo

  rm -rf $BUILDDIR
  mkdir -p $BUILDDIR
  pushd $BUILDDIR
  cmake $cmakeOptions .. || return $cmakeError
  popd

  echo
  echo Building
  echo
  pushd $BUILDDIR
  $MAKE_CMD
  rc=$?
  popd
  [ $rc -ne 0 ] && return $buildError

  if [[ $doTest -eq 1 ]]; then
    echo
    echo Testing
    echo
    pushd $BUILDDIR
    $CTEST_CMD 
    rc=$?
    popd
    [ $rc -ne 0 ] && return $testError
  fi
  return $success
}
