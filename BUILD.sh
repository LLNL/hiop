#!/bin/bash

#
# A script file that builds and runs/tests HiOp on various paricular clusters, such as
# summit and ascent @ ORNL and newell and marianas @ PNNL
#
# Usage: In a shell run
#
# ./BUILD.sh
#
#
# Sometimes the cluster name is not detected correctly; in this cases, one can specify
# the cluster name by prefixing the command with MY_CLUSTER=cluster_name, e.g., 
#
# MY_CLUSTER=ascent ./BUILD.sh
#
# All variables that will change the build script's behaviour:
#
# - MAKE_CMD command the script will use to run makefiles
# - CTEST_CMD command the script will use to run ctest
# - BUILDDIR path to temp build directory
# - EXTRA_CMAKE_ARGS extra arguments passed to CMake
# - MY_CLUSTER determines cluster-specific variables to use

export BASE_PATH=$(dirname $0)
export MAKE_CMD=${MAKE_CMD:-'make -j 8'}
export CTEST_CMD=${CTEST_CMD:-'ctest -VV --timeout 1800'}

# we want this settable via env var to make CI triggered builds simpler
export FULL_BUILD_MATRIX=${FULL_BUILD_MATRIX:-0}
export BUILDDIR=${BUILDDIR:-"$(pwd)/build"}
export EXTRA_CMAKE_ARGS=${EXTRA_CMAKE_ARGS:-""}
export BUILD=1
export TEST=1

cleanup() {
  echo
  echo Exit code $1 caught in build script.
  echo
  if [[ "$1" == "0" ]]; then
    echo BUILD_STATUS:0
  else
    echo
    echo Failure found in build script.
    echo
    echo BUILD_STATUS:1
  fi
}

trap 'cleanup $? $LINENO' EXIT

while [[ $# -gt 0 ]]
do
  case $1 in
    --build-only|-B)
      echo
      echo Building only
      echo
      export BUILD=1
      export TEST=0
      shift
      ;;
    --test-only|-T)
      echo
      echo Testing only
      echo
      export BUILD=0
      export TEST=1
      shift
      ;;
    --help|*)
      trap - 1 2 3 15 # we don't need traps if a user is asking for help
      cat <<EOD
      Usage:

        $ MY_CLUSTER='clustername' $0
      
      Optional arguments:
      
        --build-only          Only build, don't test
        --test-only           Only run tests, don't build
        --help                Show this message
EOD
      exit 1
      ;;
  esac
done

# set -xv

# If MY_CLUSTER is not set by user, try to discover it from environment
if [[ ! -v MY_CLUSTER ]]
then
  export MY_CLUSTER=`uname -n | sed -e 's/[0-9]//g' -e 's/\..*//'`
fi

# Some clusters have compute nodes with slightly different hostnames, so we
# set MY_CLUSTER appropriately
if [[ $MY_CLUSTER =~ ^newell.* ]]; then
  export MY_CLUSTER=newell
elif [[ $MY_CLUSTER =~ ^dl.* ]]; then
  export MY_CLUSTER=marianas
elif [[ $MY_CLUSTER =~ ^dmi.* ]]; then
  export MY_CLUSTER=incline
elif [[ $MY_CLUSTER =~ ^dane.* ]]; then
  export MY_CLUSTER=dane
fi

module purge

# If we have modules/variables defined for the current cluster, use them
if [ -f "./scripts/$(echo $MY_CLUSTER)Variables.sh" ]; then
  source "./scripts/$(echo $MY_CLUSTER)Variables.sh"
  echo "Using ./scripts/$(echo $MY_CLUSTER)Variables.sh"
fi

# The following is required when running from Gitlab CI via slurm job
if [ -z "$SLURM_SUBMIT_DIR" ]; then
    cd $BASE_PATH          || exit 1
fi

# Fail fast if we can't find NVBLAS_CONFIG_FILE since it's needed for all CUDA GPU builds
if [[ ! $MY_CLUSTER =~ incline ]] && [[ ! $MY_CLUSTER =~ spock ]] && [[ ! $MY_CLUSTER =~ quartz ]]; then
  if [[ ! -v NVBLAS_CONFIG_FILE ]] || [[ ! -f "$NVBLAS_CONFIG_FILE" ]]
  then
    echo "Please provide file 'nvblas.conf' or set variable to desired location."
    exit 1
  fi
fi

module list

source ./scripts/defaultBuild.sh
defaultBuild
exit $?
