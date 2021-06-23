
# Generate a sane default nvblas configuration file
function generateNvblasConfigFile {
  if [[ $# -ne 2 ]]
  then
    echo 'Usage: generateNvblasConfigFile builddir blaslib'
    exit 1
  fi
  local builddir=$1
  local blaslib=$2

  if [ -f $builddir/nvblas.conf ]
  then
    rm $builddir/nvblas.conf
  fi

  if [[ ! -f $builddir/nvblas.conf ]]; then
    cat > $builddir/nvblas.conf <<-EOD
NVBLAS_LOGFILE  nvblas.log
NVBLAS_CPU_BLAS_LIB blaslib
NVBLAS_GPU_LIST ALL
NVBLAS_TILE_DIM 2048
NVBLAS_AUTOPIN_MEM_ENABLED
EOD
  fi
  export NVBLAS_CONFIG_FILE=$builddir/nvblas.conf
}
