module use /gpfs/wolf/proj-shared/csc359/src/spack/share/spack/modules/linux-rhel7-power9le

module purge

# Load spack-built modules
# automake@1.16.3%gcc@7.4.0 arch=linux-rhel7-power9le
module load exasgd-automake/1.16.3/gcc-7.4.0-qprf3ee
# berkeley-db@18.1.40%gcc@7.4.0+cxx~docs+stl patches=b231fcc4d5cff05e5c3a4814f6a5af0e9a966428dc2176540d2c05aff41de522 arch=linux-rhel7-power9le
module load exasgd-berkeley-db/18.1.40/gcc-7.4.0-hn7z7sb
# coinhsl@2015.06.23%gcc@7.4.0+blas arch=linux-rhel7-power9le
module load exasgd-coinhsl/2015.06.23/gcc-7.4.0-k5ullmz
# gdbm@1.19%gcc@7.4.0 arch=linux-rhel7-power9le
module load exasgd-gdbm/1.19/gcc-7.4.0-v4st5ql
# gmp@6.2.1%gcc@7.4.0 arch=linux-rhel7-power9le
module load exasgd-gmp/6.2.1/gcc-7.4.0-6svtfvr
# ipopt@3.12.10%gcc@7.4.0+coinhsl~debug~mumps arch=linux-rhel7-power9le
module load exasgd-ipopt/3.12.10/gcc-7.4.0-tj6jbm2
# libsigsegv@2.13%gcc@7.4.0 arch=linux-rhel7-power9le
module load exasgd-libsigsegv/2.13/gcc-7.4.0-garv4jn
# libtool@2.4.6%gcc@7.4.0 arch=linux-rhel7-power9le
module load exasgd-libtool/2.4.6/gcc-7.4.0-lbasl6y
module load exasgd-magma/2.6.1/cuda-10.2.89-system/gcc-7.4.0-exkqayg
# metis@5.1.0%gcc@7.4.0~gdb~int64~real64+shared build_type=Release patches=4991da938c1d3a1d3dea78e49bbebecba00273f98df2a656e38b83d55b281da1,b1225da886605ea558db7ac08dd8054742ea5afe5ed61ad4d0fe7a495b1270d2 arch=linux-rhel7-power9le
module load exasgd-metis/5.1.0/gcc-7.4.0-7cjo5kb
# mpfr@4.1.0%gcc@7.4.0 arch=linux-rhel7-power9le
module load exasgd-mpfr/4.1.0/gcc-7.4.0-bkusb67
# ncurses@6.2%gcc@7.4.0~symlinks+termlib abi=none arch=linux-rhel7-power9le
module load exasgd-ncurses/6.2/gcc-7.4.0-rzufnf5
# openblas@0.3.15%gcc@7.4.0~bignuma~consistent_fpcsr~ilp64+locking+pic+shared threads=none arch=linux-rhel7-power9le
module load exasgd-openblas/0.3.15/gcc-7.4.0-i4ax3dw
# perl@5.34.0%gcc@7.4.0+cpanm+shared+threads arch=linux-rhel7-power9le
module load exasgd-perl/5.34.0/gcc-7.4.0-fh5ekqg
# raja@0.13.0%gcc@7.4.0+cuda+examples+exercises~hip~ipo+openmp+shared~tests amdgpu_target=none build_type=RelWithDebInfo cuda_arch=none arch=linux-rhel7-power9le
module load exasgd-raja/0.13.0/cuda-10.2.89/gcc-7.4.0-hq3qj3a
# readline@8.1%gcc@7.4.0 arch=linux-rhel7-power9le
module load exasgd-readline/8.1/gcc-7.4.0-vqamj6w
# suite-sparse@5.8.1%gcc@7.4.0~cuda~openmp+pic~tbb arch=linux-rhel7-power9le
module load exasgd-suite-sparse/5.8.1/gcc-7.4.0-zggphid
# texinfo@6.5%gcc@7.4.0 patches=12f6edb0c6b270b8c8dba2ce17998c580db01182d871ee32b7b6e4129bd1d23a,1732115f651cff98989cb0215d8f64da5e0f7911ebf0c13b064920f088f2ffe1 arch=linux-rhel7-power9le
module load exasgd-texinfo/6.5/gcc-7.4.0-w4cx3wo
# 4q6czdk umpire@5.0.1+c+cuda~deviceconst~examples~fortran~hip~ipo~numa+openmp~shared amdgpu_target=none build_type=RelWithDebInfo cuda_arch=none tests=none
module load exasgd-umpire/5.0.1/cuda-10.2.89/gcc-7.4.0-4q6czdk
#
module load exasgd-strumpack/5.1.1/cuda-10.1.243/spectrum-mpi-10.3.1.2-20200121/gcc-7.4.0-a4l5d6w

# Load system modules
module load cuda/10.2.89
module load gcc/7.4.0
module load spectrum-mpi/10.3.1.2-20200121
module load cmake/3.18.2

#These are not picked up by the module for some reason
export CC=/sw/ascent/gcc/7.4.0/bin/gcc
export CXX=/sw/ascent/gcc/7.4.0/bin/g++
export FC=/sw/ascent/gcc/7.4.0/bin/gfortran

[ -f $PWD/nvblas.conf ] && rm $PWD/nvblas.conf
cat > $PWD/nvblas.conf <<-EOD
NVBLAS_LOGFILE  nvblas.log
NVBLAS_CPU_BLAS_LIB $OPENBLAS_LIBRARY_DIR/libopenblas.so
NVBLAS_GPU_LIST ALL
NVBLAS_TILE_DIM 2048
NVBLAS_AUTOPIN_MEM_ENABLED
EOD
export NVBLAS_CONFIG_FILE=$PWD/nvblas.conf
echo "Generated $PWD/nvblas.conf"

EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DCMAKE_CUDA_ARCHITECTURES=70 -DHIOP_TEST_WITH_BSUB=ON"
