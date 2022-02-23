module purge

module use -a /gpfs/wolf/csc359/proj-shared/src/spack/share/spack/modules/linux-rhel8-power9le

# Load spack-built modules
# autoconf@2.69%gcc@9.1.0 patches=35c4492,7793209,a49dd5b arch=linux-rhel8-power9le
module load autoconf-2.69-gcc-9.1.0-vnxzsnr
# autoconf-archive@2019.01.06%gcc@9.1.0 arch=linux-rhel8-power9le
module load autoconf-archive-2019.01.06-gcc-9.1.0-2kjmyyv
# automake@1.16.5%gcc@9.1.0 arch=linux-rhel8-power9le
module load automake-1.16.5-gcc-9.1.0-x5ndgg2
# blt@0.4.1%gcc@9.1.0 arch=linux-rhel8-power9le
module load blt-0.4.1-gcc-9.1.0-pvxpp36
# camp@0.2.2%gcc@9.1.0+cuda~ipo~rocm~tests build_type=RelWithDebInfo cuda_arch=70 arch=linux-rhel8-power9le
module load camp-0.2.2-gcc-9.1.0-kd2joie
# cmake@3.22.2%gcc@9.1.0~doc+ncurses+openssl+ownlibs~qt build_type=Release arch=linux-rhel8-power9le
module load cmake-3.22.2-gcc-9.1.0-bcxmnmg
# coinhsl@2015.06.23%gcc@9.1.0+blas arch=linux-rhel8-power9le
module load coinhsl-2015.06.23-gcc-9.1.0-go24cay
# cub@1.12.0-rc0%gcc@9.1.0 arch=linux-rhel8-power9le
module load cub-1.12.0-rc0-gcc-9.1.0-ipbwuis
# cuda@11.4.2%gcc@9.1.0~allow-unsupported-compilers~dev arch=linux-rhel8-power9le
module load cuda-11.4.2-gcc-9.1.0-4676kh5
# gmp@6.2.1%gcc@9.1.0 arch=linux-rhel8-power9le
module load gmp-6.2.1-gcc-9.1.0-76za3mf
# gnuconfig@2021-08-14%gcc@9.1.0 arch=linux-rhel8-power9le
module load gnuconfig-2021-08-14-gcc-9.1.0-wt2yuir
# libsigsegv@2.13%gcc@9.1.0 arch=linux-rhel8-power9le
module load libsigsegv-2.13-gcc-9.1.0-lrapsw3
# libtool@2.4.6%gcc@9.1.0 arch=linux-rhel8-power9le
module load libtool-2.4.6-gcc-9.1.0-c7ioa2h
# m4@1.4.19%gcc@9.1.0+sigsegv patches=9dc5fbd,bfdffa7 arch=linux-rhel8-power9le
module load m4-1.4.19-gcc-9.1.0-aw4chza
# magma@master%gcc@9.1.0+cuda+fortran~ipo~rocm+shared build_type=RelWithDebInfo cuda_arch=70 arch=linux-rhel8-power9le
module load magma-master-gcc-9.1.0-b3jqbnq
# metis@5.1.0%gcc@9.1.0~gdb~int64~real64+shared build_type=Release patches=4991da9,b1225da arch=linux-rhel8-power9le
module load metis-5.1.0-gcc-9.1.0-sg2ca2t
# mpfr@4.1.0%gcc@9.1.0 arch=linux-rhel8-power9le
module load mpfr-4.1.0-gcc-9.1.0-2qjplyw
# openblas@0.3.19%gcc@9.1.0~bignuma~consistent_fpcsr~ilp64+locking+pic+shared symbol_suffix=none threads=none arch=linux-rhel8-power9le
module load openblas-0.3.19-gcc-9.1.0-c6nslyv
# perl@5.30.1%gcc@9.1.0+cpanm+shared+threads arch=linux-rhel8-power9le
module load perl-5.30.1-gcc-9.1.0-qmsmncp
# raja@0.14.0%gcc@9.1.0+cuda~examples~exercises~ipo+openmp~rocm+shared~tests build_type=RelWithDebInfo cuda_arch=70 arch=linux-rhel8-power9le
module load raja-0.14.0-gcc-9.1.0-ili5h35
# suite-sparse@5.10.1%gcc@9.1.0~cuda~graphblas~openmp+pic~tbb arch=linux-rhel8-power9le
module load suite-sparse-5.10.1-gcc-9.1.0-nlk7pqe
# texinfo@6.5%gcc@9.1.0 patches=12f6edb,1732115 arch=linux-rhel8-power9le
module load texinfo-6.5-gcc-9.1.0-jkkfv4q
# umpire@6.0.0%gcc@9.1.0+c+cuda~deviceconst~examples~fortran~ipo~numa+openmp~rocm~shared build_type=RelWithDebInfo cuda_arch=70 tests=none arch=linux-rhel8-power9le
module load umpire-6.0.0-gcc-9.1.0-bosktbw

#Load system modules
module load gcc/9.1.0
module load spectrum-mpi/10.4.0.3-20210112
export CC=/sw/ascent/gcc/9.1.0-3/bin/gcc
export CXX=/sw/ascent/gcc/9.1.0-3/bin/g++
export FC=/sw/ascent/gcc/9.1.0-3/bin/gfortran

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
export CMAKE_CACHE_SCRIPT=gcc-cuda.cmake
