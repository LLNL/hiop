module use /gpfs/wolf/proj-shared/csc359/src/spack/share/spack/modules/linux-rhel7-power9le

module purge

# Load spack-built modules

#utoconf@2.69%gcc@7.4.0 patches=35c449281546376449766f92d49fc121ca50e330e60fefcfc9be2af3253082c2,7793209b33013dc0f81208718c68440c5aae80e7a1c4b8d336e382525af791a7,a49dd5bac3b62daa0ff688ab4d508d71dbd2f4f8d7e2a02321926346161bf3ee arch=linux-rhel7-power9le
module load exasgd-autoconf/2.69/gcc-7.4.0-6vut5vy
# autoconf-archive@2019.01.06%gcc@7.4.0 arch=linux-rhel7-power9le
module load exasgd-autoconf-archive/2019.01.06/gcc-7.4.0-nn453cx
# automake@1.16.3%gcc@7.4.0 arch=linux-rhel7-power9le
module load exasgd-automake/1.16.3/gcc-7.4.0-hvhod3j
# berkeley-db@18.1.40%gcc@7.4.0+cxx~docs+stl patches=b231fcc4d5cff05e5c3a4814f6a5af0e9a966428dc2176540d2c05aff41de522 arch=linux-rhel7-power9le
module load exasgd-berkeley-db/18.1.40/gcc-7.4.0-ic4cqif
# blt@develop%gcc@7.4.0 arch=linux-rhel7-power9le
module load exasgd-blt/develop/gcc-7.4.0-gvzmwlc
# bzip2@1.0.8%gcc@7.4.0~debug~pic+shared arch=linux-rhel7-power9le
module load exasgd-bzip2/1.0.8/gcc-7.4.0-jty62q7
# camp@0.2.2%gcc@7.4.0+cuda~ipo~rocm~tests amdgpu_target=none build_type=RelWithDebInfo cuda_arch=70 arch=linux-rhel7-power9le
module load exasgd-camp/0.2.2/cuda-11.0.194/gcc-7.4.0-xtula6d
# cmake@3.18.2%gcc@7.4.0~doc+ncurses+openssl+ownlibs~qt build_type=Release patches=bf695e3febb222da2ed94b3beea600650e4318975da90e4a71d6f31a6d5d8c3d arch=linux-rhel7-power9le
module load exasgd-cmake/3.18.2/gcc-7.4.0-m42tpk4
# coinhsl@2015.06.23%gcc@7.4.0+blas arch=linux-rhel7-power9le
module load exasgd-coinhsl/2015.06.23/gcc-7.4.0-hacdh23
# cub@1.12.0-rc0%gcc@7.4.0 arch=linux-rhel7-power9le
module load exasgd-cub/1.12.0-rc0/gcc-7.4.0-iwyj63t
# cuda@11.0.194%gcc@7.4.0~dev arch=linux-rhel7-power9le
module load exasgd-cuda/11.0.194/gcc-7.4.0-e5nl5fx
# diffutils@3.8%gcc@7.4.0 arch=linux-rhel7-power9le
module load exasgd-diffutils/3.8/gcc-7.4.0-cy55hsj
# gdbm@1.19%gcc@7.4.0 arch=linux-rhel7-power9le
module load exasgd-gdbm/1.19/gcc-7.4.0-ahdwucz
# gmp@6.2.1%gcc@7.4.0 arch=linux-rhel7-power9le
module load exasgd-gmp/6.2.1/gcc-7.4.0-ur2a3rb
# gnuconfig@2021-08-14%gcc@7.4.0 arch=linux-rhel7-power9le
module load exasgd-gnuconfig/2021-08-14/gcc-7.4.0-qr6nxuq
# libiconv@1.16%gcc@7.4.0 libs=shared,static arch=linux-rhel7-power9le
module load exasgd-libiconv/1.16/gcc-7.4.0-idqno7d
# libsigsegv@2.13%gcc@7.4.0 arch=linux-rhel7-power9le
module load exasgd-libsigsegv/2.13/gcc-7.4.0-cbn4dja
# libtool@2.4.6%gcc@7.4.0 arch=linux-rhel7-power9le
module load exasgd-libtool/2.4.6/gcc-7.4.0-x5h54ly
# m4@1.4.19%gcc@7.4.0+sigsegv patches=9dc5fbd0d5cb1037ab1e6d0ecc74a30df218d0a94bdd5a02759a97f62daca573,bfdffa7c2eb01021d5849b36972c069693654ad826c1a20b53534009a4ec7a89 arch=linux-rhel7-power9le
module load exasgd-m4/1.4.19/gcc-7.4.0-nrrlksm
# magma@2.6.1%gcc@7.4.0+cuda+fortran~ipo~rocm+shared amdgpu_target=none build_type=RelWithDebInfo cuda_arch=70 arch=linux-rhel7-power9le
module load exasgd-magma/2.6.1/cuda-11.0.194/gcc-7.4.0-finywiw
# metis@5.1.0%gcc@7.4.0~gdb~int64~real64+shared build_type=Release patches=4991da938c1d3a1d3dea78e49bbebecba00273f98df2a656e38b83d55b281da1,b1225da886605ea558db7ac08dd8054742ea5afe5ed61ad4d0fe7a495b1270d2 arch=linux-rhel7-power9le
module load exasgd-metis/5.1.0/gcc-7.4.0-shhhyku
# mpfr@4.1.0%gcc@7.4.0 arch=linux-rhel7-power9le
module load exasgd-mpfr/4.1.0/gcc-7.4.0-33dvnf2
# ncurses@6.2%gcc@7.4.0~symlinks+termlib abi=none arch=linux-rhel7-power9le
module load exasgd-ncurses/6.2/gcc-7.4.0-kqhmmpv
# openblas@0.3.18%gcc@7.4.0~bignuma~consistent_fpcsr~ilp64+locking+pic+shared threads=none arch=linux-rhel7-power9le
module load exasgd-openblas/0.3.18/gcc-7.4.0-we6zwc7
# perl@5.34.0%gcc@7.4.0+cpanm+shared+threads arch=linux-rhel7-power9le
module load exasgd-perl/5.34.0/gcc-7.4.0-h45ivzd
# pkgconf@1.8.0%gcc@7.4.0 arch=linux-rhel7-power9le
module load exasgd-pkgconf/1.8.0/gcc-7.4.0-jfmmybn
# raja@0.14.0%gcc@7.4.0+cuda~examples~exercises~ipo+openmp~rocm+shared~tests amdgpu_target=none build_type=RelWithDebInfo cuda_arch=70 arch=linux-rhel7-power9le
module load exasgd-raja/0.14.0/cuda-11.0.194/gcc-7.4.0-nxnxc7i
# readline@8.1%gcc@7.4.0 arch=linux-rhel7-power9le
module load exasgd-readline/8.1/gcc-7.4.0-cszha3o
# suite-sparse@5.10.1%gcc@7.4.0~cuda~openmp+pic~tbb arch=linux-rhel7-power9le
module load exasgd-suite-sparse/5.10.1/gcc-7.4.0-fiwehvv
# texinfo@6.5%gcc@7.4.0 patches=12f6edb0c6b270b8c8dba2ce17998c580db01182d871ee32b7b6e4129bd1d23a,1732115f651cff98989cb0215d8f64da5e0f7911ebf0c13b064920f088f2ffe1 arch=linux-rhel7-power9le
module load exasgd-texinfo/6.5/gcc-7.4.0-do3jrb5
# umpire@6.0.0%gcc@7.4.0+c+cuda~deviceconst~examples~fortran~ipo~numa+openmp~rocm~shared amdgpu_target=none build_type=RelWithDebInfo cuda_arch=70 tests=none arch=linux-rhel7-power9le
module load exasgd-umpire/6.0.0/cuda-11.0.194/gcc-7.4.0-2suiwkb
# zlib@1.2.11%gcc@7.4.0+optimize+pic+shared arch=linux-rhel7-power9le
module load exasgd-zlib/1.2.11/gcc-7.4.0-vnk3szsLoad system modules

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
