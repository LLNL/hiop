#  NOTE: The following is required when running from Gitlab CI via slurm job
source /etc/profile.d/modules.sh
module use -a /usr/share/Modules/modulefiles
module use -a /share/apps/modules/tools
module use -a /share/apps/modules/compilers
module use -a /share/apps/modules/mpi
module use -a /etc/modulefiles
module use -a /qfs/projects/exasgd/src/jaelyn-spack/spack/share/spack/modules/linux-centos8-power9le

# Load spack-built modules
# autoconf@2.69%gcc@8.5.0 patches=35c4492,7793209,a49dd5b arch=linux-centos8-power9le
module load autoconf-2.69-gcc-8.5.0-2mzbyqj
# autoconf-archive@2022.02.11%gcc@8.5.0 patches=130cd48 arch=linux-centos8-power9le
module load autoconf-archive-2022.02.11-gcc-8.5.0-nolgalj
# automake@1.16.5%gcc@8.5.0 arch=linux-centos8-power9le
module load automake-1.16.5-gcc-8.5.0-pnnvoal
# berkeley-db@18.1.40%gcc@8.5.0+cxx~docs+stl patches=b231fcc arch=linux-centos8-power9le
module load berkeley-db-18.1.40-gcc-8.5.0-cuzn6qn
# blt@0.5.1%gcc@8.5.0 arch=linux-centos8-power9le
module load blt-0.5.1-gcc-8.5.0-efxzffy
# bzip2@1.0.8%gcc@8.5.0~debug~pic+shared arch=linux-centos8-power9le
module load bzip2-1.0.8-gcc-8.5.0-tsweuon
# ca-certificates-mozilla@2022-03-29%gcc@8.5.0 arch=linux-centos8-power9le
module load ca-certificates-mozilla-2022-03-29-gcc-8.5.0-zyzfhdf
# camp@2022.03.0%gcc@8.5.0+cuda~ipo~rocm~tests build_type=RelWithDebInfo cuda_arch=70 arch=linux-centos8-power9le
module load camp-2022.03.0-gcc-8.5.0-ijcsmol
# cmake@3.23.2%gcc@8.5.0~doc+ncurses+ownlibs~qt build_type=Release arch=linux-centos8-power9le
module load cmake-3.23.2-gcc-8.5.0-pr3l2mn
# coinhsl@2015.06.23%gcc@8.5.0+blas arch=linux-centos8-power9le
module load coinhsl-2015.06.23-gcc-8.5.0-f6ka4rc
# cub@1.16.0%gcc@8.5.0 arch=linux-centos8-power9le
module load cub-1.16.0-gcc-8.5.0-p3cnthb
# diffutils@3.8%gcc@8.5.0 arch=linux-centos8-power9le
module load diffutils-3.8-gcc-8.5.0-ppyuisg
# gdbm@1.19%gcc@8.5.0 arch=linux-centos8-power9le
module load gdbm-1.19-gcc-8.5.0-unfo3x4
# ginkgo@glu%gcc@8.5.0+cuda~develtools~full_optimizations~hwloc~ipo~oneapi+openmp~rocm+shared build_type=Release cuda_arch=70 arch=linux-centos8-power9le
module load ginkgo-glu-gcc-8.5.0-xqxxanf
# gmp@6.2.1%gcc@8.5.0 libs=shared,static arch=linux-centos8-power9le
module load gmp-6.2.1-gcc-8.5.0-xlcuuht
# gnuconfig@2021-08-14%gcc@8.5.0 arch=linux-centos8-power9le
module load gnuconfig-2021-08-14-gcc-8.5.0-qjyg7ls
# libiconv@1.16%gcc@8.5.0 libs=shared,static arch=linux-centos8-power9le
module load libiconv-1.16-gcc-8.5.0-qqwmnok
# libsigsegv@2.13%gcc@8.5.0 arch=linux-centos8-power9le
module load libsigsegv-2.13-gcc-8.5.0-pa77xit
# libtool@2.4.7%gcc@8.5.0 arch=linux-centos8-power9le
module load libtool-2.4.7-gcc-8.5.0-kxdso3c
# m4@1.4.19%gcc@8.5.0+sigsegv patches=9dc5fbd,bfdffa7 arch=linux-centos8-power9le
module load m4-1.4.19-gcc-8.5.0-untfsqf
# magma@2.6.2%gcc@8.5.0+cuda+fortran~ipo~rocm+shared build_type=RelWithDebInfo cuda_arch=70 arch=linux-centos8-power9le
module load magma-2.6.2-gcc-8.5.0-ee3572c
# metis@5.1.0%gcc@8.5.0~gdb~int64~real64+shared build_type=Release patches=4991da9,b1225da arch=linux-centos8-power9le
module load metis-5.1.0-gcc-8.5.0-ldsei63
# mpfr@4.1.0%gcc@8.5.0 libs=shared,static arch=linux-centos8-power9le
module load mpfr-4.1.0-gcc-8.5.0-esdxmf2
# ncurses@6.2%gcc@8.5.0~symlinks+termlib abi=none arch=linux-centos8-power9le
module load ncurses-6.2-gcc-8.5.0-v24hmxo
# openblas@0.3.20%gcc@8.5.0~bignuma~consistent_fpcsr~ilp64+locking+pic+shared symbol_suffix=none threads=none arch=linux-centos8-power9le
module load openblas-0.3.20-gcc-8.5.0-rwstn2s
# perl@5.34.1%gcc@8.5.0+cpanm+shared+threads arch=linux-centos8-power9le
module load perl-5.34.1-gcc-8.5.0-fn534xj
# pkgconf@1.8.0%gcc@8.5.0 arch=linux-centos8-power9le
module load pkgconf-1.8.0-gcc-8.5.0-imrnro2
# raja@2022.03.0%gcc@8.5.0+cuda+disable_blt_export+examples+exercises~ipo+openmp~rocm+shared~tests build_type=RelWithDebInfo cuda_arch=70 arch=linux-centos8-power9le
module load raja-2022.03.0-gcc-8.5.0-f6ruftz
# readline@8.1.2%gcc@8.5.0 arch=linux-centos8-power9le
module load readline-8.1.2-gcc-8.5.0-l4hzlyf
# suite-sparse@5.10.1%gcc@8.5.0~cuda~graphblas~openmp+pic~tbb arch=linux-centos8-power9le
module load suite-sparse-5.10.1-gcc-8.5.0-ykffgpl
# texinfo@6.5%gcc@8.5.0 patches=12f6edb,1732115 arch=linux-centos8-power9le
module load texinfo-6.5-gcc-8.5.0-fvxyl2q
# umpire@2022.03.1%gcc@8.5.0+c+cuda+device_alloc~deviceconst+examples~fortran~ipo~numa~openmp~rocm~shared build_type=RelWithDebInfo cuda_arch=70 tests=none arch=linux-centos8-power9le
module load umpire-2022.03.1-gcc-8.5.0-aef4vlb
# zlib@1.2.12%gcc@8.5.0+optimize+pic+shared patches=0d38234 arch=linux-centos8-power9le
module load zlib-1.2.12-gcc-8.5.0-spb5k73

# Load system modules
module load gcc/8.5.0
module load cuda/11.4
module load openmpi/4.1.4

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

export EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DCMAKE_CUDA_ARCHITECTURES=70"
export CMAKE_CACHE_SCRIPT=gcc-cuda.cmake
