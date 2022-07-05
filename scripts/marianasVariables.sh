#  NOTE: The following is required when running from Gitlab CI via slurm job
source /etc/profile.d/modules.sh
module use -a /qfs/projects/exasgd/src/cameron-spack/share/spack/modules/linux-centos7-x86_64_v3


# Load spack-built modules
# autoconf@2.69%gcc@7.3.0 patches=35c4492,7793209,a49dd5b arch=linux-centos7-x86_64_v3
module load autoconf-2.69-gcc-7.3.0-gvh7nxv
# autoconf-archive@2022.02.11%gcc@7.3.0 patches=130cd48 arch=linux-centos7-x86_64_v3
module load autoconf-archive-2022.02.11-gcc-7.3.0-lrajcp3
# automake@1.16.5%gcc@7.3.0 arch=linux-centos7-x86_64_v3
module load automake-1.16.5-gcc-7.3.0-la5kvuy
# blt@0.4.1%gcc@7.3.0 arch=linux-centos7-x86_64_v3
module load blt-0.4.1-gcc-7.3.0-qeolwyb
# ca-certificates-mozilla@2022-03-29%gcc@7.3.0 arch=linux-centos7-x86_64_v3
module load ca-certificates-mozilla-2022-03-29-gcc-7.3.0-fjb4zc5
# camp@0.2.2%gcc@7.3.0+cuda~ipo~rocm~tests build_type=RelWithDebInfo cuda_arch=60 arch=linux-centos7-x86_64_v3
module load camp-0.2.2-gcc-7.3.0-ifdwyok
# cmake@3.23.2%gcc@7.3.0~doc+ncurses+ownlibs~qt build_type=Release arch=linux-centos7-x86_64_v3
module load cmake-3.23.2-gcc-7.3.0-riu7fla
# coinhsl@2015.06.23%gcc@7.3.0+blas arch=linux-centos7-x86_64_v3
module load coinhsl-2015.06.23-gcc-7.3.0-r42slsl
# cub@1.16.0%gcc@7.3.0 arch=linux-centos7-x86_64_v3
module load cub-1.16.0-gcc-7.3.0-4zaltzb
# ginkgo@glu%gcc@7.3.0+cuda~develtools~full_optimizations~hwloc~ipo~oneapi+openmp~rocm+shared build_type=Release cuda_arch=60 arch=linux-centos7-x86_64_v3
module load ginkgo-glu-gcc-7.3.0-63ouzce
# gmp@6.2.1%gcc@7.3.0 libs=shared,static arch=linux-centos7-x86_64_v3
module load gmp-6.2.1-gcc-7.3.0-if7iflm
# libsigsegv@2.13%gcc@7.3.0 arch=linux-centos7-x86_64_v3
module load libsigsegv-2.13-gcc-7.3.0-n653jc7
# libtool@2.4.7%gcc@7.3.0 arch=linux-centos7-x86_64_v3
module load libtool-2.4.7-gcc-7.3.0-atzgxc2
# m4@1.4.19%gcc@7.3.0+sigsegv patches=9dc5fbd,bfdffa7 arch=linux-centos7-x86_64_v3
module load m4-1.4.19-gcc-7.3.0-lcthdqt
# magma@2.6.2%gcc@7.3.0+cuda+fortran~ipo~rocm+shared build_type=RelWithDebInfo cuda_arch=60 arch=linux-centos7-x86_64_v3
module load magma-2.6.2-gcc-7.3.0-kqvdxay
# metis@5.1.0%gcc@7.3.0~gdb~int64~real64+shared build_type=Release patches=4991da9,b1225da arch=linux-centos7-x86_64_v3
module load metis-5.1.0-gcc-7.3.0-xfajh3x
# mpfr@4.1.0%gcc@7.3.0 libs=shared,static arch=linux-centos7-x86_64_v3
module load mpfr-4.1.0-gcc-7.3.0-zcatq2v
# ncurses@6.2%gcc@7.3.0~symlinks+termlib abi=none arch=linux-centos7-x86_64_v3
module load ncurses-6.2-gcc-7.3.0-sqnhgdg
# openblas@0.3.20%gcc@4.8.5~bignuma~consistent_fpcsr~ilp64+locking+pic+shared symbol_suffix=none threads=none arch=linux-centos7-x86_64_v3
module load openblas-0.3.20-gcc-4.8.5-rpolrqa
# openmpi@3.1.3%gcc@7.3.0~atomics~cuda~cxx~cxx_exceptions~gpfs~internal-hwloc~java~legacylaunchers~lustre~memchecker~pmi+romio+rsh~singularity+static+vt+wrapper-rpath fabrics=none schedulers=none arch=linux-centos7-x86_64_v3
module load openmpi-3.1.3-gcc-7.3.0-2ocdjy4
# openssl@1.1.1p%gcc@7.3.0~docs~shared certs=mozilla arch=linux-centos7-x86_64_v3
module load openssl-1.1.1p-gcc-7.3.0-tz3ln5w
# perl@5.26.0%gcc@7.3.0+cpanm+shared+threads patches=0eac10e,8cf4302 arch=linux-centos7-x86_64_v3
module load perl-5.26.0-gcc-7.3.0-f7w3oxq
# pkgconf@1.8.0%gcc@7.3.0 arch=linux-centos7-x86_64_v3
module load pkgconf-1.8.0-gcc-7.3.0-gxowfdy
# raja@0.14.0%gcc@7.3.0+cuda~examples~exercises~ipo+openmp~rocm+shared~tests build_type=RelWithDebInfo cuda_arch=60 arch=linux-centos7-x86_64_v3
module load raja-0.14.0-gcc-7.3.0-vtbmo6k
# suite-sparse@5.10.1%gcc@7.3.0~cuda~graphblas~openmp+pic~tbb arch=linux-centos7-x86_64_v3
module load suite-sparse-5.10.1-gcc-7.3.0-thhoxwy
# texinfo@6.5%gcc@7.3.0 patches=12f6edb,1732115 arch=linux-centos7-x86_64_v3
module load texinfo-6.5-gcc-7.3.0-crm3bgr
# umpire@6.0.0%gcc@7.3.0+c+cuda~device_alloc~deviceconst~examples~fortran~ipo~numa~openmp~rocm~shared build_type=RelWithDebInfo cuda_arch=60 tests=none arch=linux-centos7-x86_64_v3
module load umpire-6.0.0-gcc-7.3.0-z22n3zy
# zlib@1.2.12%gcc@7.3.0+optimize+pic+shared patches=0d38234 arch=linux-centos7-x86_64_v3
module load zlib-1.2.12-gcc-7.3.0-hq7ha7b

# Load system modules
module load gcc/7.3.0
module load cuda/11.4

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

export EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DCMAKE_CUDA_ARCHITECTURES=60"
export CMAKE_CACHE_SCRIPT=gcc-cuda.cmake
