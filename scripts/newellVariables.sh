#  NOTE: The following is required when running from Gitlab CI via slurm job
source /etc/profile.d/modules.sh
module use -a /usr/share/Modules/modulefiles
module use -a /share/apps/modules/tools
module use -a /share/apps/modules/compilers
module use -a /share/apps/modules/mpi
module use -a /etc/modulefiles
module use -a /qfs/projects/exasgd/src/spack/share/spack/modules/linux-rhel7-power9le

# Load spack-built modules
# autoconf@2.69%gcc@7.4.0 patches=35c4492,7793209,a49dd5b arch=linux-rhel7-power9le
module load exasgd-autoconf/2.69/gcc-7.4.0-sdvbavp
# autoconf-archive@2019.01.06%gcc@7.4.0 arch=linux-rhel7-power9le
module load exasgd-autoconf-archive/2019.01.06/gcc-7.4.0-nn453cx
# automake@1.16.5%gcc@7.4.0 arch=linux-rhel7-power9le
module load exasgd-automake/1.16.5/gcc-7.4.0-j7nijx5
# blt@0.4.1%gcc@7.4.0 arch=linux-rhel7-power9le
module load exasgd-blt/0.4.1/gcc-7.4.0-quqlodz
# camp@0.2.2%gcc@7.4.0+cuda~ipo~rocm~tests build_type=RelWithDebInfo cuda_arch=70 arch=linux-rhel7-power9le
module load exasgd-camp/0.2.2/cuda-10.2.89/gcc-7.4.0-kcqbrqo
# cmake@3.23.1%gcc@7.4.0~doc+ncurses+ownlibs~qt build_type=Release arch=linux-rhel7-power9le
module load exasgd-cmake/3.23.1/gcc-7.4.0-tfl7wm4
# coinhsl@2015.06.23%gcc@7.4.0+blas arch=linux-rhel7-power9le
module load exasgd-coinhsl/2015.06.23/gcc-7.4.0-ts5vjfq
# cub@1.12.0%gcc@7.4.0 arch=linux-rhel7-power9le
module load exasgd-cub/1.12.0/gcc-7.4.0-4qyvoqn
# cuda@10.2.89%gcc@7.4.0~allow-unsupported-compilers~dev arch=linux-rhel7-power9le
module load exasgd-cuda/10.2.89/gcc-7.4.0-doxxhum
# ginkgo@glu%gcc@7.4.0+cuda~develtools~full_optimizations~hwloc~ipo~oneapi+openmp~rocm+shared build_type=Release cuda_arch=70 arch=linux-rhel7-power9le
module load exasgd-ginkgo/glu/cuda-10.2.89/gcc-7.4.0-f4fyoqn
# gmp@6.2.1%gcc@7.4.0 libs=shared,static arch=linux-rhel7-power9le
module load exasgd-gmp/6.2.1/gcc-7.4.0-oea2aet
# gnuconfig@2021-08-14%gcc@7.4.0 arch=linux-rhel7-power9le
module load exasgd-gnuconfig/2021-08-14/gcc-7.4.0-qr6nxuq
# libsigsegv@2.13%gcc@7.4.0 arch=linux-rhel7-power9le
module load exasgd-libsigsegv/2.13/gcc-7.4.0-cbn4dja
# libtool@2.4.7%gcc@7.4.0 arch=linux-rhel7-power9le
module load exasgd-libtool/2.4.7/gcc-7.4.0-p5juddc
# m4@1.4.19%gcc@7.4.0+sigsegv patches=9dc5fbd,bfdffa7 arch=linux-rhel7-power9le
module load exasgd-m4/1.4.19/gcc-7.4.0-nrrlksm
# magma@2.6.2%gcc@7.4.0+cuda+fortran~ipo~rocm+shared build_type=RelWithDebInfo cuda_arch=70 arch=linux-rhel7-power9le
module load exasgd-magma/2.6.2/cuda-10.2.89/gcc-7.4.0-f6uzlaf
# metis@5.1.0%gcc@7.4.0~gdb~int64~real64+shared build_type=Release patches=4991da9,b1225da arch=linux-rhel7-power9le
module load exasgd-metis/5.1.0/gcc-7.4.0-shhhyku
# mpfr@4.1.0%gcc@7.4.0 libs=shared,static arch=linux-rhel7-power9le
module load exasgd-mpfr/4.1.0/gcc-7.4.0-tz5esun
# ncurses@6.2%gcc@7.4.0~symlinks+termlib abi=none arch=linux-rhel7-power9le
module load exasgd-ncurses/6.2/gcc-7.4.0-kqhmmpv
# openblas@0.3.20%gcc@7.4.0~bignuma~consistent_fpcsr~ilp64+locking+pic+shared symbol_suffix=none threads=none arch=linux-rhel7-power9le
module load exasgd-openblas/0.3.20/gcc-7.4.0-3zdqw2i
# openmpi@3.1.5%gcc@7.4.0~atomics~cuda~cxx~cxx_exceptions~gpfs~internal-hwloc~java~legacylaunchers~lustre~memchecker~pmi~pmix+romio~singularity~sqlite3+static~thread_multiple+vt+wrapper-rpath fabrics=none schedulers=none arch=linux-rhel7-power9le
module load exasgd-openmpi/3.1.5/gcc-7.4.0-vp37g7m
# openssl@1.0.2k-fips%gcc@7.4.0~docs~shared certs=system arch=linux-rhel7-power9le
module load exasgd-openssl/1.0.2k-fips/gcc-7.4.0-5q6g7rp
# perl@5.32.1%gcc@7.4.0+cpanm+shared+threads arch=linux-rhel7-power9le
module load exasgd-perl/5.32.1/gcc-7.4.0-uqk33s3
# pkgconf@1.8.0%gcc@7.4.0 arch=linux-rhel7-power9le
module load exasgd-pkgconf/1.8.0/gcc-7.4.0-jfmmybn
# raja@0.14.0%gcc@7.4.0+cuda~examples~exercises~ipo+openmp~rocm+shared~tests build_type=RelWithDebInfo cuda_arch=70 arch=linux-rhel7-power9le
module load exasgd-raja/0.14.0/cuda-10.2.89/gcc-7.4.0-a5s47ej
# suite-sparse@5.10.1%gcc@7.4.0~cuda~graphblas~openmp+pic~tbb arch=linux-rhel7-power9le
module load exasgd-suite-sparse/5.10.1/gcc-7.4.0-e5qockg
# texinfo@6.5%gcc@7.4.0 patches=12f6edb,1732115 arch=linux-rhel7-power9le
module load exasgd-texinfo/6.5/gcc-7.4.0-2ae5zqm
# umpire@6.0.0%gcc@7.4.0+c+cuda~deviceconst+examples~fortran~ipo~numa~openmp~rocm~shared build_type=RelWithDebInfo cuda_arch=70 tests=none arch=linux-rhel7-power9le
module load exasgd-umpire/6.0.0/cuda-10.2.89/gcc-7.4.0-ybatfie

# Load system modules
module load gcc/7.4.0
module load cuda/10.2
module load openmpi/3.1.5

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
