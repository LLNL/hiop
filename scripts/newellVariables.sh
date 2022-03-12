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
module load autoconf-2.69-gcc-7.4.0-sdvbavp
# autoconf-archive@2019.01.06%gcc@7.4.0 arch=linux-rhel7-power9le
module load autoconf-archive-2019.01.06-gcc-7.4.0-nn453cx
# automake@1.16.5%gcc@7.4.0 arch=linux-rhel7-power9le
module load automake-1.16.5-gcc-7.4.0-j7nijx5
# blt@0.4.1%gcc@7.4.0 arch=linux-rhel7-power9le
module load blt-0.4.1-gcc-7.4.0-7zdbuph
# camp@0.2.2%gcc@7.4.0+cuda~ipo~rocm~tests build_type=RelWithDebInfo cuda_arch=70 arch=linux-rhel7-power9le
module load camp-0.2.2-gcc-7.4.0-hwdcxbh
# cmake@3.22.2%gcc@7.4.0~doc+ncurses+openssl+ownlibs~qt build_type=Release arch=linux-rhel7-power9le
module load cmake-3.22.2-gcc-7.4.0-akf7ts3
# coinhsl@2015.06.23%gcc@7.4.0+blas arch=linux-rhel7-power9le
module load coinhsl-2015.06.23-gcc-7.4.0-udagsad
# cub@1.12.0-rc0%gcc@7.4.0 arch=linux-rhel7-power9le
module load cub-1.12.0-rc0-gcc-7.4.0-iwyj63t
# cuda@10.2.89%gcc@7.4.0~allow-unsupported-compilers~dev arch=linux-rhel7-power9le
module load cuda-10.2.89-gcc-7.4.0-doxxhum
# gmp@6.2.1%gcc@7.4.0 arch=linux-rhel7-power9le
module load gmp-6.2.1-gcc-7.4.0-ur2a3rb
# gnuconfig@2021-08-14%gcc@7.4.0 arch=linux-rhel7-power9le
module load gnuconfig-2021-08-14-gcc-7.4.0-qr6nxuq
# libsigsegv@2.13%gcc@7.4.0 arch=linux-rhel7-power9le
module load libsigsegv-2.13-gcc-7.4.0-cbn4dja
# libtool@2.4.6%gcc@7.4.0 arch=linux-rhel7-power9le
module load libtool-2.4.6-gcc-7.4.0-x5h54ly
# m4@1.4.19%gcc@7.4.0+sigsegv patches=9dc5fbd,bfdffa7 arch=linux-rhel7-power9le
module load m4-1.4.19-gcc-7.4.0-nrrlksm
# magma@master%gcc@7.4.0+cuda+fortran~ipo~rocm+shared build_type=RelWithDebInfo cuda_arch=70 arch=linux-rhel7-power9le
module load magma-master-gcc-7.4.0-lplludi
# metis@5.1.0%gcc@7.4.0~gdb~int64~real64+shared build_type=Release patches=4991da9,b1225da arch=linux-rhel7-power9le
module load metis-5.1.0-gcc-7.4.0-shhhyku
# mpfr@4.1.0%gcc@7.4.0 arch=linux-rhel7-power9le
module load mpfr-4.1.0-gcc-7.4.0-33dvnf2
# ncurses@6.2%gcc@7.4.0~symlinks+termlib abi=none arch=linux-rhel7-power9le
module load ncurses-6.2-gcc-7.4.0-kqhmmpv
# openblas@0.3.19%gcc@7.4.0~bignuma~consistent_fpcsr~ilp64+locking+pic+shared symbol_suffix=none threads=none arch=linux-rhel7-power9le
module load openblas-0.3.19-gcc-7.4.0-w63lkax
# openmpi@3.1.5%gcc@7.4.0~atomics~cuda~cxx~cxx_exceptions+gpfs~internal-hwloc~java~legacylaunchers~lustre~memchecker~pmi~pmix+romio~singularity~sqlite3+static~thread_multiple+vt+wrapper-rpath fabrics=none schedulers=none arch=linux-rhel7-power9le
module load openmpi-3.1.5-gcc-7.4.0-z3zz5ak
# openssl@1.0.2k-fips%gcc@7.4.0~docs certs=system arch=linux-rhel7-power9le
module load openssl-1.0.2k-fips-gcc-7.4.0-na5ha7k
# perl@5.32.1%gcc@7.4.0+cpanm+shared+threads arch=linux-rhel7-power9le
module load perl-5.32.1-gcc-7.4.0-uqk33s3
# pkgconf@1.8.0%gcc@7.4.0 arch=linux-rhel7-power9le
module load pkgconf-1.8.0-gcc-7.4.0-jfmmybn
# raja@0.14.0%gcc@7.4.0+cuda+examples+exercises~ipo+openmp~rocm+shared~tests build_type=RelWithDebInfo cuda_arch=70 arch=linux-rhel7-power9le
module load raja-0.14.0-gcc-7.4.0-2zz4yu4
# suite-sparse@5.10.1%gcc@7.4.0~cuda~graphblas~openmp+pic~tbb arch=linux-rhel7-power9le
module load suite-sparse-5.10.1-gcc-7.4.0-btuc2bk
# texinfo@6.5%gcc@7.4.0 patches=12f6edb,1732115 arch=linux-rhel7-power9le
module load texinfo-6.5-gcc-7.4.0-2ae5zqm
# umpire@6.0.0%gcc@7.4.0+c+cuda~deviceconst+examples~fortran~ipo~numa~openmp~rocm~shared build_type=RelWithDebInfo cuda_arch=70 tests=none arch=linux-rhel7-power9le
module load umpire-6.0.0-gcc-7.4.0-vgjhvfa

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
