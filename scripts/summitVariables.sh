export MY_CLUSTER=summit
export PROJ_DIR=/autofs/nccs-svm1_proj/csc359

module use -a /autofs/nccs-svm1_proj/csc359/installs/spack/share/spack/modules/linux-rhel7-power9le

module purge

# autoconf-archive@2019.01.06%gcc@7.4.0 arch=linux-rhel7-power9le
module load exasgd-autoconf-archive/2019.01.06/gcc-7.4.0-zr3h7p2
# berkeley-db@18.1.40%gcc@7.4.0+cxx~docs+stl patches=b231fcc4d5cff05e5c3a4814f6a5af0e9a966428dc2176540d2c05aff41de522 arch=linux-rhel7-power9le
module load exasgd-berkeley-db/18.1.40/gcc-7.4.0-hn7z7sb
# coinhsl@2015.06.23%gcc@7.4.0+blas arch=linux-rhel7-power9le
module load exasgd-coinhsl/2015.06.23/gcc-7.4.0-k5ullmz
# gdbm@1.19%gcc@7.4.0 arch=linux-rhel7-power9le
module load exasgd-gdbm/1.19/gcc-7.4.0-3iuzy53
# gmp@6.2.1%gcc@7.4.0 arch=linux-rhel7-power9le
module load exasgd-gmp/6.2.1/gcc-7.4.0-6svtfvr
# magma@2.5.4%gcc@7.4.0+cuda+fortran~ipo+shared build_type=RelWithDebInfo cuda_arch=70 arch=linux-rhel7-power9le
module load exasgd-magma/2.5.4/cuda-10.2.89/gcc-7.4.0-mqzulie
# metis@5.1.0%gcc@7.4.0~gdb~int64~real64+shared build_type=Release patches=4991da938c1d3a1d3dea78e49bbebecba00273f98df2a656e38b83d55b281da1,b1225da886605ea558db7ac08dd8054742ea5afe5ed61ad4d0fe7a495b1270d2 arch=linux-rhel7-power9le
module load exasgd-metis/5.1.0/gcc-7.4.0-7cjo5kb
# mpfr@4.1.0%gcc@7.4.0 arch=linux-rhel7-power9le
module load exasgd-mpfr/4.1.0/gcc-7.4.0-bkusb67
# openblas@0.3.15%gcc@7.4.0~bignuma~consistent_fpcsr~ilp64+locking+pic+shared threads=none arch=linux-rhel7-power9le
module load exasgd-openblas/0.3.15/gcc-7.4.0-i4ax3dw
# parmetis@4.0.3%gcc@7.4.0~gdb~int64~ipo+shared build_type=RelWithDebInfo patches=4f892531eb0a807eb1b82e683a416d3e35154a455274cf9b162fb02054d11a5b,50ed2081bc939269689789942067c58b3e522c269269a430d5d34c00edbc5870,704b84f7c7444d4372cb59cca6e1209df4ef3b033bc4ee3cf50f369bce972a9d arch=linux-rhel7-power9le
module load exasgd-parmetis/4.0.3/spectrum-mpi-10.3.1.2-20200121/gcc-7.4.0-smlgxxh
# perl@5.34.0%gcc@7.4.0+cpanm+shared+threads arch=linux-rhel7-power9le
module load exasgd-perl/5.34.0/gcc-7.4.0-bhdgtvn
# raja@0.12.1%gcc@7.4.0+cuda~examples~exercises~hip~ipo+openmp+shared~tests amdgpu_target=none build_type=RelWithDebInfo cuda_arch=70 arch=linux-rhel7-power9le
module load exasgd-raja/0.12.1/cuda-10.2.89/gcc-7.4.0-bautwa2
# readline@7.0%gcc@7.4.0 arch=linux-rhel7-power9le
module load exasgd-readline/7.0/gcc-7.4.0-zsnthku
# suite-sparse@5.10.1%gcc@7.4.0~cuda~openmp+pic~tbb arch=linux-rhel7-power9le
module load exasgd-suite-sparse/5.10.1/gcc-7.4.0-q6quoed
# superlu-dist@6.4.0%gcc@7.4.0~cuda~int64~ipo~openmp+shared build_type=RelWithDebInfo cuda_arch=none arch=linux-rhel7-power9le
module load exasgd-superlu-dist/6.4.0/spectrum-mpi-10.3.1.2-20200121/gcc-7.4.0-yifcfo2
# texinfo@6.5%gcc@7.4.0 patches=12f6edb0c6b270b8c8dba2ce17998c580db01182d871ee32b7b6e4129bd1d23a,1732115f651cff98989cb0215d8f64da5e0f7911ebf0c13b064920f088f2ffe1 arch=linux-rhel7-power9le
module load exasgd-texinfo/6.5/gcc-7.4.0-475ptss
# umpire@4.1.2%gcc@7.4.0+c+cuda~deviceconst~examples~fortran~hip~ipo~numa+openmp~shared amdgpu_target=none build_type=RelWithDebInfo cuda_arch=none patches=7d912d31cd293df005ba74cb96c6f3e32dc3d84afff49b14509714283693db08 tests=none arch=linux-rhel7-power9le
module load exasgd-umpire/4.1.2/cuda-10.2.89/gcc-7.4.0-6b24rqv

# Spack-generated module for exago
# exago@develop%gcc@7.4.0+cuda+gpu+hiop~hip~ipo+ipopt+mpi+petsc+raja amdgpu_target=none build_type=Release cuda_arch=70 arch=linux-rhel7-power9le
# module load exasgd-exago/develop/cuda-10.2.89/spectrum-mpi-10.3.1.2-20200121/gcc-7.4.0-gwm2qah

# System modules
module load DefApps
module load gcc/7.4.0
module load spectrum-mpi/10.3.1.2-20200121
module load cuda/10.2.89
module load m4/1.4.18
module load autoconf
module load hdf5
module load automake
module load zlib
module load pkgconf
module load readline
module load hwloc
module load bison
module load diffutils
module load cmake/3.20.2

export CC=/sw/summit/gcc/7.4.0/bin/gcc CXX=/sw/summit/gcc/7.4.0/bin/g++

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
