module use -a /autofs/nccs-svm1_proj/csc359/installs/spack/share/spack/modules/linux-rhel7-power9le

module purge

# autoconf-archive@2019.01.06%gcc@7.4.0 arch=linux-rhel7-power9le
module load exasgd-autoconf-archive/2019.01.06/gcc-7.4.0-zr3h7p2
# coinhsl@2015.06.23%gcc@7.4.0+blas arch=linux-rhel7-power9le
module load exasgd-coinhsl/2015.06.23/gcc-7.4.0-k5ullmz
# gmp@6.2.1%gcc@7.4.0 arch=linux-rhel7-power9le
module load exasgd-gmp/6.2.1/gcc-7.4.0-6svtfvr
# hypre@2.20.0%gcc@7.4.0~complex~cuda~debug~int64~internal-superlu~mixedint+mpi~openmp+shared~superlu-dist~unified-memory cuda_arch=none patches=6e3336b1d62155f6350dfe42b0f9ea25d4fa0af60c7e540959139deb93a26059 arch=linux-rhel7-power9le
module load exasgd-hypre/2.20.0/openmpi-4.0.3/gcc-7.4.0-j2j6emj
# ipopt@3.12.10%gcc@7.4.0+coinhsl~debug~mumps arch=linux-rhel7-power9le
module load exasgd-ipopt/3.12.10/gcc-7.4.0-tj6jbm2
# magma@2.5.4%gcc@7.4.0+cuda+fortran~ipo+shared build_type=RelWithDebInfo cuda_arch=70 arch=linux-rhel7-power9le
module load exasgd-magma/2.5.4/cuda-10.2.89/gcc-7.4.0-mqzulie
# metis@5.1.0%gcc@7.4.0~gdb~int64~real64+shared build_type=Release patches=4991da938c1d3a1d3dea78e49bbebecba00273f98df2a656e38b83d55b281da1,b1225da886605ea558db7ac08dd8054742ea5afe5ed61ad4d0fe7a495b1270d2 arch=linux-rhel7-power9le
module load exasgd-metis/5.1.0/gcc-7.4.0-7cjo5kb
# mpfr@4.1.0%gcc@7.4.0 arch=linux-rhel7-power9le
module load exasgd-mpfr/4.1.0/gcc-7.4.0-bkusb67
# openblas@0.3.15%gcc@7.4.0~bignuma~consistent_fpcsr~ilp64+locking+pic+shared threads=none arch=linux-rhel7-power9le
module load exasgd-openblas/0.3.15/gcc-7.4.0-i4ax3dw
# parmetis@4.0.3%gcc@7.4.0~gdb~int64~ipo+shared build_type=RelWithDebInfo patches=4f892531eb0a807eb1b82e683a416d3e35154a455274cf9b162fb02054d11a5b,50ed2081bc939269689789942067c58b3e522c269269a430d5d34c00edbc5870,704b84f7c7444d4372cb59cca6e1209df4ef3b033bc4ee3cf50f369bce972a9d arch=linux-rhel7-power9le
module load exasgd-parmetis/4.0.3/openmpi-4.0.3/gcc-7.4.0-ziidpag
# perl@5.26.1%gcc@7.4.0~cpanm+shared+threads patches=0eac10ed90aeb0459ad8851f88081d439a4e41978e586ec743069e8b059370ac arch=linux-rhel7-power9le
module load exasgd-perl/5.26.1/gcc-7.4.0-dqxsxr4
# petsc@3.14.6%gcc@7.4.0~X~batch~cgns~complex~cuda~debug+double~exodusii~fftw~giflib+hdf5~hip+hypre~int64~jpeg~knl~libpng~libyaml~memkind+metis~mkl-pardiso~moab~mpfr+mpi~mumps~p4est~ptscotch~random123~saws+shared~suite-sparse+superlu-dist~trilinos~valgrind amdgpu_target=none clanguage=C arch=linux-rhel7-power9le
module load exasgd-petsc/3.14.6/openmpi-4.0.3/gcc-7.4.0-64pvrcl
# raja@0.12.1%gcc@7.4.0+cuda~examples~exercises~hip~ipo+openmp+shared~tests amdgpu_target=none build_type=RelWithDebInfo cuda_arch=70 arch=linux-rhel7-power9le
module load exasgd-raja/0.12.1/cuda-10.2.89/gcc-7.4.0-bautwa2
# suite-sparse@5.10.1%gcc@7.4.0~cuda~openmp+pic~tbb arch=linux-rhel7-power9le
module load exasgd-suite-sparse/5.10.1/gcc-7.4.0-q6quoed
# superlu-dist@6.4.0%gcc@7.4.0~cuda~int64~ipo~openmp+shared build_type=RelWithDebInfo cuda_arch=none arch=linux-rhel7-power9le
module load exasgd-superlu-dist/6.4.0/openmpi-4.0.3/gcc-7.4.0-p3u4ud4
# texinfo@6.5%gcc@7.4.0 patches=12f6edb0c6b270b8c8dba2ce17998c580db01182d871ee32b7b6e4129bd1d23a,1732115f651cff98989cb0215d8f64da5e0f7911ebf0c13b064920f088f2ffe1 arch=linux-rhel7-power9le
module load exasgd-texinfo/6.5/gcc-7.4.0-hvoep4u
# umpire@4.1.2%gcc@7.4.0+c+cuda~deviceconst~examples~fortran~hip~ipo~numa+openmp~shared amdgpu_target=none build_type=RelWithDebInfo cuda_arch=none patches=7d912d31cd293df005ba74cb96c6f3e32dc3d84afff49b14509714283693db08 tests=none arch=linux-rhel7-power9le
module load exasgd-umpire/4.1.2/cuda-10.2.89/gcc-7.4.0-6b24rqv
# zlib@1.2.11%gcc@7.4.0+optimize+pic+shared arch=linux-rhel7-power9le
module load exasgd-zlib/1.2.11/gcc-7.4.0-psrojaa

module load DefApps
module load gcc/7.4.0
module load cuda/10.2.89
module load m4/1.4.18
module load autoconf
module load hdf5
module load automake
module load zlib
module load pkgconf
module load readline
module load hwloc
module load python/3.7.0
module load bison
module load diffutils
module load cmake/3.20.2
module load openmpi/4.0.3

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

export EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DPETSC_DIR=$PETSC_DIR"
export CMAKE_PREFIX_PATH="/autofs/nccs-svm1_proj/csc359/installs/spack/opt/spack/linux-rhel7-power9le/gcc-7.4.0/coinhsl-2015.06.23-k5ullmza44iysb75syt4nbxjqc6jbpyf:/autofs/nccs-svm1_proj/csc359/installs/spack/opt/spack/linux-rhel7-power9le/gcc-7.4.0/gmp-6.2.1-6svtfvrcqf55ukqtwju65svdpjsctwnm:/autofs/nccs-svm1_proj/csc359/installs/spack/opt/spack/linux-rhel7-power9le/gcc-7.4.0/hiop-0.4.1-u64sho3aqofxck5qbneaqjlrkjrqpj3i:/autofs/nccs-svm1_proj/csc359/installs/spack/opt/spack/linux-rhel7-power9le/gcc-7.4.0/hypre-2.20.0-j2j6emjbcvbofu6gqnm55iswfk5vyo3n:/autofs/nccs-svm1_proj/csc359/installs/spack/opt/spack/linux-rhel7-power9le/gcc-7.4.0/ipopt-3.12.10-tj6jbm2o46vrqg432w6a7kow4s3ik4n5:/autofs/nccs-svm1_proj/csc359/installs/spack/opt/spack/linux-rhel7-power9le/gcc-7.4.0/magma-2.5.4-mqzulieztnvpykgogwr2vdqijtc4rg6l:/autofs/nccs-svm1_proj/csc359/installs/spack/opt/spack/linux-rhel7-power9le/gcc-7.4.0/metis-5.1.0-7cjo5kboozvz3nqiwixlib3rdv6moq6x:/autofs/nccs-svm1_proj/csc359/installs/spack/opt/spack/linux-rhel7-power9le/gcc-7.4.0/mpfr-4.1.0-bkusb67m2m5vks25nspbwxl4qjymewo4:/autofs/nccs-svm1_proj/csc359/installs/spack/opt/spack/linux-rhel7-power9le/gcc-7.4.0/openblas-0.3.15-i4ax3dwosg4tbe4a3zgyv6dq5woa6dmj:/autofs/nccs-svm1_proj/csc359/installs/spack/opt/spack/linux-rhel7-power9le/gcc-7.4.0/parmetis-4.0.3-ziidpag2e5nkycqaaoi5oik3sjhxew7x:/autofs/nccs-svm1_proj/csc359/installs/spack/opt/spack/linux-rhel7-power9le/gcc-7.4.0/petsc-3.14.6-64pvrclvkyntsq2oyuaajfjrzvdung7k:/autofs/nccs-svm1_proj/csc359/installs/spack/opt/spack/linux-rhel7-power9le/gcc-7.4.0/raja-0.12.1-bautwa2uhdioj5uqxetc224tavriaq4r:/autofs/nccs-svm1_proj/csc359/installs/spack/opt/spack/linux-rhel7-power9le/gcc-7.4.0/suite-sparse-5.10.1-q6quoedisfvmibgpynwu4o4z4biwxdlx:/autofs/nccs-svm1_proj/csc359/installs/spack/opt/spack/linux-rhel7-power9le/gcc-7.4.0/superlu-dist-6.4.0-p3u4ud4omgtpmhonst6bsjpaggmj6sy3:/autofs/nccs-svm1_proj/csc359/installs/spack/opt/spack/linux-rhel7-power9le/gcc-7.4.0/umpire-4.1.2-6b24rqvw3wyaiayttjcztzl3t6mlrzw7:/autofs/nccs-svm1_sw/summit/.swci/0-core/opt/spack/20180914/linux-rhel7-ppc64le/gcc-4.8.5/cmake-3.20.2-24ualfzy6em6ws5sbiu7rlgcuionodrm:/sw/summit/cuda/10.2.89:/autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/gcc-7.4.0/hdf5-1.10.4-ak37esaqunyb3s77xh3vbhpfjou5y3pz:/autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/gcc-7.4.0/openmpi-4.0.3-hf3xf5weredkclm4lkxadtizibv7z6i6:/autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/gcc-7.4.0/zlib-1.2.11-tdykbkiueylpgx2rshpms3k3ncw5g3f6:$CMAKE_PREFIX_PATH"

export CC=/sw/summit/gcc/7.4.0/bin/gcc CXX=/sw/summit/gcc/7.4.0/bin/g++ FC=/sw/summit/gcc/7.4.0/bin/gfortran
