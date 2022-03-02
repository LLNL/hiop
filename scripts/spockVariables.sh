module purge

module use -a /autofs/nccs-svm1_proj/csc359/src/spack/bin/../share/spack/modules/cray-sles15-zen2

# Spack modules

# blt@0.4.1%clang@13.0.0 arch=cray-sles15-zen2
module load blt-0.4.1-clang-13.0.0-xq3kaj3
# camp@0.2.2%clang@13.0.0~cuda~ipo+rocm~tests amdgpu_target=gfx908 build_type=RelWithDebInfo arch=cray-sles15-zen2
module load camp-0.2.2-clang-13.0.0-kkgc7ll
# coinhsl@2015.06.23%clang@13.0.0+blas arch=cray-sles15-zen2
module load coinhsl-2015.06.23-clang-13.0.0-bju7nmm
# magma@master%clang@13.0.0~cuda+fortran~ipo+rocm+shared amdgpu_target=gfx908 build_type=RelWithDebInfo dev_path=/autofs/nccs-svm1_proj/csc359/src/magma arch=cray-sles15-zen2
module load magma-master-clang-13.0.0-22qzkzt
# metis@5.1.0%clang@13.0.0~gdb~int64~real64+shared build_type=Release patches=4991da9 arch=cray-sles15-zen2
module load metis-5.1.0-clang-13.0.0-caz2tmn
# openblas@0.3.19%clang@13.0.0~bignuma~consistent_fpcsr~ilp64+locking+pic+shared symbol_suffix=none threads=none arch=cray-sles15-zen2
module load openblas-0.3.19-clang-13.0.0-g4htdnf
# raja@0.14.0%clang@13.0.0~cuda~examples~exercises~ipo~openmp+rocm+shared~tests amdgpu_target=gfx908 build_type=RelWithDebInfo arch=cray-sles15-zen2
module load raja-0.14.0-clang-13.0.0-rr3dsjb
# umpire@6.0.0%clang@13.0.0+c~cuda~deviceconst~examples~fortran~ipo~numa~openmp+rocm+shared amdgpu_target=gfx908 build_type=RelWithDebInfo tests=none arch=cray-sles15-zen2
module load umpire-6.0.0-clang-13.0.0-cspfkuw

# Spack wrapper module for cray-mpich/8.1.12
module load cray-mpich-8.1.12-clang-13.0.0-b443otg

# System modules

module load PrgEnv-cray/8.2.0
module load cce/13.0.0
module load gcc/11.2.0
module load rocm/4.5.0
module load libfabric/1.11.0.4.75
module load cmake/3.22.1

export EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DCMAKE_INSTALL_RPATH:STRING=\"/autofs/nccs-svm1_proj/csc359/src/spack/opt/spack/cray-sles15-zen2/clang-13.0.0/hiop-develop-tnkmqb2mtrxvwgslvpn52o35zm467cje/lib;/autofs/nccs-svm1_proj/csc359/src/spack/opt/spack/cray-sles15-zen2/clang-13.0.0/hiop-develop-tnkmqb2mtrxvwgslvpn52o35zm467cje/lib64;/autofs/nccs-svm1_proj/csc359/src/spack/opt/spack/cray-sles15-zen2/clang-13.0.0/coinhsl-2015.06.23-bju7nmmqm3pjjgsh7wejcy2aol5xemkf/lib;/autofs/nccs-svm1_proj/csc359/src/spack/opt/spack/cray-sles15-zen2/clang-13.0.0/openblas-0.3.19-g4htdnf53xhfj7zldmxkb5mqzpnlwq5q/lib;/opt/cray/pe/mpich/8.1.12/ofi/gnu/9.1/lib;/opt/rocm-4.5.0/hip/lib;/opt/rocm-4.5.0/hipblas/lib;/opt/rocm-4.5.0/hipsparse/lib;/opt/rocm-4.5.0/lib;/autofs/nccs-svm1_proj/csc359/src/spack/opt/spack/cray-sles15-zen2/clang-13.0.0/magma-master-22qzkztjyy6ktxdgvkbeio7zmvdphhhv/lib;/autofs/nccs-svm1_proj/csc359/src/spack/opt/spack/cray-sles15-zen2/clang-13.0.0/metis-5.1.0-caz2tmn2cbmt7bjfpdudtbuflbac6akq/lib;/autofs/nccs-svm1_proj/csc359/src/spack/opt/spack/cray-sles15-zen2/clang-13.0.0/raja-0.14.0-rr3dsjbdwdwnrhewf4co5p4y2ki54ala/lib;/autofs/nccs-svm1_proj/csc359/src/spack/opt/spack/cray-sles15-zen2/clang-13.0.0/camp-0.2.2-kkgc7lle5efwab75hj3xtg7sqnurw6an/lib;/autofs/nccs-svm1_proj/csc359/src/spack/opt/spack/cray-sles15-zen2/clang-13.0.0/umpire-6.0.0-cspfkuwu37jjfsn4p627n7tbra5pbtcw/lib;/opt/rocm-4.5.0/lib64;/opt/rocm-4.5.0\""
export EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DCMAKE_PREFIX_PATH:STRING=\"/autofs/nccs-svm1_proj/csc359/src/spack/opt/spack/cray-sles15-zen2/clang-13.0.0/umpire-6.0.0-cspfkuwu37jjfsn4p627n7tbra5pbtcw;/autofs/nccs-svm1_proj/csc359/src/spack/opt/spack/cray-sles15-zen2/clang-13.0.0/raja-0.14.0-rr3dsjbdwdwnrhewf4co5p4y2ki54ala;/autofs/nccs-svm1_proj/csc359/src/spack/opt/spack/cray-sles15-zen2/clang-13.0.0/camp-0.2.2-kkgc7lle5efwab75hj3xtg7sqnurw6an;/autofs/nccs-svm1_proj/csc359/src/spack/opt/spack/cray-sles15-zen2/clang-13.0.0/blt-0.4.1-xq3kaj3f6t53upqdcev5l3lkfi6ed34z;/autofs/nccs-svm1_proj/csc359/src/spack/opt/spack/cray-sles15-zen2/clang-13.0.0/metis-5.1.0-caz2tmn2cbmt7bjfpdudtbuflbac6akq;/autofs/nccs-svm1_proj/csc359/src/spack/opt/spack/cray-sles15-zen2/clang-13.0.0/magma-master-22qzkztjyy6ktxdgvkbeio7zmvdphhhv;/autofs/nccs-svm1_proj/csc359/src/spack/opt/spack/cray-sles15-zen2/clang-13.0.0/coinhsl-2015.06.23-bju7nmmqm3pjjgsh7wejcy2aol5xemkf;/autofs/nccs-svm1_proj/csc359/src/spack/opt/spack/cray-sles15-zen2/clang-13.0.0/openblas-0.3.19-g4htdnf53xhfj7zldmxkb5mqzpnlwq5q;/opt/rocm-4.5.0;/opt/rocm-4.5.0;/opt/rocm-4.5.0/hipsparse;/opt/rocm-4.5.0/hipblas;/opt/rocm-4.5.0/hip;/opt/cray/pe/mpich/8.1.12/ofi/gnu/9.1;/sw/spock/spack-envs/base/opt/linux-sles15-x86_64/gcc-7.5.0/cmake-3.22.1-wd2ptgormfxhkw3fiac75fiznbzcvpah\""
export EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DHIOP_BUILD_STATIC:BOOL=ON"
export EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DLAPACK_FOUND:BOOL=ON"
export EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DLAPACK_LIBRARIES:STRING=/autofs/nccs-svm1_proj/csc359/src/spack/opt/spack/cray-sles15-zen2/clang-13.0.0/openblas-0.3.19-g4htdnf53xhfj7zldmxkb5mqzpnlwq5q/lib/libopenblas.so"
export EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DHIOP_WITH_KRON_REDUCTION:BOOL=OFF"
export EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DHIOP_SPARSE:BOOL=ON"
export EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DHIOP_USE_COINHSL:BOOL=ON"
export EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DHIOP_TEST_WITH_BSUB:BOOL=OFF"
export EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DMPI_HOME:STRING=/opt/cray/pe/mpich/8.1.12/ofi/gnu/9.1"
export EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DMPI_C_COMPILER:STRING=/opt/cray/pe/mpich/8.1.12/ofi/gnu/9.1/bin/mpicc"
export EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DMPI_CXX_COMPILER:STRING=/opt/cray/pe/mpich/8.1.12/ofi/gnu/9.1/bin/mpicxx"
export EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DMPI_Fortran_COMPILER:STRING=/opt/cray/pe/mpich/8.1.12/ofi/gnu/9.1/bin/mpif90"
export EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DCMAKE_Fortran_FLAGS:STRING=\"-L/opt/cray/pe/gcc/11.2.0/snos/lib64 -lgfortran\""
export EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DHIOP_CTEST_LAUNCH_COMMAND:STRING=\"srun -A CSC359 -N 1 --exclusive -t 10:00 -p ecp\""
export EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DHIOP_COINHSL_DIR:STRING=/autofs/nccs-svm1_proj/csc359/src/spack/opt/spack/cray-sles15-zen2/clang-13.0.0/coinhsl-2015.06.23-bju7nmmqm3pjjgsh7wejcy2aol5xemkf"
export CMAKE_CACHE_SCRIPT=clang-hip.cmake

export CC=/opt/rocm-4.5.0/llvm/bin/clang
export CXX=/opt/rocm-4.5.0/llvm/bin/clang++
export FC=/opt/cray/pe/gcc/11.2.0/bin/gfortran
