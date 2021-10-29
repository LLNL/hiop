#  NOTE: The following is required when running from Gitlab CI via slurm job
source /etc/profile.d/modules.sh
module unload cmake
module unload metis
module unload cuda

module use -a /usr/share/Modules/modulefiles
module use -a /share/apps/modules/tools
module use -a /share/apps/modules/compilers
module use -a /share/apps/modules/mpi
module use -a /etc/modulefiles
module use -a /qfs/projects/exasgd/src/spack/share/spack/modules/linux-rhel7-power9le

# Load spack-built modules
# arpack-ng@3.8.0%gcc@7.4.0+mpi+shared arch=linux-rhel7-power9le
module load exasgd-arpack-ng/3.8.0/openmpi-3.1.5/gcc-7.4.0-rnctplf
# arpack-ng@3.8.0%gcc@7.4.0+mpi+shared arch=linux-rhel7-power9le
module load exasgd-arpack-ng/3.8.0/openmpi-3.1.5/gcc-7.4.0-sa735pe
# autoconf-archive@2019.01.06%gcc@7.4.0 arch=linux-rhel7-power9le
module load exasgd-autoconf-archive/2019.01.06/gcc-7.4.0-nn453cx
# blaspp@2021.04.01%gcc@7.4.0+cuda~ipo+openmp~rocm+shared amdgpu_target=none build_type=RelWithDebInfo cuda_arch=none arch=linux-rhel7-power9le
module load exasgd-blaspp/2021.04.01/cuda-10.2.89/gcc-7.4.0-pvfrdkh
# blt@0.4.1%gcc@7.4.0 arch=linux-rhel7-power9le
module load exasgd-blt/0.4.1/gcc-7.4.0-b72v2zy
# butterflypack@1.2.1%gcc@7.4.0~ipo+shared build_type=RelWithDebInfo arch=linux-rhel7-power9le
module load exasgd-butterflypack/1.2.1/openmpi-3.1.5/gcc-7.4.0-s3tv2as
# butterflypack@1.2.1%gcc@7.4.0~ipo+shared build_type=RelWithDebInfo arch=linux-rhel7-power9le
module load exasgd-butterflypack/1.2.1/openmpi-3.1.5/gcc-7.4.0-4piozn7
# camp@0.2.2%gcc@7.4.0+cuda~ipo~rocm~tests amdgpu_target=none build_type=RelWithDebInfo cuda_arch=70 arch=linux-rhel7-power9le
module load exasgd-camp/0.2.2/cuda-10.2.89/gcc-7.4.0-qosgvaj
# coinhsl@2015.06.23%gcc@7.4.0+blas arch=linux-rhel7-power9le
module load exasgd-coinhsl/2015.06.23/gcc-7.4.0-hacdh23
# cub@1.12.0-rc0%gcc@7.4.0 arch=linux-rhel7-power9le
module load exasgd-cub/1.12.0-rc0/gcc-7.4.0-iwyj63t
# cuda@10.2.89%gcc@7.4.0~allow-unsupported-compilers~dev arch=linux-rhel7-power9le
module load exasgd-cuda/10.2.89/gcc-7.4.0-doxxhum
# gmp@6.2.1%gcc@7.4.0 arch=linux-rhel7-power9le
module load exasgd-gmp/6.2.1/gcc-7.4.0-ur2a3rb
# gnuconfig@2021-08-14%gcc@7.4.0 arch=linux-rhel7-power9le
module load exasgd-gnuconfig/2021-08-14/gcc-7.4.0-qr6nxuq
# ipopt@3.12.10%gcc@7.4.0+coinhsl~debug~metis~mumps arch=linux-rhel7-power9le
module load exasgd-ipopt/3.12.10/gcc-7.4.0-z3unjvb
# lapackpp@2021.04.00%gcc@7.4.0~ipo+shared build_type=RelWithDebInfo arch=linux-rhel7-power9le
module load exasgd-lapackpp/2021.04.00/cuda-10.2.89/gcc-7.4.0-jaaknbt
# libtool@2.4.6%gcc@7.4.0 arch=linux-rhel7-power9le
module load exasgd-libtool/2.4.6/gcc-7.4.0-x5h54ly
# magma@2.6.1%gcc@7.4.0+cuda+fortran~ipo~rocm+shared amdgpu_target=none build_type=RelWithDebInfo cuda_arch=70 arch=linux-rhel7-power9le
module load exasgd-magma/2.6.1/cuda-10.2.89/gcc-7.4.0-ofwgvhp
# metis@5.1.0%gcc@7.4.0~gdb~int64~real64+shared build_type=Release patches=4991da938c1d3a1d3dea78e49bbebecba00273f98df2a656e38b83d55b281da1,b1225da886605ea558db7ac08dd8054742ea5afe5ed61ad4d0fe7a495b1270d2 arch=linux-rhel7-power9le
module load exasgd-metis/5.1.0/gcc-7.4.0-shhhyku
# mpfr@4.1.0%gcc@7.4.0 arch=linux-rhel7-power9le
module load exasgd-mpfr/4.1.0/gcc-7.4.0-33dvnf2
# netlib-scalapack@2.1.0%gcc@7.4.0~ipo~pic+shared build_type=Release patches=1c9ce5fee1451a08c2de3cc87f446aeda0b818ebbce4ad0d980ddf2f2a0b2dc4,f2baedde688ffe4c20943c334f580eb298e04d6f35c86b90a1f4e8cb7ae344a2 arch=linux-rhel7-power9le
module load exasgd-netlib-scalapack/2.1.0/openmpi-3.1.5/gcc-7.4.0-5ebdxj3
# netlib-scalapack@2.1.0%gcc@7.4.0~ipo~pic+shared build_type=Release patches=1c9ce5fee1451a08c2de3cc87f446aeda0b818ebbce4ad0d980ddf2f2a0b2dc4,f2baedde688ffe4c20943c334f580eb298e04d6f35c86b90a1f4e8cb7ae344a2 arch=linux-rhel7-power9le
module load exasgd-netlib-scalapack/2.1.0/openmpi-3.1.5/gcc-7.4.0-36yv2kh
# openblas@0.3.18%gcc@7.4.0~bignuma~consistent_fpcsr~ilp64+locking+pic+shared threads=openmp arch=linux-rhel7-power9le
module load exasgd-openblas/0.3.18/gcc-7.4.0-hd2vahd
# parmetis@4.0.3%gcc@7.4.0~gdb~int64~ipo+shared build_type=RelWithDebInfo patches=4f892531eb0a807eb1b82e683a416d3e35154a455274cf9b162fb02054d11a5b,50ed2081bc939269689789942067c58b3e522c269269a430d5d34c00edbc5870,704b84f7c7444d4372cb59cca6e1209df4ef3b033bc4ee3cf50f369bce972a9d arch=linux-rhel7-power9le
module load exasgd-parmetis/4.0.3/openmpi-3.1.5/gcc-7.4.0-oosko2x
# parmetis@4.0.3%gcc@7.4.0~gdb~int64~ipo+shared build_type=RelWithDebInfo patches=4f892531eb0a807eb1b82e683a416d3e35154a455274cf9b162fb02054d11a5b,50ed2081bc939269689789942067c58b3e522c269269a430d5d34c00edbc5870,704b84f7c7444d4372cb59cca6e1209df4ef3b033bc4ee3cf50f369bce972a9d arch=linux-rhel7-power9le
module load exasgd-parmetis/4.0.3/openmpi-3.1.5/gcc-7.4.0-gcj3cbb
# perl@5.26.0%gcc@7.4.0+cpanm~shared~threads patches=0eac10ed90aeb0459ad8851f88081d439a4e41978e586ec743069e8b059370ac,8cf4302ca8b480c60ccdcaa29ec53d9d50a71d4baf469ac8c6fca00ca31e58a2 arch=linux-rhel7-power9le
module load exasgd-perl/5.26.0/gcc-7.4.0-hbrcmhv
# pkgconf@1.8.0%gcc@7.4.0 arch=linux-rhel7-power9le
module load exasgd-pkgconf/1.8.0/gcc-7.4.0-jfmmybn
# raja@0.14.0%gcc@7.4.0+cuda~examples~exercises~ipo+openmp~rocm+shared~tests amdgpu_target=none build_type=RelWithDebInfo cuda_arch=70 arch=linux-rhel7-power9le
module load exasgd-raja/0.14.0/cuda-10.2.89/gcc-7.4.0-rg2znem
# slate@2021.05.02%gcc@7.4.0+cuda~ipo+mpi+openmp~rocm+shared amdgpu_target=none build_type=RelWithDebInfo cuda_arch=none arch=linux-rhel7-power9le
module load exasgd-slate/2021.05.02/cuda-10.2.89/openmpi-3.1.5/gcc-7.4.0-gafaynr
# slate@2021.05.02%gcc@7.4.0+cuda~ipo+mpi+openmp~rocm+shared amdgpu_target=none build_type=RelWithDebInfo cuda_arch=none arch=linux-rhel7-power9le
module load exasgd-slate/2021.05.02/cuda-10.2.89/openmpi-3.1.5/gcc-7.4.0-6y3sqsg
# strumpack@6.0.0%gcc@7.4.0~build_dev_tests~build_tests+butterflypack+c_interface~count_flops+cuda~ipo+mpi+openmp+parmetis~rocm~scotch+shared+slate~task_timers+zfp amdgpu_target=none build_type=RelWithDebInfo cuda_arch=none arch=linux-rhel7-power9le
module load exasgd-strumpack/6.0.0/cuda-10.2.89/openmpi-3.1.5/gcc-7.4.0-77wjito
# strumpack@6.0.0%gcc@7.4.0~build_dev_tests~build_tests+butterflypack+c_interface~count_flops+cuda~ipo+mpi+openmp+parmetis~rocm~scotch+shared+slate~task_timers+zfp amdgpu_target=none build_type=RelWithDebInfo cuda_arch=none arch=linux-rhel7-power9le
module load exasgd-strumpack/6.0.0/cuda-10.2.89/openmpi-3.1.5/gcc-7.4.0-cy6envw
# suite-sparse@5.8.1%gcc@7.4.0~cuda~openmp+pic~tbb arch=linux-rhel7-power9le
module load exasgd-suite-sparse/5.8.1/gcc-7.4.0-hrkcqf2
# texinfo@6.5%gcc@7.4.0 patches=12f6edb0c6b270b8c8dba2ce17998c580db01182d871ee32b7b6e4129bd1d23a,1732115f651cff98989cb0215d8f64da5e0f7911ebf0c13b064920f088f2ffe1 arch=linux-rhel7-power9le
module load exasgd-texinfo/6.5/gcc-7.4.0-l4ikmq2
# umpire@6.0.0%gcc@7.4.0+c+cuda~deviceconst~examples~fortran~ipo~numa+openmp~rocm~shared amdgpu_target=none build_type=RelWithDebInfo cuda_arch=70 tests=none arch=linux-rhel7-power9le
module load exasgd-umpire/6.0.0/cuda-10.2.89/gcc-7.4.0-6jzwj5y
# zfp@0.5.5%gcc@7.4.0~aligned~c~cuda~fasthash~fortran~ipo~openmp~profile~python+shared~strided~twoway bsws=64 build_type=RelWithDebInfo cuda_arch=none arch=linux-rhel7-power9le
module load exasgd-zfp/0.5.5/gcc-7.4.0-hiozbgg
module load exasgd-cmake/3.20.3/gcc-7.4.0-ya4c7w7

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
