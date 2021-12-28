// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory (LLNL).
// LLNL-CODE-742473. All rights reserved.
//
// This file is part of HiOp. For details, see https://github.com/LLNL/hiop. HiOp
// is released under the BSD 3-clause license (https://opensource.org/licenses/BSD-3-Clause).
// Please also read "Additional BSD Notice" below.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
// i. Redistributions of source code must retain the above copyright notice, this list
// of conditions and the disclaimer below.
// ii. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the disclaimer (as noted below) in the documentation and/or
// other materials provided with the distribution.
// iii. Neither the name of the LLNS/LLNL nor the names of its contributors may be used to
// endorse or promote products derived from this software without specific prior written
// permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
// SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
// AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
// EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Additional BSD Notice
// 1. This notice is required to be provided under our contract with the U.S. Department
// of Energy (DOE). This work was produced at Lawrence Livermore National Laboratory under
// Contract No. DE-AC52-07NA27344 with the DOE.
// 2. Neither the United States Government nor Lawrence Livermore National Security, LLC
// nor any of their employees, makes any warranty, express or implied, or assumes any
// liability or responsibility for the accuracy, completeness, or usefulness of any
// information, apparatus, product, or process disclosed, or represents that its use would
// not infringe privately-owned rights.
// 3. Also, reference herein to any specific commercial products, process, or services by
// trade name, trademark, manufacturer or otherwise does not necessarily constitute or
// imply its endorsement, recommendation, or favoring by the United States Government or
// Lawrence Livermore National Security, LLC. The views and opinions of authors expressed
// herein do not necessarily state or reflect those of the United States Government or
// Lawrence Livermore National Security, LLC, and shall not be used for advertising or
// product endorsement purposes.

/**
 * @file hiopLinSolverCholCuSparse.cpp
 *
 * @author Cosmin G. Petra <petra1@llnl.gov>, LLNL
 *
 */

#ifndef AAA //HIOP_USE_CUDA

#include "hiopLinSolverCholCuSparse.hpp"

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cusolverSp.h>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>

namespace hiop
{

using Ordering = Eigen::AMDOrdering<SparseMatrixCSR::StorageIndex>;
using PermutationMatrix = Ordering::PermutationType;

hiopLinSolverCholCuSparse::hiopLinSolverCholCuSparse(const size_type& n, 
                                                     const size_type& nnz, 
                                                     hiopNlpFormulation* nlp)
  : hiopLinSolverIndefSparse(n, nnz, nlp),
    nnz_(nnz),
    buf_fact_(nullptr),
    rowptr_(nullptr),
    colind_(nullptr),
    values_buf_(nullptr),
    values_(nullptr),
    P_(nullptr),
    PT_(nullptr),
    map_nnz_perm_(nullptr),
    MMM_(nullptr),
    buf_perm_h_(nullptr),
    rhs_buf1_(nullptr),
    rhs_buf2_(nullptr)
{
  cusolverStatus_t ret;
  cusparseStatus_t ret_sp;
  
  ret_sp = cusparseCreate(&h_cusparse_);
  assert(ret_sp == CUSPARSE_STATUS_SUCCESS);
  
  ret = cusolverSpCreate(&h_cusolver_);
  assert(ret == CUSOLVER_STATUS_SUCCESS);

  ret = cusolverSpCreateCsrcholInfo(&info_);
  assert(ret == CUSOLVER_STATUS_SUCCESS);
  
  //matrix description
  ret_sp = cusparseCreateMatDescr(&mat_descr_);
  assert(ret_sp == CUSPARSE_STATUS_SUCCESS);
  ret_sp = cusparseSetMatType(mat_descr_, CUSPARSE_MATRIX_TYPE_GENERAL);
  assert(ret_sp == CUSPARSE_STATUS_SUCCESS);
  ret_sp = cusparseSetMatIndexBase(mat_descr_, CUSPARSE_INDEX_BASE_ZERO);
  assert(ret_sp == CUSPARSE_STATUS_SUCCESS);
  ret_sp = cusparseSetMatDiagType(mat_descr_, CUSPARSE_DIAG_TYPE_NON_UNIT);
  assert(ret_sp == CUSPARSE_STATUS_SUCCESS);

  MMM_ = new SparseMatrixCSR(n,n);
}

hiopLinSolverCholCuSparse::~hiopLinSolverCholCuSparse()
{
  cudaFree(rhs_buf1_);
  rhs_buf1_ = nullptr;

  cudaFree(rhs_buf2_);
  rhs_buf2_ = nullptr;

  cudaFree(map_nnz_perm_);
  map_nnz_perm_ = nullptr;
  
  delete buf_perm_h_;
  buf_perm_h_ = nullptr;
  
  delete MMM_;
  MMM_ = nullptr;
  
  cudaFree(buf_fact_);
  buf_fact_ = nullptr;

  cudaFree(P_);
  P_ = nullptr;
  cudaFree(PT_);
  PT_ = nullptr;
  
  cudaFree(rowptr_);
  cudaFree(colind_);
  cudaFree(values_buf_);
  cudaFree(values_);
  
  cusparseDestroyMatDescr(mat_descr_);
  cusolverSpDestroyCsrcholInfo(info_);
  cusolverSpDestroy(h_cusolver_);
  cusparseDestroy(h_cusparse_);
}

bool hiopLinSolverCholCuSparse::initial_setup()
{
  cusolverStatus_t ret;
  assert(nullptr == buf_fact_);
  size_type m = M.m();
  assert(m == MMM_->rows() && m == MMM_->cols());
  assert(nnz_ == MMM_->nonZeros());
  
  //
  // allocate device CSR arrays; then 
  // copy row and col arrays to the device
  // 
  assert(nullptr == rowptr_);
  cudaMalloc(&rowptr_, (m+1)*sizeof(int));

  assert(nullptr == colind_);
  cudaMalloc(&colind_, nnz_*sizeof(int));
  
  assert(nullptr == values_buf_);
  cudaMalloc(&values_buf_, nnz_*sizeof(double));

  assert(nullptr == values_);
  cudaMalloc(&values_, nnz_*sizeof(double));

  
  assert(rowptr_);
  assert(colind_);
  assert(values_buf_);
  assert(values_);

  bool dopermutation = true;
  if(dopermutation) {

    Eigen::Map<SparseMatrixCSR> M(mat_csr_->m(),
                                  mat_csr_->m(),
                                  mat_csr_->nnz(),
                                  mat_csr_->irowptr(),
                                  mat_csr_->jcolind(),
                                  mat_csr_->values());
                                  
    //
    // AMD reordering to improve the sparsity of the factor
    //
    PermutationMatrix P;
    Ordering ordering;
    ordering(MMM_->selfadjointView<Eigen::Upper>(), P);
    //ordering(M.selfadjointView<Eigen::Upper>(), P);
    
    const int* P_h = P.indices().data();
    int PT_h[m];
    for(int i=0; i<m; i++) {
      PT_h[P_h[i]] = i;
    }
    
    assert(nullptr == P_);
    cudaMalloc(&P_, m*sizeof(int));
    cudaMemcpy(P_, P_h, m*sizeof(int), cudaMemcpyHostToDevice);
    
    assert(nullptr == PT_);
    cudaMalloc(&PT_, m*sizeof(int));
    cudaMemcpy(PT_, PT_h, m*sizeof(int), cudaMemcpyHostToDevice);
    
    
    // get permutation buffer size
    size_t buf_size;
    assert(nullptr == buf_perm_h_);
    ret = cusolverSpXcsrperm_bufferSizeHost(h_cusolver_,
                                            m, m,
                                            nnz_,
                                            mat_descr_,
                                            MMM_->outerIndexPtr(),
                                            MMM_->innerIndexPtr(),
                                            P_h,
                                            P_h,
                                            &buf_size);
    assert(ret == CUSOLVER_STATUS_SUCCESS);
    buf_perm_h_ = new unsigned char[buf_size];
    
    //permuted CSR arrays (on host)
    int rowptr_perm_h[m+1];
    int colind_perm_h[nnz_];
    
    memcpy(rowptr_perm_h, MMM_->outerIndexPtr(), (m+1)*sizeof(int));
    memcpy(colind_perm_h, MMM_->innerIndexPtr(), nnz_*sizeof(int));
    
    //mapping (on host)
    int map_h[nnz_];
    for(int i=0; i<nnz_; i++) {
      map_h[i] = i;
    }
    
    ret = cusolverSpXcsrpermHost(h_cusolver_,
                                 m,
                                 m,
                                 nnz_,
                                 mat_descr_,
                                 rowptr_perm_h,
                                 colind_perm_h,
                                 P_h,
                                 P_h,
                                 map_h,
                                 buf_perm_h_);
    assert(ret == CUSOLVER_STATUS_SUCCESS);
    
    delete[] buf_perm_h_;
    buf_perm_h_ = nullptr;
    
    assert(nullptr == map_nnz_perm_);
    cudaMalloc(&map_nnz_perm_, nnz_*sizeof(int));
    //transfer the permutation map for nonzeros on device
    cudaMemcpy(map_nnz_perm_, map_h, nnz_*sizeof(int),  cudaMemcpyHostToDevice);

    // transfer the CSR index arrays on device
    //
    //values_ not needed here and will be updated in matrixChanged()
    //cudaMemcpy(rowptr_, MMM_->outerIndexPtr(), (m+1)*sizeof(int), cudaMemcpyHostToDevice);
    //cudaMemcpy(colind_, MMM_->innerIndexPtr(), nnz_*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(rowptr_, rowptr_perm_h, (m+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(colind_, colind_perm_h, nnz_*sizeof(int), cudaMemcpyHostToDevice);

  } else {
    cudaMemcpy(rowptr_, MMM_->outerIndexPtr(), (m+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(colind_, MMM_->innerIndexPtr(), nnz_*sizeof(int), cudaMemcpyHostToDevice);
    
    int map_h[nnz_];
    for(int i=0; i<nnz_; i++) {
      map_h[i] = i;
    }
    assert(nullptr == map_nnz_perm_);
    cudaMalloc(&map_nnz_perm_, nnz_*sizeof(int));
    cudaMemcpy(map_nnz_perm_, map_h, nnz_*sizeof(int),  cudaMemcpyHostToDevice);


    int PT_h[m];
    int P_h[m];
    for(int i=0; i<m; i++) {
      PT_h[i] = i;
      P_h[i] = i;
    }
  
    assert(nullptr == P_);
    cudaMalloc(&P_, m*sizeof(int));
    cudaMemcpy(P_, P_h, m*sizeof(int), cudaMemcpyHostToDevice);
    
    assert(nullptr == PT_);
    cudaMalloc(&PT_, m*sizeof(int));
    cudaMemcpy(PT_, PT_h, m*sizeof(int), cudaMemcpyHostToDevice);
  }
  
  //
  //analysis -> pattern of L
  //
  ret = cusolverSpXcsrcholAnalysis(h_cusolver_, m, nnz_, mat_descr_, rowptr_, colind_, info_);
  assert(ret == CUSOLVER_STATUS_SUCCESS);

  // TODO: this call as well as values_buf_ storage will be removed when the matrix is
  //going to reside on the device
  cudaMemcpy(values_buf_, MMM_->valuePtr(), nnz_*sizeof(double), cudaMemcpyHostToDevice);
  
  // buffer size
  size_t internalData; // in BYTEs
  ret = cusolverSpDcsrcholBufferInfo(h_cusolver_, 
                                     m, 
                                     nnz_, 
                                     mat_descr_, 
                                     values_buf_, 
                                     rowptr_, 
                                     colind_, 
                                     info_, 
                                     &internalData,
                                     &buf_fact_size_);
  assert(ret == CUSOLVER_STATUS_SUCCESS);
  
  cudaError_t ret_cu = cudaMalloc(&buf_fact_, sizeof(unsigned char)*buf_fact_size_); 
  assert(ret_cu == cudaSuccess);
  return true;
}
  
/* returns -1 if zero or negative pivots are encountered */
int hiopLinSolverCholCuSparse::matrixChanged()
{
  nlp_->runStats.linsolv.tmFactTime.start();

  size_type m = M.m();
  assert(m == M.n());
  assert(nnz_ == MMM_->nonZeros());
  
  hiopTimer t;
  cusolverStatus_t ret;
  
  if(nullptr == buf_fact_) {

    t.start();
    if(!initial_setup()) {
      nlp_->log->printf(hovError, 
                        "hiopLinSolverCholCuSparse: initial setup failed.\n");
      return -1;
    }
    t.stop();
    printf("initial setup took %.3f sec\n", t.getElapsedTime());
  }

  t.reset(); t.start();
  // copy the nonzeros to the device
  // row pointers and col indexes do not change and need not be copied to device
  //
  // TODO: this call as well as values_buf_ storage will be removed when the matrix is
  //going to reside on the device
  cudaMemcpy(values_buf_, MMM_->valuePtr(), nnz_*sizeof(double), cudaMemcpyHostToDevice);
  //t.stop(); printf("fact hdcopy took %.4f sec\n", t.getElapsedTime());
  
  t.reset(); t.start();
  //permute nonzeros
#if 1  
  cusparseDgthr(h_cusparse_, nnz_, values_buf_, values_, map_nnz_perm_, CUSPARSE_INDEX_BASE_ZERO);
#else
  //the permuted array of nonzeros
  cusparseSpVecDescr_t values_descr;
  //original nonzeros
  cusparseDnVecDescr_t values_buf;
  
  // Create sparse vector
  cusparseCreateSpVec(&vecX, nnz_, nnz_, map_nnz_perm_, values_,
                      CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  // Create dense vector 
  cusparseCreateDnVec(&vecY, nnz_, values_buf_, CUDA_R_32F);

  cusparseGather(h_cusparse_, vecY, vecX);
#endif
  //t.stop(); printf("fact numperm took %.4f sec\n", t.getElapsedTime());

  t.reset(); t.start();
  //
  //cuSOLVER factorization
  //
  ret = cusolverSpDcsrcholFactor(h_cusolver_, 
                                 m,
                                 nnz_, 
                                 mat_descr_,
                                 values_, 
                                 rowptr_, 
                                 colind_, 
                                 info_, 
                                 buf_fact_);
  if(ret != CUSOLVER_STATUS_SUCCESS) {
    // maybe regularization can help -> TODO: look into CUSOLVER return codes
    nlp_->log->printf(hovWarning, 
                      "hiopLinSolverCholCuSparse: factorization failed: CUSOLVER_STATUS=%d.\n",
                      ret);
    return -1;
  }

  const double zero_piv_tol = 1e-16;
  int position = -1;
  //ret = cusolverSpDcsrcholZeroPivot(h_cusolver_, info_, zero_piv_tol, &position);

  //t.stop(); printf("fact num  took %.4f sec\n", t.getElapsedTime());
  
  if(position>=0) {
    nlp_->log->printf(hovWarning, 
                      "hiopLinSolverCholCuSparse: the %dth pivot is <%.5e\n",
                      position,
                      zero_piv_tol);
    return -1;
  } else {
    return 0;
  }

/*
//hasZeroPivot();

*/
  nlp_->runStats.linsolv.tmFactTime.stop();
  return 0;
}

bool hiopLinSolverCholCuSparse::solve(hiopVector& x_in)
{
  hiopTimer t;
  cusolverStatus_t ret;

  nlp_->runStats.linsolv.tmTriuSolves.start(); 
  int m = M.m();
  assert(m == x_in.get_size());

  if(!rhs_buf1_) {
    cudaMalloc(&rhs_buf1_, m*sizeof(double));
  }
  if(!rhs_buf2_) {
    cudaMalloc(&rhs_buf2_, m*sizeof(double));
  }

  t.reset(); t.start();
  cudaMemcpy(rhs_buf1_, x_in.local_data(), m*sizeof(double), cudaMemcpyHostToDevice);
  //t.stop(); printf("solve hdcpy %.4f\n", t.getElapsedTime());

  t.reset(); t.start();
  // b = P*b
  cusparseDgthr(h_cusparse_, m, rhs_buf1_, rhs_buf2_, P_, CUSPARSE_INDEX_BASE_ZERO);
  //t.stop(); printf("solve perm1 %.4f\n", t.getElapsedTime());
  
  t.reset(); t.start();
  //solve -> two triangular solves
  ret = cusolverSpDcsrcholSolve(h_cusolver_, m, rhs_buf2_, rhs_buf1_, info_, buf_fact_);
  //t.stop(); printf("solve solve %.4f\n", t.getElapsedTime());

  t.reset(); t.start();
  //x = P'*x
  cusparseDgthr(h_cusparse_, m, rhs_buf1_, rhs_buf2_, PT_, CUSPARSE_INDEX_BASE_ZERO);
  //t.stop(); printf("solve perm2 %.4f\n", t.getElapsedTime());

  t.reset(); t.start();
  //transfer to host
  cudaMemcpy(x_in.local_data(), rhs_buf2_, m*sizeof(double), cudaMemcpyDeviceToHost);
  //t.stop(); printf("solve dhcopy %.4f\n", t.getElapsedTime());
 
  nlp_->runStats.linsolv.tmTriuSolves.stop(); 

  if(ret != CUSOLVER_STATUS_SUCCESS) {
    nlp_->log->printf(hovWarning,
                      "hiopLinSolverCholCuSparse: solve failed: CUSOLVER_STATUS=%d.\n",
                      ret);
    return false;
  }

  return true;
}
  
} // end of namespace

#endif //HIOP_USE_CUDA
