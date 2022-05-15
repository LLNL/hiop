// Copyright (c) 2022, Lawrence Livermore National Security, LLC.
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
 * @file hiopMatrixSparseCSRCUDA.cpp
 *
 * @author Cosmin G. Petra <petra1@llnl.gov>, LNNL
 *
 */

#include "hiopMatrixSparseCSRCUDA.hpp"

#ifdef HIOP_USE_CUDA

#include "hiopVectorPar.hpp"

#include <algorithm> //for std::min
#include <cmath> //for std::isfinite
#include <cstring>
#include <vector>
#include <numeric>
#include <cassert>
#include <sstream>
#include <iostream>     // std::cout, std::fixed
#include <iomanip>      // std::setprecision

#include "hiopCppStdUtils.hpp"
#include <set>
#include <map>

namespace hiop
{

hiopMatrixSparseCSRCUDA::hiopMatrixSparseCSRCUDA(size_type rows, size_type cols, size_type nnz)
  : hiopMatrixSparseCSR(rows, cols, nnz),
    irowptr_(nullptr),
    jcolind_(nullptr),
    values_(nullptr),
    buf_col_(nullptr),
    row_starts_(nullptr)
{
  cusparseStatus_t ret_sp = cusparseCreate(&h_cusparse_);
  assert(ret_sp == CUSPARSE_STATUS_SUCCESS);

  // the call below initializes the fields of mat_descr_ to what we need
  //  - MatrixType -> CUSPARSE_MATRIX_TYPE_GENERAL
  //  - IndexBase  -> CUSPARSE_INDEX_BASE_ZERO
  //  - leaves other fields uninitialized
  cusparseStatus_t st = cusparseCreateMatDescr(&mat_descr_);
  assert(st == CUSPARSE_STATUS_SUCCESS);

}

hiopMatrixSparseCSRCUDA::hiopMatrixSparseCSRCUDA()
  : hiopMatrixSparseCSR(0, 0, 0),
    irowptr_(nullptr),
    jcolind_(nullptr),
    values_(nullptr),
    buf_col_(nullptr),
    row_starts_(nullptr)
{
  cusparseStatus_t ret_sp = cusparseCreate(&h_cusparse_);
  assert(ret_sp == CUSPARSE_STATUS_SUCCESS);

  // the call below initializes the fields of mat_descr_ to what we need
  //  - MatrixType -> CUSPARSE_MATRIX_TYPE_GENERAL
  //  - IndexBase  -> CUSPARSE_INDEX_BASE_ZERO
  //  - leaves other fields uninitialized
  cusparseStatus_t st = cusparseCreateMatDescr(&mat_descr_);
  assert(st == CUSPARSE_STATUS_SUCCESS);
}

  
hiopMatrixSparseCSRCUDA::~hiopMatrixSparseCSRCUDA()
{
  dealloc();
  cusparseDestroy(h_cusparse_);
  //cusolverSpDestroy(h_cusolver_);

  cusparseStatus_t st = cusparseDestroyMatDescr(mat_descr_);
  assert(st == CUSPARSE_STATUS_SUCCESS);
}

void hiopMatrixSparseCSRCUDA::alloc()
{
  cudaError_t err;
  err = cudaMalloc(&irowptr_, (nrows_+1)*sizeof(index_type));
  assert(cudaSuccess == err && irowptr_);
  
  err = cudaMalloc(&jcolind_, nnz_*sizeof(index_type));
  assert(cudaSuccess == err && jcolind_);
  
  err = cudaMalloc(&values_, nnz_*sizeof(double));
  assert(cudaSuccess == err && values_);
}


void hiopMatrixSparseCSRCUDA::dealloc()
{  
  cudaError_t err;
  err = cudaFree(values_);
  assert(cudaSuccess == err);
  values_ = nullptr;

  err = cudaFree(jcolind_);
  assert(cudaSuccess == err);
  jcolind_ = nullptr;
  
  err = cudaFree(irowptr_);
  assert(cudaSuccess == err);
  irowptr_ = nullptr;
}
  
void hiopMatrixSparseCSRCUDA::setToZero()
{
  assert(false && "work in progress");
}
void hiopMatrixSparseCSRCUDA::setToConstant(double c)
{
  assert(false && "work in progress");
}

/** y = beta * y + alpha * this * x */
void hiopMatrixSparseCSRCUDA::timesVec(double beta,
                                       hiopVector& y,
                                       double alpha,
                                       const hiopVector& x) const
{
  assert(false && "work in progress");
  assert(x.get_size() == ncols_);
  assert(y.get_size() == nrows_);

  hiopVectorPar& yy = dynamic_cast<hiopVectorPar&>(y);
  const hiopVectorPar& xx = dynamic_cast<const hiopVectorPar&>(x);

  double* y_data = yy.local_data();
  const double* x_data = xx.local_data_const();

  timesVec(beta, y_data, alpha, x_data);
}

/** y = beta * y + alpha * this * x */
void hiopMatrixSparseCSRCUDA::timesVec(double beta,
                                       double* y,
                                       double alpha,
                                       const double* x) const
{
  assert(false && "not yet implemented");
}

/** y = beta * y + alpha * this^T * x */
void hiopMatrixSparseCSRCUDA::transTimesVec(double beta,
                                            hiopVector& y,
                                            double alpha,
                                            const hiopVector& x) const
{
  assert(false && "work in progress");
  assert(x.get_size() == nrows_);
  assert(y.get_size() == ncols_);
  
  hiopVectorPar& yy = dynamic_cast<hiopVectorPar&>(y);
  const hiopVectorPar& xx = dynamic_cast<const hiopVectorPar&>(x);
  
  double* y_data = yy.local_data();
  const double* x_data = xx.local_data_const();
  
  transTimesVec(beta, y_data, alpha, x_data);
}

/** y = beta * y + alpha * this^T * x */
void hiopMatrixSparseCSRCUDA::transTimesVec(double beta,
                                            double* y,
                                            double alpha,
                                            const double* x) const
{
  assert(false && "not yet implemented");
} 

void hiopMatrixSparseCSRCUDA::timesMat(double beta,
                                       hiopMatrix& W,
                                       double alpha,
                                       const hiopMatrix& X) const
{
  assert(false && "not needed");
}

void hiopMatrixSparseCSRCUDA::transTimesMat(double beta,
                                            hiopMatrix& W,
                                            double alpha,
                                            const hiopMatrix& X) const
{
  assert(false && "not needed");
}

void hiopMatrixSparseCSRCUDA::timesMatTrans(double beta,
                                            hiopMatrix& Wmat,
                                            double alpha,
                                            const hiopMatrix& M2mat) const
{
  assert(false && "not needed");
}

void hiopMatrixSparseCSRCUDA::addDiagonal(const double& alpha, const hiopVector& D)
{
  assert(false && "work in progress");
}

void hiopMatrixSparseCSRCUDA::addDiagonal(const double& value)
{
  assert(false && "not needed");
}
void hiopMatrixSparseCSRCUDA::addSubDiagonal(const double& alpha, index_type start, const hiopVector& d_)
{
  assert(false && "not needed");
}

void hiopMatrixSparseCSRCUDA::copySubDiagonalFrom(const index_type& start_on_dest_diag,
                                                  const size_type& num_elems,
                                                  const hiopVector& d_,
                                                  const index_type& start_on_nnz_idx,
                                                  double scal)
{
  assert(false && "not implemented");
}

void hiopMatrixSparseCSRCUDA::setSubDiagonalTo(const index_type& start_on_dest_diag,
                                               const size_type& num_elems,
                                               const double& c,
                                               const index_type& start_on_nnz_idx)
{
  assert(start_on_dest_diag>=0 && start_on_dest_diag+num_elems<=this->nrows_);
  assert(false && "not implemented");
}

void hiopMatrixSparseCSRCUDA::addMatrix(double alpha, const hiopMatrix& X)
{
  assert(false && "not needed");
}

/* block of W += alpha*transpose(this)
 * Note W; contains only the upper triangular entries */
void hiopMatrixSparseCSRCUDA::
transAddToSymDenseMatrixUpperTriangle(index_type row_start,
                                      index_type col_start,
                                      double alpha,
                                      hiopMatrixDense& W) const
{
  assert(row_start>=0 && row_start+ncols_<=W.m());
  assert(col_start>=0 && col_start+nrows_<=W.n());
  assert(W.n()==W.m());
  
  assert(false && "not yet implemented");
}

double hiopMatrixSparseCSRCUDA::max_abs_value()
{
  assert(false && "work in progress");
  //char norm='M'; size_type one=1;
  //double maxv = DLANGE(&norm, &one, &nnz_, values_, &one, nullptr);
  //return maxv;
  return 0.0;
}

void hiopMatrixSparseCSRCUDA::row_max_abs_value(hiopVector &ret_vec)
{
  assert(ret_vec.get_local_size() == nrows_);

  hiopVectorPar& yy = dynamic_cast<hiopVectorPar&>(ret_vec);
  yy.setToZero();
  double* y_data = yy.local_data();

  assert(false && "not yet implemented");
}

void hiopMatrixSparseCSRCUDA::scale_row(hiopVector &vec_scal, const bool inv_scale)
{
  assert(vec_scal.get_local_size() == nrows_);

  hiopVectorPar& vscal = dynamic_cast<hiopVectorPar&>(vec_scal);  
  double* vd = vscal.local_data();
  assert(false && "not yet implemented");
}

bool hiopMatrixSparseCSRCUDA::isfinite() const
{
  assert(false && "work in progress");
  for(index_type i=0; i<nnz_; i++)
    if(false==std::isfinite(values_[i])) return false;
  return true;
}

hiopMatrixSparse* hiopMatrixSparseCSRCUDA::alloc_clone() const
{
  return new hiopMatrixSparseCSRCUDA(nrows_, ncols_, nnz_);
}

hiopMatrixSparse* hiopMatrixSparseCSRCUDA::new_copy() const
{
  hiopMatrixSparseCSRCUDA* copy = new hiopMatrixSparseCSRCUDA(nrows_, ncols_, nnz_);
  assert(false && "work in progress");
  return copy;
}
void hiopMatrixSparseCSRCUDA::copyFrom(const hiopMatrixSparse& dm)
{
  assert(false && "to be implemented - method def too vague for now");
}

/// @brief copy to 3 arrays.
/// @pre these 3 arrays are not nullptr
void hiopMatrixSparseCSRCUDA::copy_to(index_type* irow, index_type* jcol, double* val)
{
  assert(irow && jcol && val);
  assert(false && "work in progress");
  //memcpy(irow, irowptr_, (1+nrows_)*sizeof(index_type));
  //memcpy(jcol, jcolind_, nnz_*sizeof(index_type));
  //memcpy(val, values_, nnz_*sizeof(double));
}

void hiopMatrixSparseCSRCUDA::copy_to(hiopMatrixDense& W)
{
  assert(false && "not needed");
  assert(W.m() == nrows_);
  assert(W.n() == ncols_);
}

void hiopMatrixSparseCSRCUDA::
addMDinvMtransToDiagBlockOfSymDeMatUTri(index_type rowAndCol_dest_start,
                                        const double& alpha,
                                        const hiopVector& D, hiopMatrixDense& W) const
{
  assert(false && "not needed");
}

/*
 * block of W += alpha * M1 * D^{-1} * transpose(M2), where M1=this
 *  Sizes: M1 is (m1 x nx);  D is vector of len nx, M2 is  (m2, nx)
 */
void hiopMatrixSparseCSRCUDA::
addMDinvNtransToSymDeMatUTri(index_type row_dest_start,
                             index_type col_dest_start,
                             const double& alpha,
                             const hiopVector& D,
                             const hiopMatrixSparse& M2mat,
                             hiopMatrixDense& W) const
{
  assert(false && "not needed");
}

void hiopMatrixSparseCSRCUDA::copyRowsFrom(const hiopMatrix& src_gen,
                                           const index_type* rows_idxs,
                                           size_type n_rows)
{
  const hiopMatrixSparseCSRCUDA& src = dynamic_cast<const hiopMatrixSparseCSRCUDA&>(src_gen);
  assert(this->m() == n_rows);
  assert(this->numberOfNonzeros() <= src.numberOfNonzeros());
  assert(this->n() == src.n());
  assert(n_rows <= src.m());

  assert(false && "not yet implemented");
}

/**
 * @brief Copy 'n_rows' rows started from 'rows_src_idx_st' (array of size 'n_rows') from 'src' to the destination,
 * which starts from the 'rows_dest_idx_st'th row in 'this'
 *
 * @pre 'this' must have exactly, or more than 'n_rows' rows
 * @pre 'this' must have exactly, or more cols than 'src'
 */
void hiopMatrixSparseCSRCUDA::copyRowsBlockFrom(const hiopMatrix& src_gen,
                                                const index_type& rows_src_idx_st, const size_type& n_rows,
                                                const index_type& rows_dest_idx_st, const size_type& dest_nnz_st)
{
  const hiopMatrixSparseCSRCUDA& src = dynamic_cast<const hiopMatrixSparseCSRCUDA&>(src_gen);
  assert(this->numberOfNonzeros() >= src.numberOfNonzeros());
  assert(this->n() >= src.n());
  assert(n_rows + rows_src_idx_st <= src.m());
  assert(n_rows + rows_dest_idx_st <= this->m());

  assert(false && "not yet implemented");
}

void hiopMatrixSparseCSRCUDA::copySubmatrixFrom(const hiopMatrix& src_gen,
                                                const index_type& dest_row_st,
                                                const index_type& dest_col_st,
                                                const size_type& dest_nnz_st,
                                                const bool offdiag_only)
{
  const hiopMatrixSparseCSRCUDA& src = dynamic_cast<const hiopMatrixSparseCSRCUDA&>(src_gen);
  auto m_rows = src.m();
  auto n_cols = src.n();

  assert(this->numberOfNonzeros() >= src.numberOfNonzeros());
  assert(n_cols + dest_col_st <= this->n() );
  assert(m_rows + dest_row_st <= this->m());
  assert(dest_nnz_st <= this->numberOfNonzeros());

  assert(false && "not yet implemented");
}

void hiopMatrixSparseCSRCUDA::copySubmatrixFromTrans(const hiopMatrix& src_gen,
                                                     const index_type& dest_row_st,
                                                     const index_type& dest_col_st,
                                                     const size_type& dest_nnz_st,
                                                     const bool offdiag_only)
{
  const auto& src = dynamic_cast<const hiopMatrixSparseCSRCUDA&>(src_gen);
  auto m_rows = src.n();
  auto n_cols = src.m();

  assert(this->numberOfNonzeros() >= src.numberOfNonzeros());
  assert(n_cols + dest_col_st <= this->n() );
  assert(m_rows + dest_row_st <= this->m());
  assert(dest_nnz_st <= this->numberOfNonzeros());

  assert(false && "not yet implemented");
}

void hiopMatrixSparseCSRCUDA::
setSubmatrixToConstantDiag_w_colpattern(const double& scalar,
                                        const index_type& dest_row_st,
                                        const index_type& dest_col_st,
                                        const size_type& dest_nnz_st,
                                        const size_type& nnz_to_copy,
                                        const hiopVector& ix)
{
  assert(ix.get_local_size() + dest_row_st <= this->m());
  assert(nnz_to_copy + dest_col_st <= this->n() );
  assert(dest_nnz_st + nnz_to_copy <= this->numberOfNonzeros());
  
  assert(false && "not yet implemented");
}

void hiopMatrixSparseCSRCUDA::
setSubmatrixToConstantDiag_w_rowpattern(const double& scalar,
                                        const index_type& dest_row_st,
                                        const index_type& dest_col_st,
                                        const size_type& dest_nnz_st,
                                        const size_type& nnz_to_copy,
                                        const hiopVector& ix)
{
  assert(nnz_to_copy + dest_row_st <= this->m());
  assert(ix.get_local_size() + dest_col_st <= this->n() );
  assert(dest_nnz_st + nnz_to_copy <= this->numberOfNonzeros());
  
  assert(false && "not yet implemented");
}

void hiopMatrixSparseCSRCUDA::
copyDiagMatrixToSubblock(const double& src_val,
                         const index_type& dest_row_st,
                         const index_type& col_dest_st,
                         const size_type& dest_nnz_st,
                         const size_type &nnz_to_copy)
{
  assert(this->numberOfNonzeros() >= nnz_to_copy+dest_nnz_st);
  assert(this->n() >= nnz_to_copy);
  assert(nnz_to_copy + dest_row_st <= this->m());
  assert(nnz_to_copy + col_dest_st <= this->n());

  assert(false && "not yet implemented");
}

void hiopMatrixSparseCSRCUDA::
copyDiagMatrixToSubblock_w_pattern(const hiopVector& dx,
                                   const index_type& dest_row_st,
                                   const index_type& dest_col_st,
                                   const size_type& dest_nnz_st,
                                   const size_type &nnz_to_copy,
                                   const hiopVector& ix)
{
  assert(this->numberOfNonzeros() >= nnz_to_copy+dest_nnz_st);
  assert(this->n() >= nnz_to_copy);
  assert(nnz_to_copy + dest_row_st <= this->m());
  assert(nnz_to_copy + dest_col_st <= this->n());

  assert(false && "not yet implemented");
}

void hiopMatrixSparseCSRCUDA::print(FILE* file,
                                    const char* msg/*=nullptr*/,
                                    int maxRows/*=-1*/,
                                    int maxCols/*=-1*/,
                                    int rank/*=-1*/) const
{
  
  int myrank_=0, numranks=1; //this is a local object => always print

  if(file==nullptr) file = stdout;

  int max_elems = maxRows>=0 ? maxRows : nnz_;
  max_elems = std::min(max_elems, nnz_);
  
  if(myrank_==rank || rank==-1) {

    index_type* irowptr = new index_type[nrows_+1];
    index_type* jcolind = new index_type[nnz_];
    double* values = new double[nnz_];
    
    cudaMemcpy(irowptr, irowptr_, (nrows_+1)*sizeof(index_type), cudaMemcpyDeviceToHost);
    cudaMemcpy(jcolind, jcolind_, nnz_*sizeof(index_type), cudaMemcpyDeviceToHost);
    cudaMemcpy(values, values_, nnz_*sizeof(double), cudaMemcpyDeviceToHost);
    
    std::stringstream ss;
    if(nullptr==msg) {
      if(numranks>1) {
        ss << "CSR matrix of size " << m() << " " << n() << " and nonzeros " 
           << numberOfNonzeros() << ", printing " <<  max_elems << " elems (on rank="
           << myrank_ << ")" << std::endl;
      } else {
        ss << "CSR matrix of size " << m() << " " << n() << " and nonzeros " 
           << numberOfNonzeros() << ", printing " <<  max_elems << " elems" << std::endl;
      }
    } else {
      ss << msg << " ";
    }

    // using matlab indices (starting at 1)
    //fprintf(file, "iRow_=[");
    ss << "iRow_=[";

    for(index_type i=0; i<nrows_; i++) {
      const index_type ip1 = i+1;
      for(int p=irowptr[i]; p<irowptr[i+1] && p<max_elems; ++p) {
        ss << ip1 << "; ";
      }
    }
    ss << "];" << std::endl;

    ss << "jCol_=[";
    for(index_type it=0; it<max_elems; it++) {
      ss << (jcolind[it]+1) << "; ";
    }
    ss << "];" << std::endl;
    
    ss << "v=[";
    ss << std::scientific << std::setprecision(16);
    for(index_type it=0; it<max_elems; it++) {
      ss << values[it] << "; ";
    }
    //fprintf(file, "];\n");
    ss << "];" << std::endl;

    fprintf(file, "%s", ss.str().c_str());

    delete[] values;
    delete[] irowptr;
    delete[] jcolind;
  }
}


//M = X*D*Y -> computes nnz in M and allocates M 
//By convention, M is mxn, X is mxK and Y is Kxn
hiopMatrixSparseCSR* hiopMatrixSparseCSRCUDA::times_mat_alloc(const hiopMatrixSparseCSR& Y) const
{
  assert(false && "work in progress");
  //allocate result M
  return nullptr; //new hiopMatrixSparseCSRCUDA(m, n, nnzM);
} 

/**
 *  M = X*D*Y -> computes nnz in M and allocates M 
 * By convention, M is mxn, X is mxK, Y is Kxn, and D is size K.
 * 
 * The algorithm uses the fact that the sparsity pattern of the i-th row of M is
 *           K
 * M_{i*} = sum x_{ik} Y_{k*}   (see Tim Davis book p.17)
 *          k=1
 * Therefore, to get sparsity pattern of the i-th row of M:
 *  1. we k-iterate over nonzeros (i,k) in the i-th row of X
 *  2. for each such k we j-iterate over the nonzeros (k,j) in the k-th row of Y and 
 *  3. count (i,j) as nonzero of M 
 */
void hiopMatrixSparseCSRCUDA::times_mat_symbolic(hiopMatrixSparseCSR& M_in,
                                                 const hiopMatrixSparseCSR& Y_in) const
{
  auto& M = dynamic_cast<hiopMatrixSparseCSRCUDA&>(M_in);
  auto& Y = dynamic_cast<const hiopMatrixSparseCSRCUDA&>(Y_in);
  assert(false && "work in progress");
}

void hiopMatrixSparseCSRCUDA::times_mat_numeric(double beta,
                                                hiopMatrixSparseCSR& M_in,
                                                double alpha,
                                                const hiopMatrixSparseCSR& Y_in)
{
  auto& M = dynamic_cast<hiopMatrixSparseCSRCUDA&>(M_in);
  auto& Y = dynamic_cast<const hiopMatrixSparseCSRCUDA&>(Y_in);
  assert(false && "work in progress");
}

void hiopMatrixSparseCSRCUDA::form_from_symbolic(const hiopMatrixSparseTriplet& M)
{
  if(M.m()!=nrows_ || M.n()!=ncols_ || M.numberOfNonzeros()!=nnz_) {
    dealloc();
    
    nrows_ = M.m();
    ncols_ = M.n();
    nnz_ = M.numberOfNonzeros();

    alloc();
  }

  assert(nnz_>=0);
  if(nnz_<=0) {
    return;
  }
  
  assert(irowptr_);
  assert(jcolind_);
  assert(values_);

  //transfer coo/triplet to device
  int* d_rowind=nullptr;
  cudaMalloc(&d_rowind, nnz_*sizeof(index_type));
  assert(d_rowind);
  cudaMemcpy(d_rowind, M.i_row(), nnz_*sizeof(index_type), cudaMemcpyHostToDevice);

  //use cuda API
  cusparseStatus_t st = cusparseXcoo2csr(h_cusparse_,
                                         d_rowind,
                                         nnz_,
                                         nrows_,
                                         irowptr_,
                                         CUSPARSE_INDEX_BASE_ZERO);
  assert(CUSPARSE_STATUS_SUCCESS == st);

  cudaFree(d_rowind);
  
  //j indexes can be just transfered
  cudaMemcpy(jcolind_, M.j_col(), nnz_*sizeof(index_type), cudaMemcpyHostToDevice);
}

void hiopMatrixSparseCSRCUDA::form_from_numeric(const hiopMatrixSparseTriplet& M)
{
  assert(irowptr_ && jcolind_ && values_);
  assert(nrows_ == M.m());
  assert(ncols_ == M.n());
  assert(nnz_ == M.numberOfNonzeros());

  cudaMemcpy(values_, M.M(), nnz_*sizeof(double), cudaMemcpyHostToDevice);
}

void hiopMatrixSparseCSRCUDA::form_transpose_from_symbolic(const hiopMatrixSparseTriplet& M)
{
  if(M.m()!=ncols_ || M.n()!=nrows_ || M.numberOfNonzeros()!=nnz_) {
    dealloc();
    
    nrows_ = M.n();
    ncols_ = M.m();
    nnz_ = M.numberOfNonzeros();

    alloc();
  }

  assert(nnz_>=0);
  if(nnz_<=0) {
    return;
  }
  
  assert(irowptr_);
  assert(jcolind_);
  assert(values_);

  //First create a CUDA CSR matrix from M and then transpose it with cusparseCsr2cscEx2
  //hiopMatrixSparseCSRCUDA d_M;
  //d_M.form_from_symbol
  
}

void hiopMatrixSparseCSRCUDA::form_transpose_from_numeric(const hiopMatrixSparseTriplet& M)
{
  assert(irowptr_ && jcolind_ && values_ && row_starts_);
  assert(nrows_ == M.n());
  assert(ncols_ == M.m());
  assert(nnz_ == M.numberOfNonzeros());

  assert(false && "work in progress");
}

void hiopMatrixSparseCSRCUDA::form_transpose_from_symbolic(const hiopMatrixSparseCSR& M)
{
  if(M.m()!=ncols_ || M.n()!=nrows_ || M.numberOfNonzeros()!=nnz_) {
    dealloc();
    
    nrows_ = M.n();
    ncols_ = M.m();
    nnz_ = M.numberOfNonzeros();

    alloc();
  }

  assert(nnz_>=0);
  if(nnz_<=0) {
    return;
  }
  
  assert(irowptr_);
  assert(jcolind_);
  assert(values_);
  
}

void hiopMatrixSparseCSRCUDA::form_transpose_from_numeric(const hiopMatrixSparseCSR& M)
{
  assert(irowptr_ && jcolind_ && values_ && row_starts_);
  assert(nrows_ == M.n());
  assert(ncols_ == M.m());
  assert(nnz_ == M.numberOfNonzeros());

  assert(false && "work in progress");
}


void hiopMatrixSparseCSRCUDA::form_diag_from_symbolic(const hiopVector& D)
{
  int m = D.get_size();
  if(m!=ncols_ || m!=nrows_ || m!=nnz_) {
    dealloc();
    
    nrows_ = m;
    ncols_ = m;
    nnz_ = m;

    alloc();
  }

  assert(irowptr_);
  assert(jcolind_);
  assert(values_);

  assert(false && "work in progress");
}

void hiopMatrixSparseCSRCUDA::form_diag_from_numeric(const hiopVector& D)
{
  assert(false && "work in progress");
  assert(D.get_size()==ncols_ && D.get_size()==nrows_ && D.get_size()==nnz_);
  //memcpy(values_, D.local_data_const(), nrows_*sizeof(double));
}

///Column scaling or right multiplication by a diagonal: `this`=`this`*D
void hiopMatrixSparseCSRCUDA::scale_cols(const hiopVector& D)
{
  assert(false && "work in progress");
  assert(ncols_ == D.get_size());
}

/// @brief Row scaling or left multiplication by a diagonal: `this`=D*`this`
void hiopMatrixSparseCSRCUDA::scale_rows(const hiopVector& D)
{
  assert(false && "work in progress");
  assert(nrows_ == D.get_size());
}

// sparsity pattern of M=X+Y, where X is `this`
hiopMatrixSparseCSR* hiopMatrixSparseCSRCUDA::add_matrix_alloc(const hiopMatrixSparseCSR& Y) const
{
  assert(nrows_ == Y.m());
  assert(ncols_ == Y.n());
  assert(false && "work in progress");
  //allocate result M
  return nullptr; //new hiopMatrixSparseCSRCUDA(nrows_, ncols_, nnzM);
}

/**
 * Computes sparsity pattern of M = X+Y (i.e., populates the row pointers and 
 * column indexes arrays) of `M`.
 *
 */
void hiopMatrixSparseCSRCUDA::
add_matrix_symbolic(hiopMatrixSparseCSR& M_in, const hiopMatrixSparseCSR& Y_in) const
{
  auto& M = dynamic_cast<hiopMatrixSparseCSRCUDA&>(M_in);
  auto& Y = dynamic_cast<const hiopMatrixSparseCSRCUDA&>(Y_in);

  assert(nrows_ == Y.m());
  assert(ncols_ == Y.n());

  assert(false && "work in progress");
}

/**
 * Performs matrix addition M = alpha*X + beta*Y numerically
 */
void hiopMatrixSparseCSRCUDA::add_matrix_numeric(hiopMatrixSparseCSR& M_in,
                                                 double alpha,
                                                 const hiopMatrixSparseCSR& Y_in,
                                                 double beta) const
{
  auto& M = dynamic_cast<hiopMatrixSparseCSRCUDA&>(M_in);
  auto& Y = dynamic_cast<const hiopMatrixSparseCSRCUDA&>(Y_in);

  assert(nrows_ == Y.m());
  assert(ncols_ == Y.n());

  assert(false && "work in progress");
}

void hiopMatrixSparseCSRCUDA::set_diagonal(const double& val)
{
  assert(false && "work in progress");
  assert(irowptr_ && jcolind_ && values_);
}

bool hiopMatrixSparseCSRCUDA::check_csr_is_ordered()
{
  assert(false && "work in progress");
  return true;
}

} //end of namespace

#endif //#ifdef HIOP_USE_CUDA
