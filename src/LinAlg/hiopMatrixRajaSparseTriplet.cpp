// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory (LLNL).
// Written by Cosmin G. Petra, petra1@llnl.gov.
// LLNL-CODE-742473. All rights reserved.
//
// This file is part of HiOp. For details, see https://github.com/LLNL/hiop. HiOp
// is released under the BSD 3-clause license (https://opensource.org/licenses/BSD-3-Clause).
// Please also read “Additional BSD Notice” below.
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
 * @file hiopMatrixRajaSparseTriplet.cpp
 *
 * @author Asher Mancinelli <asher.mancinelli@pnnl.gov>, PNNL
 * @author Slaven Peles <slaven.peles@pnnl.gov>, PNNL
 * @author Cameron Rutherford <robert.rutherford@pnnl.gov>, PNNL
 * @author Jake K. Ryan <jake.ryan@pnnl.gov>, PNNL
 *
 */
#include "hiopMatrixRajaSparseTriplet.hpp"
#include "hiopVectorRajaPar.hpp"

#include <umpire/Allocator.hpp>
#include <umpire/ResourceManager.hpp>
#include <RAJA/RAJA.hpp>

#include "hiop_blasdefs.hpp"

#include <algorithm> //for std::min
#include <cmath> //for std::isfinite
#include <cstring>

#include <cassert>

namespace hiop
{
#ifdef HIOP_USE_GPU
  #include "cuda.h"
  using hiop_raja_exec   = RAJA::cuda_exec<128>;
  using hiop_raja_reduce = RAJA::cuda_reduce;
  using hiop_raja_atomic = RAJA::cuda_atomic;
  #define RAJA_LAMBDA [=] __device__
#else
  using hiop_raja_exec   = RAJA::omp_parallel_for_exec;
  using hiop_raja_reduce = RAJA::omp_reduce;
  using hiop_raja_atomic = RAJA::omp_atomic;
  #define RAJA_LAMBDA [=]
#endif


/// @brief Constructs a hiopMatrixRajaSparseTriplet with the given dimensions and memory space
hiopMatrixRajaSparseTriplet::hiopMatrixRajaSparseTriplet(int rows, int cols, int _nnz, std::string memspace)
  : hiopMatrixSparse(rows, cols, _nnz), row_starts_host(NULL), mem_space_(memspace)
{
  if(rows==0 || cols==0)
  {
    assert(nnz_==0 && "number of nonzeros must be zero when any of the dimensions are 0");
    nnz_ = 0;
  }

#ifndef HIOP_USE_GPU
  mem_space_ = "HOST";
#endif

  //printf("Memory space: %s\n", mem_space_.c_str());

  auto& resmgr = umpire::ResourceManager::getInstance();
  umpire::Allocator devAlloc = resmgr.getAllocator(mem_space_);
  umpire::Allocator hostAlloc = resmgr.getAllocator("HOST");

  iRow_ = static_cast<int*>(devAlloc.allocate(nnz_ * sizeof(int)));
  jCol_ = static_cast<int*>(devAlloc.allocate(nnz_ * sizeof(int)));
  values_ = static_cast<double*>(devAlloc.allocate(nnz_ * sizeof(double)));

  // create host mirror if memory space is on the device
  if (mem_space_ == "DEVICE")
  {
    iRow_host_ = static_cast<int*>(hostAlloc.allocate(nnz_ * sizeof(int)));
    jCol_host_ = static_cast<int*>(hostAlloc.allocate(nnz_ * sizeof(int)));
    values_host_ = static_cast<double*>(hostAlloc.allocate(nnz_ * sizeof(double)));
  }
  else
  {
    iRow_host_ = iRow_;
    jCol_host_ = jCol_;
    values_host_ = values_;
  }
}

/// @brief Destructor for hiopMatrixRajaSparseTriplet
hiopMatrixRajaSparseTriplet::~hiopMatrixRajaSparseTriplet()
{
  delete row_starts_host;
  auto& resmgr = umpire::ResourceManager::getInstance();
  umpire::Allocator devAlloc = resmgr.getAllocator(mem_space_);
  umpire::Allocator hostAlloc = resmgr.getAllocator("HOST");

  devAlloc.deallocate(iRow_);
  devAlloc.deallocate(jCol_);
  devAlloc.deallocate(values_);

  // deallocate host mirror if memory space is on device
  if (mem_space_ == "DEVICE")
  {
    hostAlloc.deallocate(iRow_host_);
    hostAlloc.deallocate(jCol_host_);
    hostAlloc.deallocate(values_host_);
  }
}

/**
 * @brief Sets all the values of this matrix to zero.
 */
void hiopMatrixRajaSparseTriplet::setToZero()
{
  setToConstant(0.0);
}

/**
 * @brief Sets all the values of this matrix to some constant.
 * 
 * @param c A real number.
 */
void hiopMatrixRajaSparseTriplet::setToConstant(double c)
{
  double* dd = this->values_;
  auto nz = nnz_;
  RAJA::forall<hiop_raja_exec>(RAJA::RangeSegment(0, nz),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      dd[i] = c;
    });
}

/**
 * @brief Multiplies this matrix by a vector and stores it in an output vector.
 * 
 * @param beta Amount to scale the output vector by before adding to it.
 * @param y The output vector.
 * @param alpha The amount to scale this matrix by before multiplying.
 * @param x The vector by which to multiply this matrix.
 * 
 * @pre _x_'s length must equal the number of columns in this matrix.
 * @pre _y_'s length must equal the number of rows in this matrix.
 * @post _y_ will contain the output of the following equation:
 * 
 * The full operation performed is:
 * _y_ = _beta_ * _y_ + _alpha_ * this * _x_
 */
void hiopMatrixRajaSparseTriplet::timesVec(double beta,  hiopVector& y,
  double alpha, const hiopVector& x) const
{
  assert(x.get_size() == ncols_);
  assert(y.get_size() == nrows_);

  auto& yy = dynamic_cast<hiopVectorRajaPar&>(y);
  const auto& xx = dynamic_cast<const hiopVectorRajaPar&>(x);

  double* y_data = yy.local_data();
  const double* x_data = xx.local_data_const();

  timesVec(beta, y_data, alpha, x_data);
}
 
/**
 * @brief Multiplies this matrix by a vector and stores it in an output vector.
 * 
 * @see above timesVec function for more detail. This overload takes raw data
 * pointers rather than hiop constructs.
 */
void hiopMatrixRajaSparseTriplet::timesVec(double beta,  double* y,
  double alpha, const double* x) const
{
  // y = beta * y
  RAJA::forall<hiop_raja_exec>(RAJA::RangeSegment(0, nrows_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      y[i] *= beta;
    });

#ifndef NDEBUG
  auto nrs = nrows_;
  auto ncs = ncols_;
#endif
  auto irw = iRow_;
  auto jcl = jCol_;
  auto vls = values_;
  // atomic is needed to prevent data race from ocurring;
  // y[jCol_[i]] can be referenced by multiple threads concurrently
  RAJA::forall<hiop_raja_exec>(RAJA::RangeSegment(0, nnz_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      assert(irw[i] < nrs);
      assert(jcl[i] < ncs);
      RAJA::AtomicRef<double, hiop_raja_atomic> yy(&y[irw[i]]);
      yy += alpha * x[jcl[i]] * vls[i];
    });
}

/**
 * @brief Multiplies the transpose of this matrix by a vector and stores it 
 * in an output vector.
 * 
 * @see above timesVec function for more detail. This function implicitly transposes
 * this matrix for the multiplication.
 * 
 * The full operation performed is:
 * y = beta * y + alpha * this^T * x
 */
void hiopMatrixRajaSparseTriplet::transTimesVec(double beta, hiopVector& y,
  double alpha, const hiopVector& x) const
{
  assert(x.get_size() == nrows_);
  assert(y.get_size() == ncols_);

  hiopVectorRajaPar& yy = dynamic_cast<hiopVectorRajaPar&>(y);
  const hiopVectorRajaPar& xx = dynamic_cast<const hiopVectorRajaPar&>(x);
  
  double* y_data = yy.local_data();
  const double* x_data = xx.local_data_const();
  
  transTimesVec(beta, y_data, alpha, x_data);
}
 
/**
 * @brief Multiplies the transpose of this matrix by a vector and stores it 
 * in an output vector.
 * 
 * @see above transTimesVec function for more detail. This overload takes raw data
 * pointers rather than hiop constructs.
 * 
 * The full operation performed is:
 * y = beta * y + alpha * this^T * x
 */
void hiopMatrixRajaSparseTriplet::transTimesVec(double beta, double* y,
  double alpha, const double* x ) const
{
  RAJA::forall<hiop_raja_exec>(RAJA::RangeSegment(0, ncols_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      y[i] *= beta;
    });
  
#ifndef NDEBUG
  int num_rows = nrows_;
  int num_cols = ncols_;
#endif
  int* iRow = iRow_;
  int* jCol = jCol_;
  double* values = values_;
  // atomic is needed to prevent data race from ocurring;
  // y[jCol_[i]] can be referenced by multiple threads concurrently
  RAJA::forall<hiop_raja_exec>(RAJA::RangeSegment(0, nnz_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      assert(iRow[i] < num_rows);
      assert(jCol[i] < num_cols);
      RAJA::AtomicRef<double, hiop_raja_atomic> yy(&y[jCol[i]]);
      yy += alpha * x[iRow[i]] * values[i];
    });
}

void hiopMatrixRajaSparseTriplet::timesMat(double beta, hiopMatrix& W, 
				       double alpha, const hiopMatrix& X) const
{
  assert(false && "not needed");
}

void hiopMatrixRajaSparseTriplet::transTimesMat(double beta, hiopMatrix& W, 
					    double alpha, const hiopMatrix& X) const
{
  assert(false && "not needed");
}

/**
 * @brief W = beta*W + alpha*this*X^T
 * Sizes: M1(this) is (m1 x nx) and M2 is (m2, nx).
 */
void hiopMatrixRajaSparseTriplet::
timesMatTrans(double beta, hiopMatrix& Wmat, double alpha, const hiopMatrix& M2mat) const
{
  auto& W = dynamic_cast<hiopMatrixDense&>(Wmat);
  const auto& M2 = dynamic_cast<const hiopMatrixRajaSparseTriplet&>(M2mat);
  const hiopMatrixRajaSparseTriplet& M1 = *this;
  
  const int m1 = M1.nrows_;
  const int nx = M1.ncols_;
  const int m2 = M2.nrows_;
  assert(nx==M1.ncols_);
  assert(nx==M2.ncols_);
  assert(M2.ncols_ == nx);

  assert(m1==W.m());
  assert(m2==W.n());

  int M1nnz = M1.nnz_;
  int M2nnz = M2.nnz_;
    
  //double** WM = W.get_M();
  RAJA::View<double, RAJA::Layout<2>> WM(W.local_data(), W.m(), W.n());

  // TODO: allocAndBuildRowStarts -> should create row_starts_host internally (name='prepareRowStarts' ?)
  if(M1.row_starts_host==NULL)
    M1.row_starts_host = M1.allocAndBuildRowStarts();
  assert(M1.row_starts_host);

  if(M2.row_starts_host==NULL)
    M2.row_starts_host = M2.allocAndBuildRowStarts();
  assert(M2.row_starts_host);

  int* M1_idx_start = M1.row_starts_host->idx_start_;
  int* M2_idx_start = M2.row_starts_host->idx_start_;

  int* M1jCol = M1.jCol_;
  int* M2jCol = M2.jCol_;
  double* M1values = M1.values_;
  double* M2values = M2.values_;

  RAJA::forall<hiop_raja_exec>(RAJA::RangeSegment(0, m1),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      for(int j=0; j<m2; j++)
      {
        // dest[i,j] = weigthed_dotprod(M1_row_i,M2_row_j)
        double acc = 0.;
        int ki=M1_idx_start[i];
        int kj=M2_idx_start[j];
        
        while(ki<M1_idx_start[i+1] && kj<M2_idx_start[j+1])
        {
          assert(ki<M1nnz);
          assert(kj<M2nnz);

          if(M1jCol[ki] == M2jCol[kj])
          {
            acc += M1values[ki] * M2values[kj];
            ki++;
            kj++;
          }
          else if(M1jCol[ki] < M2jCol[kj])
          {
            ki++;
          }
          else
          {
            kj++;
          } 
        } //end of while(ki... && kj...)
        WM(i, j) = beta*WM(i, j) + alpha*acc;
      } //end j
    });
}

void hiopMatrixRajaSparseTriplet::addDiagonal(const double& alpha, const hiopVector& d_)
{
  assert(false && "not needed");
}
void hiopMatrixRajaSparseTriplet::addDiagonal(const double& value)
{
  assert(false && "not needed");
}
void hiopMatrixRajaSparseTriplet::addSubDiagonal(const double& alpha, long long start, const hiopVector& d_)
{
  assert(false && "not needed");
}

void hiopMatrixRajaSparseTriplet::addMatrix(double alpha, const hiopMatrix& X)
{
  assert(false && "not needed");
}

/**
 * @brief Adds the transpose of this matrix to a block within a dense matrix.
 * 
 * @todo Test this function
 * @todo Better document this function
 * 
 * block of W += alpha*transpose(this) 
 * Note W; contains only the upper triangular entries
 */
void hiopMatrixRajaSparseTriplet::transAddToSymDenseMatrixUpperTriangle(int row_start, int col_start, 
  double alpha, hiopMatrixDense& W) const
{
  assert(row_start>=0 && row_start+ncols_<=W.m());
  assert(col_start>=0 && col_start+nrows_<=W.n());
  assert(W.n()==W.m());

  RAJA::View<double, RAJA::Layout<2>> WM(W.local_data(), W.m(), W.n());
  auto Wm = W.m();
  auto Wn = W.n();
  int* iRow = iRow_;
  int* jCol = jCol_;
  double* values = values_;
  RAJA::forall<hiop_raja_exec>(RAJA::RangeSegment(0, nnz_),
    RAJA_LAMBDA(RAJA::Index_type it)
    {
      const int i = jCol[it] + row_start;
      const int j = iRow[it] + col_start;
#ifdef HIOP_DEEPCHECKS
      assert(i < Wm && j < Wn);
      assert(i>=0 && j>=0);
      assert(i<=j && "source entries need to map inside the upper triangular part of destination");
#endif
      WM(i, j) += alpha * values[it];
    });
}

/**
 * @brief Finds the maximum absolute value of the values in this matrix.
 */
double hiopMatrixRajaSparseTriplet::max_abs_value()
{
  double* values = values_;
  RAJA::ReduceMax<hiop_raja_reduce, double> norm(0.0);
  RAJA::forall<hiop_raja_exec>(RAJA::RangeSegment(0, nnz_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      norm.max(fabs(values[i]));
    });
  double maxv = static_cast<double>(norm.get());
  return maxv;
}

/**
 * @brief Returns whether all the values of this matrix are finite or not.
 */
bool hiopMatrixRajaSparseTriplet::isfinite() const
{
#ifdef HIOP_DEEPCHECKS
  assert(this->checkIndexesAreOrdered());
#endif
  double* values = values_;
  RAJA::ReduceSum<hiop_raja_reduce, int> any(0);
  RAJA::forall<hiop_raja_exec>(RAJA::RangeSegment(0, nnz_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      if (!std::isfinite(values[i]))
        any += 1;
    });

  return any.get() == 0;
}

/**
 * @brief Allocates a new hiopMatrixRajaSparseTriplet with the same dimensions
 * and size as this one.
 */
hiopMatrixSparse* hiopMatrixRajaSparseTriplet::alloc_clone() const
{
  return new hiopMatrixRajaSparseTriplet(nrows_, ncols_, nnz_, mem_space_);
}

/**
 * @brief Creates a deep copy of this matrix.
 */
hiopMatrixSparse* hiopMatrixRajaSparseTriplet::new_copy() const
{
#ifdef HIOP_DEEPCHECKS
  assert(this->checkIndexesAreOrdered());
#endif
  hiopMatrixRajaSparseTriplet* copy = new hiopMatrixRajaSparseTriplet(nrows_, ncols_, nnz_, mem_space_);
  auto& resmgr = umpire::ResourceManager::getInstance();
  resmgr.copy(copy->iRow_, iRow_);
  resmgr.copy(copy->jCol_, jCol_);
  resmgr.copy(copy->values_, values_);
  resmgr.copy(copy->iRow_host_, iRow_host_);
  resmgr.copy(copy->jCol_host_, jCol_host_);
  resmgr.copy(copy->values_host_, values_host_);
  return copy;
}

void hiopMatrixRajaSparseTriplet::copyFrom(const hiopMatrixSparse& dm)
{
  assert(false && "this is to be implemented - method def too vague for now");
}

#ifdef HIOP_DEEPCHECKS
/// @brief Ensures the rows and column triplet entries are ordered.
bool hiopMatrixRajaSparseTriplet::checkIndexesAreOrdered() const
{
  copyFromDev();
  if(nnz_==0)
    return true;
  for(int i=1; i<nnz_; i++)
  {
    if(iRow_host_[i] < iRow_host_[i-1])
      return false;
    /* else */
    if(iRow_host_[i] == iRow_host_[i-1])
      if(jCol_host_[i] < jCol_host_[i-1])
        return false;
  }
  return true;
}
#endif

/**
 * @brief This function cannot be described briefly. See below for more detail.
 * 
 * @param rowAndCol_dest_start Starting row & col within _W_ to be added to
 * in the operation.
 * @param alpha Amount to scale this matrix's values by in the operation.
 * @param D The inverse of this vector's values will be multiplied by with this
 * matrix's values in the operation.
 * @param W The output matrix, a block of which's values will be added to in
 * the operation.
 * 
 * @pre rowAndCol_dest_start >= 0
 * @pre rowAndCol_dest_start + this->nrows_ <= W.m()
 * @pre rowAndCol_dest_start + this->nrows_ <= W.n()
 * @pre D.get_size() == this->ncols_
 * 
 * @post A this->nrows_^2 block will be written to in _W_, containing the output
 * of the operation. 
 * 
 * The full operation performed is:
 * diag block of _W_ += _alpha_ * this * _D_^{-1} * transpose(this)
 */
void hiopMatrixRajaSparseTriplet::
addMDinvMtransToDiagBlockOfSymDeMatUTri(int rowAndCol_dest_start,
  const double& alpha, 
  const hiopVector& D, hiopMatrixDense& W) const
{
  const int row_dest_start = rowAndCol_dest_start, col_dest_start = rowAndCol_dest_start;
#ifndef NDEBUG
  int n = this->nrows_;
  int num_non_zero = this->nnz_;
#endif
  assert(row_dest_start>=0 && row_dest_start+n<=W.m());
  assert(col_dest_start>=0 && col_dest_start+nrows_<=W.n());
  assert(D.get_size() == this->ncols_);
  RAJA::View<double, RAJA::Layout<2>> WM(W.local_data(), W.m(), W.n());
  const double* DM = D.local_data_const();
  
  if(row_starts_host==NULL)
    row_starts_host = allocAndBuildRowStarts();
  assert(row_starts_host);

  int num_rows = this->nrows_;
  int* idx_start = row_starts_host->idx_start_;
  int* jCol = jCol_;
  double* values = values_;
  RAJA::forall<hiop_raja_exec>(RAJA::RangeSegment(0, this->nrows_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      //j==i
      double acc = 0.;
      for(int k=idx_start[i]; k<idx_start[i+1]; k++)
      {
        acc += values[k] / DM[jCol[k]] * values[k];
      }
      WM(i + row_dest_start, i + col_dest_start) += alpha*acc;

      //j>i
      for(int j=i+1; j<num_rows; j++)
      {
        //dest[i,j] = weigthed_dotprod(this_row_i,this_row_j)
        acc = 0.;

        int ki=idx_start[i];
        int kj=idx_start[j];
        while(ki<idx_start[i+1] && kj<idx_start[j+1])
        {
          assert(ki < num_non_zero);
          assert(kj < num_non_zero);
          if(jCol[ki] == jCol[kj])
          {
            acc += values[ki] / DM[jCol[ki]] * values[kj];
            ki++;
            kj++;
          }
          else
          {
            if(jCol[ki] < jCol[kj])
              ki++;
            else
              kj++;
          }
        } //end of loop over ki and kj

        WM(i + row_dest_start, j + col_dest_start) += alpha*acc;
      } //end j
    });
}

/**
 * @brief This function cannot be described briefly. See below for more detail.
 * 
 * @param row_dest_start Starting row in destination block.
 * @param col_dest_start Starting col in destination block.
 * @param alpha Amount to scale this matrix by during the operation.
 * @param D The inverse of this vector's values will be multiplied by with this
 * matrix's values in the operation.
 * @param M2mat Another sparse matrix, the transpose of which will be multiplied in
 * the following operation.
 * @param W A dense matrix, a block in which will be used to store the result of 
 * the operation.
 * 
 * @pre this->ncols_ == M2mat.ncols_
 * @pre D.get_size() == this->ncols_
 * @pre row_dest_start >= 0 
 * @pre row_dest_start + this->nrows_ <= W.m()
 * @pre col_dest_start >= 0
 * @pre col_dest_start + M2mat.nrows_ <= W.n()
 * 
 * The full operation performed is:
 * block of _W_ += _alpha_ * this * _D_^{-1} * transpose(_M2mat_)
 * Sizes: M1 is (m1 x nx);  D is vector of len nx, M2 is  (m2, nx).
 */
void hiopMatrixRajaSparseTriplet::
addMDinvNtransToSymDeMatUTri(int row_dest_start, int col_dest_start,
  const double& alpha, 
  const hiopVector& D, const hiopMatrixSparse& M2mat,
  hiopMatrixDense& W) const
{
  const auto& M2 = dynamic_cast<const hiopMatrixRajaSparseTriplet&>(M2mat);
  const hiopMatrixRajaSparseTriplet& M1 = *this;
  
  const int m1 = M1.nrows_;
  const int nx = M1.ncols_;
  const int m2 = M2.nrows_;
  assert(nx==M2.ncols_);
  assert(D.get_size() == nx);

  //does it fit in W ?
  assert(row_dest_start>=0 && row_dest_start+m1<=W.m());
  assert(col_dest_start>=0 && col_dest_start+m2<=W.n());

  //double** WM = W.get_M();
  RAJA::View<double, RAJA::Layout<2>> WM(W.local_data(), W.m(), W.n());
  const double* DM = D.local_data_const();

  // TODO: allocAndBuildRowStarts -> should create row_starts_host internally (name='prepareRowStarts' ?)
  if(M1.row_starts_host==NULL)
    M1.row_starts_host = M1.allocAndBuildRowStarts();
  assert(M1.row_starts_host);

  if(M2.row_starts_host==NULL)
    M2.row_starts_host = M2.allocAndBuildRowStarts();
  assert(M2.row_starts_host);

  int* M1_idx_start = M1.row_starts_host->idx_start_;
  int* M2_idx_start = M2.row_starts_host->idx_start_;
#ifndef NDEBUG
  int M1nnz = M1.nnz_;
  int M2nnz = M2.nnz_;
#endif
  int* M1jCol = M1.jCol_;
  int* M2jCol = M2.jCol_;
  double* M1values = M1.values_;
  double* M2values = M2.values_;
  int* jCol = jCol_;
  RAJA::forall<hiop_raja_exec>(RAJA::RangeSegment(0, m1),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      for(int j=0; j<m2; j++)
      {
        // dest[i,j] = weigthed_dotprod(M1_row_i,M2_row_j)
        double acc = 0.;
        int ki=M1_idx_start[i];
        int kj=M2_idx_start[j];
        
        while(ki<M1_idx_start[i+1] && kj<M2_idx_start[j+1])
        {
          assert(ki<M1nnz);
          assert(kj<M2nnz);

          if(M1jCol[ki] == M2jCol[kj])
          {
            acc += M1values[ki] / DM[jCol[ki]] * M2values[kj];
            ki++;
            kj++;
          }
          else
          {
            if(M1jCol[ki] < M2jCol[kj])
              ki++;
            else
              kj++;
          }
        } //end of loop over ki and kj

#ifdef HIOP_DEEPCHECKS
        if(i+row_dest_start > j+col_dest_start)
          printf("[warning] lower triangular element updated in addMDinvNtransToSymDeMatUTri\n");
        assert(i+row_dest_start <= j+col_dest_start);
#endif
        WM(i+row_dest_start, j+col_dest_start) += alpha*acc;
      } //end j
    });
}


/**
 * @brief Generates a pointer to a single RowStartsInfo struct containing
 * the number of rows and indices at which row data starts from this matrix.
 * 
 * Assumes triplets are ordered.
 */
hiopMatrixRajaSparseTriplet::RowStartsInfo* 
hiopMatrixRajaSparseTriplet::allocAndBuildRowStarts() const
{
  assert(nrows_ >= 0);

  RowStartsInfo* rsi = new RowStartsInfo(nrows_, "HOST"); assert(rsi);
  RowStartsInfo* rsi_dev = new RowStartsInfo(nrows_, mem_space_); assert(rsi_dev);
  if(nrows_<=0)
  {
    delete rsi;
    return rsi_dev;
  }

  // build rsi on the host, then copy it to the device for simplicity
  int it_triplet = 0;
  rsi->idx_start_[0] = 0;
  for(int i = 1; i <= this->nrows_; i++)
  {
    rsi->idx_start_[i]=rsi->idx_start_[i-1];
    
    while(it_triplet < this->nnz_ && this->iRow_host_[it_triplet] == i - 1)
    {
#ifdef HIOP_DEEPCHECKS
      if(it_triplet>=1)
      {
        assert(iRow_host_[it_triplet-1]<=iRow_host_[it_triplet] && "row indices are not sorted");
        //assert(iCol[it_triplet-1]<=iCol[it_triplet]);
        if(iRow_host_[it_triplet-1]==iRow_host_[it_triplet])
          assert(jCol_host_[it_triplet-1] < jCol_host_[it_triplet] && "col indices are not sorted");
      }
#endif
      rsi->idx_start_[i]++;
      it_triplet++;
    }
    assert(rsi->idx_start_[i] == it_triplet);
  }
  assert(it_triplet==this->nnz_);

  auto& rm = umpire::ResourceManager::getInstance();
  rm.copy(rsi_dev->idx_start_, rsi->idx_start_);

  delete rsi;
  return rsi_dev;
}

/**
 * @brief Copies rows from another sparse matrix into this one.
 * 
 * @todo Better document this function.
 */
void hiopMatrixRajaSparseTriplet::copyRowsFrom(const hiopMatrix& src_gen,
					       const long long* rows_idxs,
					       long long n_rows)
{
  const hiopMatrixRajaSparseTriplet& src = dynamic_cast<const hiopMatrixRajaSparseTriplet&>(src_gen);
  assert(this->m() == n_rows);
  assert(this->numberOfNonzeros() <= src.numberOfNonzeros());
  assert(this->n() == src.n());
  assert(n_rows <= src.m());

  const int* iRow_src = src.i_row();
  const int* jCol_src = src.j_col();
  const double* values_src = src.M();
  int nnz_src = src.numberOfNonzeros();
  int itnz_src=0;
  int itnz_dest=0;
  //int iterators should suffice
  for(int row_dest=0; row_dest<n_rows; ++row_dest)
  {
    const int& row_src = rows_idxs[row_dest];

    while(itnz_src<nnz_src && iRow_src[itnz_src]<row_src)
    {
      #ifdef HIOP_DEEPCHECKS
      if(itnz_src>0)
      {
	      assert(iRow_src[itnz_src]>=iRow_src[itnz_src-1] && "row indexes are not sorted");
	      if(iRow_src[itnz_src]==iRow_src[itnz_src-1])
	        assert(jCol_src[itnz_src] >= jCol_src[itnz_src-1] && "col indexes are not sorted");
      }
      #endif
      ++itnz_src;
    }

    while(itnz_src<nnz_src && iRow_src[itnz_src]==row_src)
    {
      assert(itnz_dest < nnz_);
      #ifdef HIOP_DEEPCHECKS
      if(itnz_src>0)
      {
      	assert(iRow_src[itnz_src]>=iRow_src[itnz_src-1] && "row indexes are not sorted");
	      if(iRow_src[itnz_src]==iRow_src[itnz_src-1])
	        assert(jCol_src[itnz_src] >= jCol_src[itnz_src-1] && "col indexes are not sorted");
      }
      #endif
      iRow_[itnz_dest] = row_dest;//iRow_src[itnz_src];
      jCol_[itnz_dest] = jCol_src[itnz_src];
      values_[itnz_dest++] = values_src[itnz_src++];
      
      assert(itnz_dest <= nnz_);
    }
  }
  assert(itnz_dest == nnz_);
}
  
/// @brief Prints the contents of this function to a file.
void hiopMatrixRajaSparseTriplet::print(FILE* file, const char* msg/*=NULL*/, 
				    int maxRows/*=-1*/, int maxCols/*=-1*/, 
				    int rank/*=-1*/) const 
{
  int myrank_=0, numranks=1; //this is a local object => always print
  copyFromDev();

  if(file==NULL) file = stdout;

  int max_elems = maxRows>=0 ? maxRows : nnz_;
  max_elems = std::min(max_elems, nnz_);

  if(myrank_==rank || rank==-1) {

    if(NULL==msg) {
      if(numranks>1)
        fprintf(file, "matrix of size %lld %lld and nonzeros %lld, printing %d elems (on rank=%d)\n", 
		m(), n(), numberOfNonzeros(), max_elems, myrank_);
      else
        fprintf(file, "matrix of size %lld %lld and nonzeros %lld, printing %d elems\n", 
		m(), n(), numberOfNonzeros(), max_elems);
    } else {
      fprintf(file, "%s ", msg);
    }    

    // using matlab indices
    fprintf(file, "iRow_host_=[");
    for(int it=0; it<max_elems; it++)  fprintf(file, "%d; ", iRow_host_[it]+1);
    fprintf(file, "];\n");
    
    fprintf(file, "jCol_host_=[");
    for(int it=0; it<max_elems; it++)  fprintf(file, "%d; ", jCol_host_[it]+1);
    fprintf(file, "];\n");
    
    fprintf(file, "v=[");
    for(int it=0; it<max_elems; it++)  fprintf(file, "%22.16e; ", values_host_[it]);
    fprintf(file, "];\n");
  }
}

/// @brief Copies the data stored in the host mirror to the device.
void hiopMatrixRajaSparseTriplet::copyToDev()
{
  if (mem_space_ == "DEVICE")
  {
    auto& resmgr = umpire::ResourceManager::getInstance();
    resmgr.copy(iRow_, iRow_host_);
    resmgr.copy(jCol_, jCol_host_);
    resmgr.copy(values_, values_host_);
  }
}

/// @brief Copies the data stored on the device to the host mirror.
void hiopMatrixRajaSparseTriplet::copyFromDev() const
{
  if (mem_space_ == "DEVICE")
  {
    auto& resmgr = umpire::ResourceManager::getInstance();
    resmgr.copy(iRow_host_, iRow_);
    resmgr.copy(jCol_host_, jCol_);
    resmgr.copy(values_host_, values_);
  }
}

hiopMatrixRajaSparseTriplet::RowStartsInfo::RowStartsInfo(int n_rows, std::string memspace)
  : num_rows_(n_rows), mem_space_(memspace)
{
  auto& rm = umpire::ResourceManager::getInstance();
  umpire::Allocator alloc = rm.getAllocator(mem_space_);
  idx_start_ = static_cast<int*>(alloc.allocate((num_rows_ + 1) * sizeof(int)));
}

hiopMatrixRajaSparseTriplet::RowStartsInfo::~RowStartsInfo()
{
  auto& rm = umpire::ResourceManager::getInstance();
  umpire::Allocator alloc = rm.getAllocator(mem_space_);
  alloc.deallocate(idx_start_);
}


/**********************************************************************************
  * Sparse symmetric matrix in triplet format. Only the UPPER triangle is stored
  **********************************************************************************/
void hiopMatrixRajaSymSparseTriplet::timesVec(double beta,  hiopVector& y,
					  double alpha, const hiopVector& x ) const
{
  assert(ncols_ == nrows_);
  assert(x.get_size() == ncols_);
  assert(y.get_size() == nrows_);

  auto& yy = dynamic_cast<hiopVectorRajaPar&>(y);
  const auto& xx = dynamic_cast<const hiopVectorRajaPar&>(x);

  double* y_data = yy.local_data();
  const double* x_data = xx.local_data_const();

  timesVec(beta, y_data, alpha, x_data);
}
 
/** y = beta * y + alpha * this * x */
void hiopMatrixRajaSymSparseTriplet::timesVec(double beta,  double* y,
					  double alpha, const double* x) const
{
  assert(ncols_ == nrows_);
  
  RAJA::forall<hiop_raja_exec>(RAJA::RangeSegment(0, nrows_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      y[i] *= beta;
    });

  // addition to y[iRow[i]] must be atomic
  auto iRow = this->iRow_;
  auto jCol = this->jCol_;
  auto values = this->values_;
#ifndef NDEBUG
  auto nrows = this->nrows_;
  auto ncols = this->ncols_;
#endif
  RAJA::forall<hiop_raja_exec>(RAJA::RangeSegment(0, nnz_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      assert(iRow[i] < nrows);
      assert(jCol[i] < ncols);
      RAJA::AtomicRef<double, hiop_raja_atomic> yy1(&y[iRow[i]]);
      yy1 += alpha * x[jCol[i]] * values[i];
      if(iRow[i] != jCol[i])
      {
        RAJA::AtomicRef<double, hiop_raja_atomic> yy2(&y[jCol[i]]);
        yy2 += alpha * x[iRow[i]] * values[i];
      }
    });
}

hiopMatrixSparse* hiopMatrixRajaSymSparseTriplet::alloc_clone() const
{
  assert(nrows_ == ncols_);
  return new hiopMatrixRajaSymSparseTriplet(nrows_, nnz_, mem_space_);
}

hiopMatrixSparse* hiopMatrixRajaSymSparseTriplet::new_copy() const
{
  assert(nrows_ == ncols_);
  auto* copy = new hiopMatrixRajaSymSparseTriplet(nrows_, nnz_, mem_space_);
  auto& resmgr = umpire::ResourceManager::getInstance();
  resmgr.copy(copy->iRow_, iRow_);
  resmgr.copy(copy->jCol_, jCol_);
  resmgr.copy(copy->values_, values_);
  resmgr.copy(copy->iRow_host_, iRow_host_);
  resmgr.copy(copy->jCol_host_, jCol_host_);
  resmgr.copy(copy->values_host_, values_host_);
  return copy;
}

/** 
 * @brief block of W += alpha*this 
 * @note W contains only the upper triangular entries
 */ 
void hiopMatrixRajaSymSparseTriplet::addUpperTriangleToSymDenseMatrixUpperTriangle(
  int diag_start, 
	double alpha,
  hiopMatrixDense& W) const
{
  assert(diag_start>=0 && diag_start+nrows_<=W.m());
  assert(diag_start+ncols_<=W.n());
  assert(W.n()==W.m());

  // double** WM = W.get_M();
  RAJA::View<double, RAJA::Layout<2>> WM(W.local_data(), W.get_local_size_m(), W.get_local_size_n());
  auto Wm = W.m();
  auto Wn = W.n();
  auto iRow = this->iRow_;
  auto jCol = this->jCol_;
  auto values = this->values_;
  RAJA::forall<hiop_raja_exec>(RAJA::RangeSegment(0, nnz_),
    RAJA_LAMBDA(RAJA::Index_type it)
    {
      assert(iRow[it]<=jCol[it] && "sparse symmetric matrices should contain only upper triangular entries");
      const int i = iRow[it]+diag_start;
      const int j = jCol[it]+diag_start;
      assert(i<Wm && j<Wn); assert(i>=0 && j>=0);
      assert(i<=j && "symMatrices not aligned; source entries need to map inside the upper triangular part of destination");
      WM(i, j) += alpha * values[it];
    });
 }

/** 
 * @brief block of W += alpha*(this)^T 
 * @note W contains only the upper triangular entries
 * 
 * @warning This method should not be called directly.
 * Use addUpperTriangleToSymDenseMatrixUpperTriangle instead.
 */
void hiopMatrixRajaSymSparseTriplet::transAddToSymDenseMatrixUpperTriangle(int row_start, int col_start, 
  double alpha, hiopMatrixDense& W) const
{
  assert(0 && "This method should not be called for symmetric matrices.");
}

/* extract subdiagonal from 'this' (source) and adds the entries to 'vec_dest' starting at
 * index 'vec_start'. If num_elems>=0, 'num_elems' are copied; otherwise copies as many as
 * are available in 'vec_dest' starting at 'vec_start'
 */
void hiopMatrixRajaSymSparseTriplet::startingAtAddSubDiagonalToStartingAt(
  int diag_src_start,
  const double& alpha, 
  hiopVector& vec_dest,
  int vec_start,
  int num_elems/*=-1*/) const
{
  auto& vd = dynamic_cast<hiopVectorRajaPar&>(vec_dest);
  if(num_elems < 0)
    num_elems = vd.get_size();
  assert(num_elems<=vd.get_size());

  assert(diag_src_start>=0 && diag_src_start+num_elems<=this->nrows_);
  double* v = vd.local_data();

  auto vds = vd.get_size();
  auto iRow = this->iRow_;
  auto jCol = this->jCol_;
  auto values = this->values_;
  RAJA::forall<hiop_raja_exec>(RAJA::RangeSegment(0, nnz_),
    RAJA_LAMBDA(RAJA::Index_type itnz)
    {
      const int row = iRow[itnz];
      if(row == jCol[itnz])
      {
        if(row >= diag_src_start && row < diag_src_start + num_elems)
        {
          assert(row+vec_start < vds);
          v[vec_start + row] += alpha * values[itnz];
        }
      }
    });
}

} //end of namespace
