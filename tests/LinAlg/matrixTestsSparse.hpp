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
 * @file matrixTestsSparse.hpp
 *
 * @author Asher Mancinelli <asher.mancinelli@pnnl.gov>,  PNNL
 * @author Slaven Peles <slaven.peles@pnnl.gov>, PNNL
 * @author Cameron Rutherford <robert.rutherford@pnnl.gov>, PNNL
 * 
 */

#pragma once

#include <iostream>
#include <functional>

#include <hiopMatrixSparseTriplet.hpp>
#include <hiopVectorPar.hpp>
#include "testBase.hpp"

namespace hiop { namespace tests {

/**
 * @brief Tests are re-implemented here if necessary for SparseTriplet Matrices,
 * as the data layout is significantly different compares to dense matrices.
 * Any tests that would modify the sparsity pattern are not implemented.
 * Any tests that would make calls to non-implemented/needed functions are not implemented.
 * 
*/
class MatrixTestsSparse : public TestBase
{
public:
  MatrixTestsSparse() {}
  virtual ~MatrixTestsSparse(){}

  /// @brief Verify function returning number of rows
  bool matrixNumRows(hiop::hiopMatrix& A, global_ordinal_type M)
  {
    // Method m() returns `global_ordinal_type` even though the matrix is not distributed.
    const bool fail = A.m() == M ? 0 : 1;
    printMessage(fail, __func__);
    return fail;
  }

  /// @brief Verify function returning number of columns
  bool matrixNumCols(hiop::hiopMatrix& A, global_ordinal_type N)
  {
    // Method n() returns `global_ordinal_type` even though the matrix is not distributed.
    const bool fail = A.n() == N ? 0 : 1;
    printMessage(fail, __func__);
    return fail;
  }

  /// Verify function setting matrix elements to zero (depends on `setToConstant`).
  bool matrixSetToZero(hiop::hiopMatrix& A)
  {
    A.setToConstant(one);
    A.setToZero();
    const int fail = verifyAnswer(&A, zero);

    printMessage(fail, __func__);
    return fail;
  }

  /// Test method that sets all structural nonzeros to constant
  bool matrixSetToConstant(hiop::hiopMatrix& A)
  {
    A.setToConstant(zero);
    int fail = verifyAnswer(&A, zero);
    A.setToConstant(two);
    fail += verifyAnswer(&A, two);

    printMessage(fail, __func__);
    return fail;
  }
  
  /// Test y <- beta * y + alpha * A * x
  bool matrixTimesVec(
      hiop::hiopMatrix& matA,
      hiop::hiopVector& y,
      hiop::hiopVector& x)
  {
    auto A = dynamic_cast<hiop::hiopMatrixSparseTriplet*>(&matA);
    assert(y.get_size() == A->m() && "Did you pass in vectors of the correct sizes?");
    assert(x.get_size() == A->n() && "Did you pass in vectors of the correct sizes?");
    const real_type alpha = two,
          beta  = half,
          A_val = half,
          y_val = two,
          x_val = three;
    int fail = 0;

    y.setToConstant(y_val);
    x.setToConstant(x_val);
    A->setToConstant(A_val);
    local_ordinal_type* sparsity_pattern = numNonzerosPerRow(A);

    A->timesVec(beta, y, alpha, x);

    fail += verifyAnswer(&y,
      [=] (local_ordinal_type i)
      {
        const local_ordinal_type numValuesInRow = sparsity_pattern[i];
        return (beta * y_val) + (alpha * A_val * x_val * numValuesInRow);
      });

    delete [] sparsity_pattern;
    printMessage(fail, __func__);
    return fail;
  }

  /// Test: y <- beta * y + alpha * A^T * x
  bool matrixTransTimesVec(
      hiop::hiopMatrix& matA,
      hiop::hiopVector& x,
      hiop::hiopVector& y)
  {
    auto A = dynamic_cast<hiop::hiopMatrixSparseTriplet*>(&matA);
    assert(x.get_size() == A->m() && "Did you pass in vectors of the correct sizes?");
    assert(y.get_size() == A->n() && "Did you pass in vectors of the correct sizes?");
    const real_type alpha = one,
          beta  = one,
          A_val = one,
          y_val = three,
          x_val = three;
    int fail = 0;

    A->setToConstant(A_val);
    y.setToConstant(y_val);
    x.setToConstant(x_val);
    local_ordinal_type* sparsity_pattern = numNonzerosPerCol(A);

    A->transTimesVec(beta, y, alpha, x);

    fail += verifyAnswer(&y,
      [=] (local_ordinal_type i) -> real_type
      {
        return (beta * y_val) + (alpha * A_val * x_val * sparsity_pattern[i]);
      });

    delete [] sparsity_pattern;
    printMessage(fail, __func__);
    return fail;
  }

  /// Test function that returns matrix element with maximum absolute value
  bool matrixMaxAbsValue(
      hiop::hiopMatrix& matA,
      const int rank=0)
  {

    auto A = dynamic_cast<hiop::hiopMatrixSparseTriplet*>(&matA);
    auto nnz = A->numberOfNonzeros();
    auto val = A->M();

    int fail = 0;

    // Positive largest value
    A->setToConstant(zero);
    val[nnz - 1] = one;
    fail += A->max_abs_value() != one;

    // Negative largest value
    A->setToConstant(one);
    val[nnz - 1] = -two;
    fail += A->max_abs_value() != two;

    printMessage(fail, __func__);
    return fail;
  }
  
  /// Test method that checks if matrix elements are finite
  bool matrixIsFinite(hiop::hiopMatrix& matA)
  {

    auto A = dynamic_cast<hiop::hiopMatrixSparseTriplet*>(&matA);
    auto nnz = A->numberOfNonzeros();
    auto val = A->M();

    int fail = 0;

    A->setToConstant(two);
    if (!A->isfinite())
      fail++;

    val[nnz - 1] = INFINITY;
    if (A->isfinite()) 
      fail++;

    printMessage(fail, __func__);
    return fail;
  }

  /**
   * @brief Test for method [W] += A * D^(-1) * A^T
   * 
   * Size of A is m x n; size of D is n x n.
   * The method adds the matrix product to a block above the diagonal of W. 
   * 
   * @param[in] A - sparse matrix object which invokes the method (this)
   * @param[in] D - diagonal matrix stored in a vector
   * @param[in] W - dense matrix where the product is stored
   * @param[in] offset - row/column offset in W, from where A*D^(-1)*A^T is added in place
   */
  int tripletAddMDinvMtransToDiagBlockOfSymDeMatUTri(
    hiop::hiopMatrix& matA,
    hiop::hiopVectorPar& D,
    hiop::hiopMatrixDense& W,
    local_ordinal_type offset)
  {
    auto A = dynamic_cast<hiop::hiopMatrixSparseTriplet*>(&matA);

    int fail = 0;

    // Assertion is using API calls.
    assert(D.get_size() == A->n() && "Did you pass in a diagonal matrix of the correct size?");

    const real_type alpha = half,
          A_val = one,
          d_val = half,
          W_val = zero;

    D.setToConstant(d_val);
    W.setToConstant(W_val);
    A->setToConstant(A_val);

    A->addMDinvMtransToDiagBlockOfSymDeMatUTri(offset, alpha, D, W);

    local_ordinal_type* iRow = A->i_row();
    local_ordinal_type* jCol = A->j_col();
    local_ordinal_type nnz = A->numberOfNonzeros();

    fail += verifyAnswer(&W,
      [=] (local_ordinal_type i, local_ordinal_type j) -> real_type
      {
        // Dense matrix elements that are not modified
        if(i < offset || j < offset || i > j || i >= offset + A->m() || j >= offset + A->m())
        {
          return W_val;
        }
        else 
        {
          // The equivalent indices for the sparse matrices used in the calculation
          local_ordinal_type d_i = i - offset;
          local_ordinal_type d_j = j - offset;

          // Counting the number of columns with entries in rows d_i and d_j
          local_ordinal_type count = 0;

          // Searching for the row index d_i in triplet structure
          local_ordinal_type rs_di = 0;
          while(iRow[rs_di] != d_i && rs_di < nnz)
            ++rs_di;

          // Searching for the row index d_j in triplet structure
          local_ordinal_type rs_dj = 0;
          while(iRow[rs_dj] != d_j && rs_dj < nnz)
            ++rs_dj;

          // Counting nonzero terms of the matrix product innermost loop
          // \sum_k A_ik * A^T_jk / D_kk
          while(iRow[rs_di] == d_i && iRow[rs_dj] == d_j)
          {
            if(jCol[rs_di] == jCol[rs_dj])
            {
              count++;
            }

            if(jCol[rs_di]<jCol[rs_dj])
            {
              rs_di++;
            }
            else
            {
              rs_dj++;
            }
          }
          return W_val + (alpha * A_val * A_val / d_val * count);
        }
      });

    printMessage(fail, __func__);
    return fail;
  }
  
  /**
   * @brief Test for (W) += this * D^(-1) * B^T
   *                    
   * The method adds the matrix product to a block above the diagonal of W.
   * 
   * @param[in] A - sparse matrix object which invokes the method (this)
   * @param[in] B - sparse matrix
   * @param[in] D - diagonal matrix stored in a vector
   * @param[in] W - dense matrix where the product is stored
   * @param[in] i_offset - row offset in W, from where A*D^(-1)*B^T is stored
   * @param[in] j_offset - row offset in W, from where A*D^(-1)*B^T is stored
   */
  bool tripletAddMDinvNtransToSymDeMatUTri(
    hiop::hiopMatrix& matA,
    hiop::hiopMatrixSparseTriplet& B,
    hiop::hiopVectorPar& D,
    hiop::hiopMatrixDense& W,
    local_ordinal_type i_offset,
    local_ordinal_type j_offset)
  {
    auto A = dynamic_cast<hiop::hiopMatrixSparseTriplet*>(&matA);

    int fail = 0;

    assert(D.get_size() == A->n() && "Did you pass in a vector of the correct size?");
    assert(A->n() == B.n() && "Did you pass in matrices with the same number of cols?");

    const real_type alpha = half;
    const real_type A_val = one;
    const real_type B_val = one;
    const real_type d_val = half;
    const real_type W_val = zero;

    D.setToConstant(d_val);
    W.setToConstant(W_val);
    A->setToConstant(A_val);
    B.setToConstant(B_val);

    A->addMDinvNtransToSymDeMatUTri(i_offset, j_offset, alpha, D, B, W);

    // int Mrows = W.m(); std::cout << "M = " << Mrows << "\n";
    // int Ncols = W.n(); std::cout << "N = " << Ncols << "\n";
    // for(int i=0; i<Mrows; ++i)
    // {
    //   for(int j=0; j<Ncols; ++j)
    //   {
    //     std::cout << getLocalElement(&W, i, j) << " ";
    //   }
    //   std::cout << "\n";
    // }
    // std::cout << "\n";

    local_ordinal_type* A_iRow = A->i_row();
    local_ordinal_type* A_jCol = A->j_col();
    local_ordinal_type A_nnz = A->numberOfNonzeros();

    local_ordinal_type* B_iRow = B.i_row();
    local_ordinal_type* B_jCol = B.j_col();
    local_ordinal_type B_nnz = B.numberOfNonzeros();

    local_ordinal_type i_max = i_offset + A->m();
    local_ordinal_type j_max = j_offset + B.m();

    fail += verifyAnswer(&W,
      [=] (local_ordinal_type i, local_ordinal_type j) -> real_type
      {
        // Dense matrix elements that are not modified
        if(i < i_offset || j < j_offset || i > j || i >= i_max || j >= j_max)
        {
          return W_val;
        }
        else 
        {
          // Counting the number of columns with entries in row d_i in A and row d_j in B
          local_ordinal_type count = 0;

          local_ordinal_type d_i = i - i_offset;
          local_ordinal_type d_j = j - j_offset;

          // Searching for the row index d_i in triplet structure
          local_ordinal_type rs_di = 0;
          while(A_iRow[rs_di] != d_i && rs_di < A_nnz)
            rs_di++;
          // Searching for the row index d_j in triplet structure
          local_ordinal_type rs_dj = 0;
          while(B_iRow[rs_dj] != d_j && rs_dj < B_nnz)
            rs_dj++;

          // Counting nonzero terms of the matrix product innermost loop
          // \sum_k A_ik * B^T_jk / D_kk
          while(A_iRow[rs_di] == d_i && B_iRow[rs_dj] == d_j)
          {
            if(A_jCol[rs_di] == B_jCol[rs_dj])
            {
              count++;
            }

            if(A_jCol[rs_di]<B_jCol[rs_dj])
            {
              rs_di++;
            }
            else
            {
              rs_dj++;
            }
          }
          return W_val + (alpha * A_val * B_val / d_val * count);
        }
      });

    printMessage(fail, __func__);
    return fail;
  }
private:
  // TODO: The sparse matrix is not distributed - all is local. 
  // Rename functions to remove redundant "local" from their names?
  virtual void setLocalElement(
      hiop::hiopVector* x,
      const local_ordinal_type i,
      const real_type val) = 0;
  virtual real_type getLocalElement(const hiop::hiopMatrix* a, local_ordinal_type i, local_ordinal_type j) = 0;
  virtual real_type getLocalElement(const hiop::hiopVector* x, local_ordinal_type i) = 0;
  virtual local_ordinal_type getLocalSize(const hiop::hiopVector* x) = 0;
  virtual int verifyAnswer(hiop::hiopMatrix* A, real_type answer) = 0;
  virtual int verifyAnswer(
      hiop::hiopMatrix* A,
      std::function<real_type(local_ordinal_type, local_ordinal_type)> expect) = 0;
  virtual int verifyAnswer(hiop::hiopVector* x, real_type answer) = 0;
  virtual int verifyAnswer(
      hiop::hiopVector* x,
      std::function<real_type(local_ordinal_type)> expect) = 0;
  virtual local_ordinal_type* numNonzerosPerRow(hiop::hiopMatrixSparseTriplet* mat) = 0;
  virtual local_ordinal_type* numNonzerosPerCol(hiop::hiopMatrixSparseTriplet* mat) = 0;
};

}} // namespace hiop::tests
