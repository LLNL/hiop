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
 * @author Asher Mancinelli <asher.mancinelli@pnnl.gov>, PNNL
 * @author Slaven Peles <slaven.peles@pnnl.gov>, PNNL
 * @author Cameron Rutherford <robert.rutherford@pnnl.gov>, PNNL
 * @author Jake K. Ryan <jake.ryan@pnnl.gov>, PNNL
 * 
 */

#pragma once

#include <iostream>
#include <functional>

#include <hiopMatrixSparseTriplet.hpp>
#include <hiopMatrixRajaSparseTriplet.hpp>
#include <hiopVectorPar.hpp>
#include "testBase.hpp"

namespace hiop { namespace tests {

/**
 * @brief Tests are re-implemented here if necessary for SparseTriplet Matrices,
 * as the data layout is significantly different compares to dense matrices.
 *
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

  /// @brief Verify function setting matrix elements to zero (depends on `setToConstant`).
  bool matrixSetToZero(hiop::hiopMatrixSparse& A)
  {
    A.setToConstant(one);
    A.setToZero();
    const int fail = verifyAnswer(&A, zero);

    printMessage(fail, __func__);
    return fail;
  }

  /// @brief Test method that sets all structural nonzeros to constant
  bool matrixSetToConstant(hiop::hiopMatrixSparse& A)
  {
    A.setToConstant(zero);
    int fail = verifyAnswer(&A, zero);
    A.setToConstant(two);
    fail += verifyAnswer(&A, two);

    printMessage(fail, __func__);
    return fail;
  }
  
  /// @brief Test y <- beta * y + alpha * A * x
  bool matrixTimesVec(
      hiop::hiopMatrixSparse& A,
      hiop::hiopVector& y,
      hiop::hiopVector& x)
  {
    assert(y.get_size() == A.m() && "Did you pass in vectors of the correct sizes?");
    assert(x.get_size() == A.n() && "Did you pass in vectors of the correct sizes?");
    const real_type alpha = two,
          beta  = half,
          A_val = half,
          y_val = two,
          x_val = three;
    int fail = 0;

    y.setToConstant(y_val);
    x.setToConstant(x_val);
    A.setToConstant(A_val);
    local_ordinal_type* sparsity_pattern = numNonzerosPerRow(&A);

    A.timesVec(beta, y, alpha, x);

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

  /// @brief Test: y <- beta * y + alpha * A^T * x
  bool matrixTransTimesVec(
      hiop::hiopMatrixSparse& A,
      hiop::hiopVector& x,
      hiop::hiopVector& y)
  {
    assert(x.get_size() == A.m() && "Did you pass in vectors of the correct sizes?");
    assert(y.get_size() == A.n() && "Did you pass in vectors of the correct sizes?");
    const real_type alpha = one,
          beta  = one,
          A_val = one,
          y_val = three,
          x_val = three;
    int fail = 0;

    A.setToConstant(A_val);
    y.setToConstant(y_val);
    x.setToConstant(x_val);
    local_ordinal_type* sparsity_pattern = numNonzerosPerCol(&A);

    A.transTimesVec(beta, y, alpha, x);

    fail += verifyAnswer(&y,
      [=] (local_ordinal_type i) -> real_type
      {
        return (beta * y_val) + (alpha * A_val * x_val * sparsity_pattern[i]);
      });

    delete [] sparsity_pattern;
    printMessage(fail, __func__);
    return fail;
  }

  /// @brief Test function that returns matrix element with maximum absolute value
  bool matrixMaxAbsValue(
      hiop::hiopMatrixSparse& A,
      const int rank=0)
  {
    auto nnz = A.numberOfNonzeros();
    auto val = getMatrixData(&A);

    int fail = 0;

    // Positive largest value
    A.setToConstant(zero);
    maybeCopyFromDev(&A);
    val[nnz - 1] = one;
    maybeCopyToDev(&A);
    fail += A.max_abs_value() != one;

    // Negative largest value
    A.setToConstant(one);
    maybeCopyFromDev(&A);
    val[nnz - 1] = -two;
    maybeCopyToDev(&A);
    fail += A.max_abs_value() != two;

    printMessage(fail, __func__);
    return fail;
  }
  
  /// @brief Test method that checks if matrix elements are finite
  bool matrixIsFinite(hiop::hiopMatrixSparse& A)
  {
    auto nnz = A.numberOfNonzeros();
    auto val = getMatrixData(&A);

    int fail = 0;

    A.setToConstant(two);
    if (!A.isfinite())
      fail++;

    val[nnz - 1] = INFINITY;
    maybeCopyToDev(&A);
    if (A.isfinite()) 
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
  int matrixAddMDinvMtransToDiagBlockOfSymDeMatUTri(
    hiop::hiopMatrixSparse& A,
    hiop::hiopVector& D,
    hiop::hiopMatrixDense& W,
    local_ordinal_type offset)
  {
    int fail = 0;

    // Assertion is using API calls.
    assert(D.get_size() == A.n() && "Did you pass in a diagonal matrix of the correct size?");

    const real_type alpha = half,
          A_val = one,
          d_val = half,
          W_val = zero;

    D.setToConstant(d_val);
    W.setToConstant(W_val);
    A.setToConstant(A_val);

    A.addMDinvMtransToDiagBlockOfSymDeMatUTri(offset, alpha, D, W);

    const local_ordinal_type* iRow = getRowIndices(&A);
    const local_ordinal_type* jCol = getColumnIndices(&A);
    const local_ordinal_type nnz = A.numberOfNonzeros();

    fail += verifyAnswer(&W,
      [&] (local_ordinal_type i, local_ordinal_type j) -> real_type
      {
        // Dense matrix elements that are not modified
        if(i < offset || j < offset || i > j || i >= offset + A.m() || j >= offset + A.m())
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
  bool matrixAddMDinvNtransToSymDeMatUTri(
    hiop::hiopMatrixSparse& A,
    hiop::hiopMatrixSparse& B,
    hiop::hiopVector& D,
    hiop::hiopMatrixDense& W,
    local_ordinal_type i_offset,
    local_ordinal_type j_offset)
  {
    int fail = 0;

    assert(D.get_size() == A.n() && "Did you pass in a vector of the correct size?");
    assert(A.n() == B.n() && "Did you pass in matrices with the same number of cols?");

    const real_type alpha = half;
    const real_type A_val = one;
    const real_type B_val = one;
    const real_type d_val = half;
    const real_type W_val = zero;

    D.setToConstant(d_val);
    W.setToConstant(W_val);
    A.setToConstant(A_val);
    B.setToConstant(B_val);

    A.addMDinvNtransToSymDeMatUTri(i_offset, j_offset, alpha, D, B, W);

    const local_ordinal_type* A_iRow = getRowIndices(&A);
    const local_ordinal_type* A_jCol = getColumnIndices(&A);
    const local_ordinal_type A_nnz = A.numberOfNonzeros();

    const local_ordinal_type* B_iRow = getRowIndices(&B);
    const local_ordinal_type* B_jCol = getColumnIndices(&B);
    const local_ordinal_type B_nnz = B.numberOfNonzeros();

    local_ordinal_type i_max = i_offset + A.m();
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

  /**
   * Block of W += alpha*A
   *
   * Precondition: W is square
   * 
   * @todo Change parameter _A_ to be of abstract class hiopMatrixSymSparse
   * as soon as this interface exists.
   */
  bool symAddToSymDenseMatrixUpperTriangle(
    hiop::hiopMatrixDense& W,
    hiop::hiopMatrixSparse& A, // sym sparse matrix
    const int rank=0)
  {
    const local_ordinal_type N_loc = W.get_local_size_n();
    const local_ordinal_type A_M = A.m();
    const local_ordinal_type A_N_loc = A.n();
    assert(W.m() == W.n());
    assert(W.m() >= A.m());
    assert(W.n() >= A.n());

    const local_ordinal_type start_idx_row = 0;
    const local_ordinal_type start_idx_col = N_loc - A_N_loc;
    const real_type alpha = half,
          A_val = half,
          W_val = one;
    int fail = 0;

    // Check with non-1 alpha
    A.setToConstant(A_val);
    W.setToConstant(W_val);
    A.addToSymDenseMatrixUpperTriangle(start_idx_row, start_idx_col, alpha, W);
    
    // get sparsity pattern
    const auto* iRow = getRowIndices(&A);
    const auto* jCol = getColumnIndices(&A);
    auto nnz = A.numberOfNonzeros();
    fail += verifyAnswer(&W,
      [=] (local_ordinal_type i, local_ordinal_type j) -> real_type
      {
        // check if (i, j) within bounds of A
        // then check if (i, j) within upper triangle of W
        const bool isUpperTriangle = ( 
          i>=start_idx_row && i<start_idx_row+A_M &&
          j>=start_idx_col && j<start_idx_col+A_N_loc &&
          j >= i);

        // only nonzero entries in A will be added
        const bool indexExists = find_unsorted_pair(i, j, iRow, jCol, nnz);
        real_type ans = (isUpperTriangle && indexExists) ? W_val + A_val*alpha : W_val; // 1 + .5 * .5 = 1.25
        return ans;
      });

    printMessage(fail, __func__, rank);
    return fail;
  }

  /**
   * Block of W += alpha*A
   *
   * Block of W summed with A is in the trasposed
   * location of the same call to addToSymDenseMatrixUpperTriangle
   *
   * Precondition: W is square
   * 
   * @todo Remove implementations specific code from this test!!!
   * @todo Format documentation correctly
   */
  bool symTransAddToSymDenseMatrixUpperTriangle(
    hiop::hiopMatrixDense& W,
    hiop::hiopMatrixSparse& A,
    const int rank=0)
  {
    const local_ordinal_type N_loc = W.get_local_size_n();
    const local_ordinal_type A_M = A.m();
    const local_ordinal_type A_N_loc = A.n();
    assert(W.m() == W.n());
    assert(W.m() >= A.m());
    assert(W.n() >= A.n());

    const local_ordinal_type start_idx_row = 0;
    const local_ordinal_type start_idx_col = N_loc - A_M;
    const real_type alpha = half,
          A_val = half,
          W_val = one;

    A.setToConstant(A_val);
    W.setToConstant(W_val);

    A.transAddToSymDenseMatrixUpperTriangle(start_idx_row, start_idx_col, alpha, W);

    // get sparsity pattern
    const auto* iRow = getRowIndices(&A);
    const auto* jCol = getColumnIndices(&A);
    auto nnz = A.numberOfNonzeros();
    const int fail = verifyAnswer(&W,
      [=] (local_ordinal_type i, local_ordinal_type j) -> real_type
      {
        const bool isTransUpperTriangle = (
          i>=start_idx_row && i<start_idx_row+A_N_loc && // iCol is in A
          j>=start_idx_col && j<start_idx_col+A_M &&     // jRow is in A
          j <= i);                                       // (i, j) are in upper triangle of W^T

        const bool indexExists = find_unsorted_pair(j, i, iRow, jCol, nnz);
        return (isTransUpperTriangle && indexExists) ? W_val + A_val*alpha : W_val;
      });

    printMessage(fail, __func__, rank);
    return fail;
  }

  /**
  * @brief Test for the method block of W += alpha*this, where `this' is sparse 
  * The block of W is in the upper triangular part 
  * @remark W; contains only the upper triangular entries as it is symmetric
  * This test doesn't test if W itself is symmetric
  * (i,j) are the indices of the upper triangle of W
  */

  bool addToSymDenseMatrixUpperTriangle(
    hiop::hiopMatrixDense& W,
    hiop::hiopMatrixSparse& A,
    const int rank=0)
  {
    const local_ordinal_type N_loc = W.get_local_size_n();
    const local_ordinal_type A_M = A.m();
    const local_ordinal_type A_N_loc = A.n();
    assert(W.m() == W.n());
    assert(W.m() >= A.m());
    assert(W.n() >= A.n());

    const local_ordinal_type start_idx_row = 0;
    const local_ordinal_type start_idx_col = N_loc - A_N_loc;
    const real_type alpha = half,
          A_val = half,
          W_val = one;

    A.setToConstant(A_val);
    W.setToConstant(W_val);

    A.addToSymDenseMatrixUpperTriangle(start_idx_row, start_idx_col, alpha, W);

    // get sparsity pattern
    const local_ordinal_type* iRow = getRowIndices(&A);
    const local_ordinal_type* jCol = getColumnIndices(&A);
    //const auto* iRow = A.i_row();
    //const auto* jCol = A.j_col();
    auto nnz = A.numberOfNonzeros();
    const int fail = verifyAnswer(&W,
      [=] (local_ordinal_type i, local_ordinal_type j) -> real_type
      {
        const bool isUpperTriangle = (
          i>=start_idx_row && i<start_idx_row+A_M && 
          j>=start_idx_col && j<start_idx_col+A_N_loc &&     
          i <= j);                                       

        const bool indexExists = find_unsorted_pair(i-start_idx_row, j-start_idx_col, iRow, jCol, nnz);
        return (isUpperTriangle && indexExists) ? W_val + A_val*alpha : W_val;
      });

    printMessage(fail, __func__, rank);
    return fail;
  }

  /**
  * @brief Test for method block of W += alpha*transpose(this), where `this' is sparse.
  * The block of W is in the upper triangular part 
  * @remark W; contains only the upper triangular entries as it is symmetric
  * This test doesn't test if W itself is symmetric
  * (i,j) are the indices of the upper triangle of W
  */

  bool transAddToSymDenseMatrixUpperTriangle(
    hiop::hiopMatrixDense& W,
    hiop::hiopMatrixSparse& A,
    const int rank=0)
  {
    const local_ordinal_type N_loc = W.get_local_size_n();
    const local_ordinal_type A_M = A.m();
    const local_ordinal_type A_N_loc = A.n();
    assert(W.m() == W.n());
    assert(W.m() >= A.n());
    assert(W.n() >= A.m());

    const local_ordinal_type start_idx_row = 0;
    const local_ordinal_type start_idx_col = N_loc - A_M;
    const real_type alpha = half,
          A_val = half,
          W_val = one;

    A.setToConstant(A_val);
    W.setToConstant(W_val);

    A.transAddToSymDenseMatrixUpperTriangle(start_idx_row, start_idx_col, alpha, W);

    // get sparsity pattern
    const local_ordinal_type* iRow = getRowIndices(&A);
    const local_ordinal_type* jCol = getColumnIndices(&A);
    auto nnz = A.numberOfNonzeros();
    const int fail = verifyAnswer(&W,
      [=] (local_ordinal_type i, local_ordinal_type j) -> real_type
      {
        const bool isTransUpperTriangle = (
          i>=start_idx_row && i<start_idx_row+A_N_loc && 
          j>=start_idx_col && j<start_idx_col+A_M &&     
          i <= j);                                       

        const bool indexExists = find_unsorted_pair(i-start_idx_row, j-start_idx_col, jCol, iRow, nnz);
        return (isTransUpperTriangle && indexExists) ? W_val + A_val*alpha : W_val;
      });

    printMessage(fail, __func__, rank);
    return fail;
  }

  /*
   * Upper diagonal block of W += alpha * A
   *
   * Preconditions:
   * W is square
   * A is square
   * degree of A <= degree of W
   */
  int addUpperTriangleToSymDenseMatrixUpperTriangle(
      hiop::hiopMatrixDense& W,
      hiop::hiopMatrixSparse& A,
      const int rank=0)
  {
    const local_ordinal_type N_loc = W.get_local_size_n();
    const local_ordinal_type A_M = A.m();
    const local_ordinal_type A_N = A.n();
    assert(W.m() == W.n());
    assert(A.m() == A.n());
    assert(W.m() >= A.n());
    assert(W.n() >= A.m());
    //auto W = dynamic_cast<hiop::hiopMatrixDense*>(&_W);
    // Map the upper triangle of A to W starting
    // at W's upper left corner
    const local_ordinal_type diag_start = 0;
    int fail = 0;
    const real_type alpha = half,
          A_val = half,
          W_val = one;

    A.setToConstant(A_val);
    W.setToConstant(W_val);

    A.addUpperTriangleToSymDenseMatrixUpperTriangle(diag_start, alpha, W);

    const auto* iRow = getRowIndices(&A);
    const auto* jCol = getColumnIndices(&A);
    auto nnz = A.numberOfNonzeros();

    fail += verifyAnswer(&W,
      [=] (local_ordinal_type i, local_ordinal_type j) -> real_type
      {
        bool isUpperTriangle = (i>=diag_start && i<diag_start+A_M && j>=i && j<diag_start+A_N);
        const bool indexExists = find_unsorted_pair(i-diag_start, j-diag_start, iRow, jCol, nnz);
        return (isUpperTriangle && indexExists) ? W_val + A_val*alpha : W_val;
      });

    printMessage(fail, __func__, rank);
    return fail;
  }



  /// @todo add implementation of `startingAtAddSubDiagonalToStartingAt`
  /// for abstract sparse matrix interface and all sparse matrix classes, 
  /// then remove this dynamic cast to make the test implementation-agnostic.
  bool symStartingAtAddSubDiagonalToStartingAt(
    hiop::hiopVector& W,
    hiop::hiopMatrixSparse& Amat,
    const int rank = 0)
  {
    auto& A = dynamic_cast<hiop::hiopMatrixRajaSymSparseTriplet&>(Amat);
    assert(W.get_size() == A.m()); // A is square
    
    const auto start_src_idx = 0;
    const auto start_dest_idx = 0;
    const auto num_elems = W.get_size();
    const auto A_val = half;
    const auto W_val = one;
    const auto alpha = half;

    A.setToConstant(A_val);
    W.setToConstant(W_val);
    A.startingAtAddSubDiagonalToStartingAt(start_src_idx, alpha, W, start_dest_idx, num_elems);

    const auto* iRow = getRowIndices(&A);
    const auto* jCol = getColumnIndices(&A);
    auto nnz = A.numberOfNonzeros();
    const auto fail = verifyAnswer(&W, 
      [=](local_ordinal_type i) -> real_type
      {
        const bool indexExists = find_unsorted_pair(i, i, iRow, jCol, nnz);
        return (indexExists) ? (W_val + A_val * alpha) : W_val;
      });
    
    printMessage(fail, __func__, rank);
    return fail;
  }

private:
  /// TODO: The sparse matrix is not distributed - all is local. 
  // Rename functions to remove redundant "local" from their names?
  virtual void setLocalElement(
      hiop::hiopVector* x,
      const local_ordinal_type i,
      const real_type val) = 0;
  virtual real_type getLocalElement(const hiop::hiopMatrix* a, local_ordinal_type i, local_ordinal_type j) = 0;
  virtual real_type getLocalElement(const hiop::hiopVector* x, local_ordinal_type i) = 0;
  virtual real_type* getMatrixData(hiop::hiopMatrixSparse* a) = 0;
  virtual const local_ordinal_type* getRowIndices(const hiop::hiopMatrixSparse* a) = 0;
  virtual const local_ordinal_type* getColumnIndices(const hiop::hiopMatrixSparse* a) = 0;
  virtual local_ordinal_type getLocalSize(const hiop::hiopVector* x) = 0;
  virtual int verifyAnswer(hiop::hiopMatrixSparse* A, real_type answer) = 0;
  virtual int verifyAnswer(
      hiop::hiopMatrixDense* A,
      std::function<real_type(local_ordinal_type, local_ordinal_type)> expect) = 0;
  virtual int verifyAnswer(hiop::hiopVector* x, real_type answer) = 0;
  virtual int verifyAnswer(
      hiop::hiopVector* x,
      std::function<real_type(local_ordinal_type)> expect) = 0;
  virtual local_ordinal_type* numNonzerosPerRow(hiop::hiopMatrixSparse* mat) = 0;
  virtual local_ordinal_type* numNonzerosPerCol(hiop::hiopMatrixSparse* mat) = 0;
  virtual void maybeCopyToDev(hiop::hiopMatrixSparse*) = 0;
  virtual void maybeCopyFromDev(hiop::hiopMatrixSparse*) = 0;

public:
  /**
   * @brief Initialize sparse matrix with a homogeneous pattern to test a
   * realistic use-case.
   */
  virtual void initializeMatrix(hiop::hiopMatrixSparse* mat, local_ordinal_type entries_per_row) = 0;

private:
  // linearly scans an unsorted array
  static bool find_unsorted_pair(int valA, int valB, const int* arrA, const int* arrB, size_t arrslen)
  {
    for (int i = 0; i < arrslen; i++)
    {
      if (arrA[i] == valA && arrB[i] == valB)
      {
        return true;
      }
    }
    return false;
  }
};

}} // namespace hiop::tests
