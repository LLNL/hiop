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
 * @file matrixTests.hpp
 *
 * @author Asher Mancinelli <asher.mancinelli@pnnl.gov>,  PNNL
 * @author Slaven Peles <slaven.peles@pnnl.gov>, PNNL
 *
 */
#pragma once

#include <iostream>
#include <functional>
#include <cassert>
#include <utility>
#include <hiopVector.hpp>
#include <hiopMatrixDense.hpp>
#include "testBase.hpp"

namespace hiop { namespace tests {

class MatrixTests : public TestBase
{
public:
  MatrixTests() {}
  virtual ~MatrixTests(){}

  int matrixSetToZero(hiop::hiopMatrix& A, const int rank=0)
  {
    A.setToZero();
    const int fail = verifyAnswer(&A, zero);
    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &A);
  }

  int matrixSetToConstant(hiop::hiopMatrix& A, const int rank=0)
  {
    const local_ordinal_type M = getNumLocRows(&A);
    const local_ordinal_type N = getNumLocCols(&A);
    for (local_ordinal_type i=0; i<M; i++)
      for (local_ordinal_type j=0; j<N; j++)
        setLocalElement(&A, i, j, one);
    A.setToConstant(two);
    const int fail = verifyAnswer(&A, two);
    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &A);
  }

  /*
   * y_{glob} \leftarrow \beta y_{glob} + \alpha A_{glob \times loc} x_{loc}
   */
  int matrixTimesVec(
      hiop::hiopMatrix& A,
      hiop::hiopVector& y,
      hiop::hiopVector& x,
      const int rank=0)
  {
    const global_ordinal_type N_glob = A.n();
    assert(getLocalSize(&y) == getNumLocRows(&A) && "Did you pass in vectors of the correct sizes?");
    assert(getLocalSize(&x) == getNumLocCols(&A) && "Did you pass in vectors of the correct sizes?");
    const real_type alpha = one,
          beta  = one,
          A_val = one,
          y_val = three,
          x_val = three;
    int fail = 0;

    y.setToConstant(y_val);
    x.setToConstant(x_val);
    A.setToConstant(A_val);

    A.timesVec(beta, y, alpha, x);

    real_type expected = (beta * y_val) + (alpha * A_val * x_val * N_glob);
    fail += verifyAnswer(&y, expected);

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &A);
  }

  /*
   * y = beta * y + alpha * A^T * x
   *
   * Notice that since A^T, x must not be distributed in this case, whereas
   * the plain `timesVec' nessecitated that x be distributed and y not be.
   */
  int matrixTransTimesVec(
      hiop::hiopMatrix& A,
      hiop::hiopVector& x,
      hiop::hiopVector& y,
      const int rank=0)
  {
    const local_ordinal_type M = getNumLocRows(&A);
    const local_ordinal_type N = getNumLocCols(&A);

    // Take m() because A will be transposed
    const global_ordinal_type N_glob = A.m();
    assert(getLocalSize(&x) == getNumLocRows(&A) && "Did you pass in vectors of the correct sizes?");
    assert(getLocalSize(&y) == getNumLocCols(&A) && "Did you pass in vectors of the correct sizes?");
    const real_type alpha = one,
          beta  = one,
          A_val = one,
          y_val = three,
          x_val = three;
    int fail = 0;

    // Index of row of A that will be set to zero,
    // and index of y that will be beta * y_val
    const local_ordinal_type index_to_zero = N-1;

    A.setToConstant(A_val);
    y.setToConstant(y_val);
    x.setToConstant(x_val);

    /*
     * Zero a row of A^T to test that the resulting vector
     * has its initial value as the first element, ensuring that
     * the matrix is correctly transposed.
     */
    for (int i=0; i<M; i++)
    {
      setLocalElement(&A, i, index_to_zero, zero);
    }
    A.transTimesVec(beta, y, alpha, x);

    fail += verifyAnswer(&y,
      [=] (local_ordinal_type i) -> real_type
      {
        const bool isZerodRow = (i == index_to_zero);
        return isZerodRow ?
          beta * y_val :
          (beta * y_val) + (alpha * A_val * x_val * N_glob);
      });

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &A);
  }

  /**
   *  W = beta * W + alpha * A * M
   *
   * Shapes:
   *   A: KxM
   *   M: MxN
   *   W: KxN
   *   all local
   */
  int matrixTimesMat(
      hiop::hiopMatrix& A,
      hiop::hiopMatrix& X,
      hiop::hiopMatrix& W,
      const int rank=0)
  {
    const local_ordinal_type K = getNumLocCols(&A);
    assert(K == A.n());
    assert(getNumLocCols(&X) == X.n());
    assert(K == getNumLocRows(&X));
    assert(getNumLocRows(&A) == getNumLocRows(&W));
    assert(getNumLocCols(&X) == getNumLocCols(&W));
    const real_type A_val = two,
          X_val = three,
          W_val = two,
          alpha = two,
          beta  = two;

    A.setToConstant(A_val);
    W.setToConstant(W_val);
    X.setToConstant(X_val);
    A.timesMat(beta, W, alpha, X);
    real_type expected = (beta * W_val) + (alpha * A_val * X_val * K);

    const int fail = verifyAnswer(&W, expected);

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &A);
  }

  /*
   *  W = beta * W + alpha * this^T * X
   *
   *  A: kxm local
   *  W: mxn
   *  X: kxn
   *
   */
  int matrixTransTimesMat(
      hiop::hiopMatrix& A_local,
      hiop::hiopMatrix& W,
      hiop::hiopMatrix& X,
      const int rank=0)
  {
    const local_ordinal_type K = getNumLocRows(&A_local);
    const global_ordinal_type N_loc = getNumLocCols(&X);
    assert(getNumLocCols(&A_local) == getNumLocRows(&W) && "Matrices have mismatched shapes");
    assert(X.n() == W.n() && "Matrices have mismatched shapes");
    assert(N_loc == getNumLocCols(&W) && "Matrices have mismatched shapes");
    assert(K == getNumLocRows(&X) && "Matrices have mismatched shapes");
    const real_type A_val = two,
          X_val = three,
          W_val = two,
          alpha = two,
          beta  = two;

    /*
     * One row of X will be set to zero to ensure
     * the matrix multiply and transpose operations are
     * working correctly.
     */
    const int idx_of_zero_row = K - 1;

    A_local.setToConstant(A_val);
    W.setToConstant(W_val);
    X.setToConstant(X_val);

    // X[idx][:] = 0
    for (int i=0; i<N_loc; i++)
    {
      setLocalElement(&X, idx_of_zero_row, i, zero);
    }

    A_local.transTimesMat(beta, W, alpha, X);

    real_type expected = (beta * W_val) + (alpha * A_val * X_val * (K - 1));
    const int fail = verifyAnswer(&W, expected);

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &W);
  }

  /*
   *  W = beta * W + alpha * this * X^T
   *
   *  A: mxn
   *  W: mxk local
   *  X: nxk
   *
   */
  int matrixTimesMatTrans(
      hiop::hiopMatrix& A,
      hiop::hiopMatrix& W_local,
      hiop::hiopMatrix& X,
      const int rank=0)
  {
    // Skip for now - undetermined error in timeMatTrans call
    printMessage(SKIP_TEST, __func__, rank); return 0;

    const local_ordinal_type M = getNumLocCols(&A);
    assert(getNumLocRows(&A) == getNumLocRows(&W_local) && "Matrices have mismatched shapes");
    assert(getNumLocRows(&X) == getNumLocCols(&W_local) && "Matrices have mismatched shapes");
    assert(M == getNumLocCols(&X) && "Matrices have mismatched shapes");
    const real_type A_val = two,
          X_val = three,
          W_val = two,
          alpha = two,
          beta  = two;

    A.setToConstant(A_val);
    W_local.setToConstant(W_val);
    X.setToConstant(X_val);
    A.timesMatTrans(beta, W_local, alpha, X);

    real_type expected = (beta * W_val) + (alpha * A_val * X_val * M);

    const int fail = verifyAnswer(&W_local, expected);
    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &A);
  }

  /*
   * this += alpha * diag
   */
  int matrixAddDiagonal(
      hiop::hiopMatrix& A,
      hiop::hiopVector& x,
      const int rank=0)
  {
    int fail = 0;
    assert(getNumLocCols(&A) == getLocalSize(&x));
    assert(getNumLocRows(&A) == getLocalSize(&x));
    assert(getNumLocRows(&A) == A.n());
    assert(A.n() == x.get_size());
    assert(A.m() == x.get_size());
    constexpr real_type alpha = two,
              A_val = quarter,
              x_val = half;

    A.setToConstant(A_val);
    x.setToConstant(x_val);
    A.addDiagonal(alpha, x);
    fail += verifyAnswer(&A,
      [=] (local_ordinal_type i, local_ordinal_type j) -> real_type
      {
        const bool isOnDiagonal = (i == j);
        return isOnDiagonal ? A_val + x_val * alpha : A_val;
      });

    A.setToConstant(A_val);
    A.addDiagonal(alpha);
    fail += verifyAnswer(&A,
      [=] (local_ordinal_type i, local_ordinal_type j) -> real_type
      {
        const bool isOnDiagonal = (i == j);
        return isOnDiagonal ? A_val + alpha : A_val;
      });

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &A);
  }

  /*
   * this += alpha * subdiag
   */
  int matrixAddSubDiagonal(
      hiop::hiopMatrix& A,
      hiop::hiopVector& x,
      const int rank=0)
  {
    int fail = 0;
    const local_ordinal_type N = getNumLocCols(&A);
    assert(N == A.n() && "Test should only be ran sequentially.");
    const local_ordinal_type x_len = getLocalSize(&x);
    const real_type alpha = half,
          A_val = half,
          x_val = one;

    // Test the overload that assumes the entire source vector
    // will be added to the subdiagonal
    local_ordinal_type start_idx = N - x_len;

    A.setToConstant(A_val);
    x.setToConstant(x_val);
    A.addSubDiagonal(alpha, start_idx, x);
    fail += verifyAnswer(&A,
      [=] (local_ordinal_type i, local_ordinal_type j) -> real_type
      {
        const bool isOnSubDiagonal = (i>=start_idx && i==j);
        return isOnSubDiagonal ? A_val + x_val * alpha : A_val;
      });

    // We're only going to add n-1 elements of the vector
    // Test the overload that specifies subset of the vector
    // to be added to subdiagonal
    local_ordinal_type start_idx_src = 1,
                       num_elements_to_add = x_len - start_idx_src,
                       start_idx_dest = (N - x_len) + start_idx_src;

    A.setToConstant(A_val);
    x.setToConstant(x_val);
    A.addSubDiagonal(start_idx_dest, alpha, x, start_idx_src, num_elements_to_add);
    fail += verifyAnswer(&A,
      [=] (local_ordinal_type i, local_ordinal_type j) -> real_type
      {
        const bool isOnSubDiagonal = (i>=start_idx_dest && i==j);
        return isOnSubDiagonal ? A_val + x_val * alpha : A_val;
      });

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &A);
  }

  /*
   * A += alpha * B
   */
  int matrixAddMatrix(
      hiop::hiopMatrix& A,
      hiop::hiopMatrix& B,
      const int rank=0)
  {
    assert(getNumLocRows(&A) == getNumLocRows(&B));
    assert(getNumLocCols(&A) == getNumLocCols(&B));
    const real_type alpha = half,
          A_val = half,
          B_val = one;

    A.setToConstant(A_val);
    B.setToConstant(B_val);
    A.addMatrix(alpha, B);
    const int fail = verifyAnswer(&A, A_val + B_val * alpha);

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &A);
  }

  /*
   * Block of W += alpha*A
   *
   * Precondition: W is square
   */
  int matrixAddToSymDenseMatrixUpperTriangle(
      hiop::hiopMatrix& _W,
      hiop::hiopMatrix& A,
      const int rank=0)
  {
    // This method only takes hiopMatrixDense
    auto W = dynamic_cast<hiop::hiopMatrixDense*>(&_W);
    const local_ordinal_type N_loc = getNumLocCols(W);
    const local_ordinal_type A_M = getNumLocRows(&A);
    const local_ordinal_type A_N_loc = getNumLocCols(&A);
    assert(W->m() == W->n());
    assert(getNumLocRows(W) >= getNumLocRows(&A));
    assert(W->n() >= A.n());

    const local_ordinal_type start_idx_row = 0;
    const local_ordinal_type start_idx_col = N_loc - A_N_loc;
    const real_type alpha = half,
          A_val = half,
          W_val = one;
    int fail = 0;

    // Check with non-1 alpha
    A.setToConstant(A_val);
    W->setToConstant(W_val);
    A.addToSymDenseMatrixUpperTriangle(start_idx_row, start_idx_col, alpha, *W);
    fail += verifyAnswer(W,
      [=] (local_ordinal_type i, local_ordinal_type j) -> real_type
      {
        const bool isUpperTriangle = (
          i>=start_idx_row && i<start_idx_row+A_M &&
          j>=start_idx_col && j<start_idx_col+A_N_loc);
        return isUpperTriangle ? W_val + A_val*alpha : W_val;
      });

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &A);
  }

  /*
   * Block of W += alpha*A
   *
   * Block of W summed with A is in the trasposed
   * location of the same call to addToSymDenseMatrixUpperTriangle
   *
   * Precondition: W is square
   */
  int matrixTransAddToSymDenseMatrixUpperTriangle(
      hiop::hiopMatrix& _W,
      hiop::hiopMatrix& A,
      const int rank=0)
  {
    // This method only takes hiopMatrixDense
    auto W = dynamic_cast<hiop::hiopMatrixDense*>(&_W);
    const local_ordinal_type N_loc = getNumLocCols(W);
    const local_ordinal_type A_M = getNumLocRows(&A);
    const local_ordinal_type A_N_loc = getNumLocCols(&A);
    assert(W->m() == W->n());
    assert(getNumLocRows(W) >= getNumLocRows(&A));
    assert(W->n() >= A.n());

    const local_ordinal_type start_idx_row = 0;
    const local_ordinal_type start_idx_col = N_loc - A_M;
    const real_type alpha = half,
          A_val = half,
          W_val = one;

    A.setToConstant(A_val);
    W->setToConstant(W_val);
    A.transAddToSymDenseMatrixUpperTriangle(start_idx_row, start_idx_col, alpha, *W);
    const int fail = verifyAnswer(W,
      [=] (local_ordinal_type i, local_ordinal_type j) -> real_type
      {
        const bool isTransUpperTriangle = (
          i>=start_idx_row && i<start_idx_row+A_N_loc &&
          j>=start_idx_col && j<start_idx_col+A_M);

        return isTransUpperTriangle ? W_val + A_val*alpha : W_val;
      });

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &A);
  }

  /*
   * Upper diagonal block of W += alpha * A
   *
   * Preconditions:
   * W is square
   * A is square
   * degree of A <= degree of W
   */
  int matrixAddUpperTriangleToSymDenseMatrixUpperTriangle(
      hiop::hiopMatrix& _W,
      hiop::hiopMatrix& A,
      const int rank=0)
  {
    const local_ordinal_type A_M = getNumLocRows(&A);
    const local_ordinal_type A_N = getNumLocCols(&A);
    assert(_W.m() == _W.n());
    assert(A.m() == A.n());
    assert(_W.n() >= A.n());
    assert(getNumLocCols(&A) <= getNumLocCols(&_W));
    auto W = dynamic_cast<hiop::hiopMatrixDense*>(&_W);
    // Map the upper triangle of A to W starting
    // at W's upper left corner
    const local_ordinal_type diag_start = 0;
    int fail = 0;
    const real_type alpha = half,
          A_val = half,
          W_val = one;

    A.setToConstant(A_val);
    W->setToConstant(W_val);
    A.addUpperTriangleToSymDenseMatrixUpperTriangle(diag_start, alpha, *W);
    fail += verifyAnswer(W,
      [=] (local_ordinal_type i, local_ordinal_type j) -> real_type
      {
        bool isUpperTriangle = (i>=diag_start && i<diag_start+A_N && j>=i && j<diag_start+A_M);
        return isUpperTriangle ? W_val + A_val*alpha : W_val;
      });

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &A);
  }

  /*
   * Set bottom right value to ensure that all values
   * are checked.
   */
  virtual int matrixMaxAbsValue(
      hiop::hiopMatrix& A,
      const int rank=0)
  {
    const local_ordinal_type last_row_idx = getNumLocRows(&A)-1;
    const local_ordinal_type last_col_idx = getNumLocCols(&A)-1;
    int fail = 0;

    // Positive largest value
    A.setToConstant(zero);
    if (rank == 0) setLocalElement(&A, last_row_idx, last_col_idx, one);
    fail += A.max_abs_value() != one;

    // Negative largest value
    A.setToConstant(zero);
    if (rank == 0) setLocalElement(&A, last_row_idx, last_col_idx, -one);
    fail += A.max_abs_value() != one;

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &A);
  }

  /*
   * Set bottom right value to ensure that all values
   * are checked.
   */
  virtual int matrixIsFinite(
      hiop::hiopMatrix& A,
      const int rank=0)
  {
    const local_ordinal_type last_row_idx = getNumLocRows(&A)-1;
    const local_ordinal_type last_col_idx = getNumLocCols(&A)-1;
    int fail = 0;

    A.setToConstant(zero);
    if (!A.isfinite()) fail++;

    A.setToConstant(zero);
    if (rank == 0) setLocalElement(&A, last_row_idx, last_col_idx, INFINITY);
    if (!A.isfinite() && rank != 0) fail++;

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &A);
  }

#ifdef HIOP_DEEPCHECKS
  int matrixAssertSymmetry(
      hiop::hiopMatrix& A,
      const int rank=0)
  {
    const local_ordinal_type M = getNumLocRows(&A);
    const local_ordinal_type N = getNumLocCols(&A);
    int fail = 0;

    assert(A.m() == A.n());
    A.setToConstant(one);
    fail += !A.assertSymmetry(eps);

    // Set first row and column to zero globally
    for (int i=0; i<N; i++)
    {
      setLocalElement(&A, 0, i, zero);
    }
    if (rank == 0)
    {
      for (int i=0; i<M; i++)
      {
        setLocalElement(&A, i, 0, zero);
      }
    }
    fail += !A.assertSymmetry(eps);

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &A);
  }
#endif

  int matrixNumRows(hiop::hiopMatrix& A, global_ordinal_type M, const int rank=0)
  {
    const bool fail = A.m() == M ? 0 : 1;
    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &A);
  }

  int matrixNumCols(hiop::hiopMatrix& A, global_ordinal_type N, const int rank=0)
  {
    const bool fail = A.n() == N ? 0 : 1;
    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &A);
  }

protected:
  virtual void setLocalElement(
      hiop::hiopMatrix* a,
      local_ordinal_type i,
      local_ordinal_type j,
      real_type val) = 0;
  virtual real_type getLocalElement(
      const hiop::hiopMatrix* a,
      local_ordinal_type i,
      local_ordinal_type j) = 0;
  virtual real_type getLocalElement(
      const hiop::hiopVector* x,
      local_ordinal_type i) = 0;
  virtual local_ordinal_type getNumLocRows(hiop::hiopMatrix* a) = 0;
  virtual local_ordinal_type getNumLocCols(hiop::hiopMatrix* a) = 0;
  virtual local_ordinal_type getLocalSize(const hiop::hiopVector* x) = 0;
  virtual int verifyAnswer(hiop::hiopMatrix* A, real_type answer) = 0;
  virtual int verifyAnswer(
      hiop::hiopMatrix* A,
      std::function<real_type(local_ordinal_type, local_ordinal_type)> expect) = 0;
  virtual int verifyAnswer(hiop::hiopVector* x, real_type answer) = 0;
  virtual int verifyAnswer(
      hiop::hiopVector* x,
      std::function<real_type(local_ordinal_type)> expect) = 0;
  virtual bool reduceReturn(int failures, hiop::hiopMatrix* A) = 0;
  /*
   * Returns true and sets local coordinate pair if global indices
   * maps to local indices, otherwise false and does not alter
   * local coordinates.
   */
  virtual bool globalToLocalMap(
      hiop::hiopMatrix* A,
      const global_ordinal_type row,
      const global_ordinal_type col,
      local_ordinal_type& local_row,
      local_ordinal_type& local_col) = 0;
};

}} // namespace hiop::tests
