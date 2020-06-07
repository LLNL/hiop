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
 * @file vectorTests.hpp
 *
 * @author Asher Mancinelli <asher.mancinelli@pnnl.gov>, PNNL
 * @author Slaven Peles <slaven.peles@pnnl.gov>, PNNL
 *
 */
#pragma once

#include <iostream>
#include <cmath>
#include <cfloat>
#include <assert.h>
#include <limits>
#include <functional>

#include <hiopVector.hpp>
#include "testBase.hpp"

namespace hiop::tests {

/**
 * @brief Collection of tests for abstract hiopVector implementations.
 *
 * This class contains implementation of all vector unit tests and abstract
 * interface for testing utility functions, which are specific to vector
 * implementation.
 *
 * @pre All input vectors should be allocated to the same size and have
 * the same partitioning.
 *
 * @post All tests return `true` on all ranks if the test fails on any rank
 * and return `false` otherwise.
 *
 */
class VectorTests : public TestBase
{
public:
  VectorTests(){}
  virtual ~VectorTests(){}

  /*
   * this[i] = 0
   */
  bool vectorSetToZero(hiop::hiopVector& v, const int rank)
  {
    v.setToConstant(one);

    v.setToZero();

    int fail = verifyAnswer(&v, zero);
    printMessage(fail, __func__, rank);

    return reduceReturn(fail, &v);
  }

  /// Test get_size() method of hiop vector implementation
  bool vectorGetSize(hiop::hiopVector& x, global_ordinal_type answer, const int rank)
  {
    bool fail = (x.get_size() != answer);
    printMessage(fail, __func__, rank);
    return fail;
  }

  /// Test setToConstant method of hiop vector implementation
  bool vectorSetToConstant(hiop::hiopVector& x, const int rank)
  {
    int fail = 0;
    local_ordinal_type N = getLocalSize(&x);

    for(local_ordinal_type i=0; i<N; ++i)
    {
      setLocalElement(&x, i, zero);
    }

    x.setToConstant(one);

    fail = verifyAnswer(&x, one);
    printMessage(fail, __func__, rank);

    return reduceReturn(fail, &x);
  }

  /*
   * \forall n in n_local if (pattern[n] != 0.0) this[n] = x_val
   */
  bool vectorSetToConstant_w_patternSelect(
      hiop::hiopVector& x,
      hiop::hiopVector& pattern,
      const int rank)
  {
    const local_ordinal_type N = getLocalSize(&x);
    assert(N == getLocalSize(&pattern));
    static constexpr real_type x_val = two;

    x.setToConstant(zero);
    pattern.setToConstant(one);

    // Ensure that a single element (globally) is
    // set to zero in the pattern
    if (rank == 0)
      setLocalElement(&pattern, N-1, zero);

    x.setToConstant_w_patternSelect(x_val, pattern);

    // Check that the last element of rank zero's vector is
    // zero, and that x_val was added to all other elements
    const int fail = verifyAnswer(&x,
      [=] (local_ordinal_type i) -> real_type
      {
        return (rank == 0 && i == N-1) ? zero : x_val;
      });

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &x);
  }

  /*
   * Test for function that copies data from x to this.
   */
  bool vectorCopyFrom(hiop::hiopVector& v, hiop::hiopVector& from, const int rank)
  {
    assert(v.get_size() == from.get_size());
    assert(getLocalSize(&v) == getLocalSize(&from));

    from.setToConstant(one);
    v.setToConstant(two);
    v.copyFrom(from);

    int fail = verifyAnswer(&v, one);
    printMessage(fail, __func__, rank);

    return reduceReturn(fail, &v);
  }

  bool vectorCopyFromStarting(
      hiop::hiopVector& x,
      hiop::hiopVector& from,
      const int rank)
  {
    int fail = 0;
    const local_ordinal_type N = getLocalSize(&x);
    assert(N == x.get_size() && "This test cannot be ran with distributed vectors");
    assert(N == getLocalSize(&from));
    x.setToConstant(two);

    auto _from = new real_type[N];
    for (local_ordinal_type i=0; i<N; i++)
      _from[i] = one;

    if (rank == 0)
    {
      x.copyFromStarting(1, _from, N-1);
    }
    else
    {
      x.copyFromStarting(0, _from, N);
    }

    for (local_ordinal_type i=0; i<N; i++)
    {
      if (getLocalElement(&x, i) != one && !(i == 0 && rank == 0))
        fail++;
    }

    x.setToConstant(two);
    from.setToConstant(one);
    x.copyFromStarting(0, from);
    fail += verifyAnswer(&x, one);

    delete[] _from;
    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &x);
  }

  /*
   * Copy from one vector to another, specifying both
   * the start index in the source and the destination.
   */
  bool vectorStartingAtCopyFromStartingAt(
      hiop::hiopVector& x,
      hiop::hiopVector& from,
      const int rank)
  {
    int fail = 0;
    const local_ordinal_type N = getLocalSize(&x);
    assert(N == x.get_size() && "This test cannot be ran with distributed vectors");
    assert(N == getLocalSize(&from));

    const real_type x_val = one;
    const real_type from_val = two;
    const local_ordinal_type start_idx = 1;

    x.setToConstant(x_val);
    from.setToConstant(from_val);

    x.startingAtCopyFromStartingAt(start_idx, from, 0);

    /*
     * Ensure that elements in the vector before the start
     * index remain unchanged, and elements after or equal to the
     * start index are copied to the destination vector
     */
    verifyAnswer(&x,
      [=] (local_ordinal_type i) -> real_type
      {
        return i < start_idx ? x_val : from_val;
      });

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &x);
  }

  /*
   * Test for function that copies data from this to x.
   */
  bool vectorCopyTo(hiop::hiopVector& v, hiop::hiopVector& to, const int rank)
  {
    assert(v.get_size() == to.get_size());
    assert(getLocalSize(&v) == getLocalSize(&to));

    to.setToConstant(one);
    v.setToConstant(two);

    real_type* todata = getLocalData(&to);
    v.copyTo(todata);

    int fail = verifyAnswer(&to, two);
    printMessage(fail, __func__, rank);

    return reduceReturn(fail, &v);
  }

  bool vectorCopyToStarting(
      hiop::hiopVector& to,
      hiop::hiopVector& from,
      const int rank)
  {
    const local_ordinal_type dest_size = getLocalSize(&to);
    const local_ordinal_type src_size = getLocalSize(&from);
    assert(dest_size == to.get_size()
        && "This test cannot be ran with distributed vectors");
    assert(dest_size > src_size
        && "Must pass in a destination vector larger than source vector");

    const int start_idx = dest_size - src_size;
    const real_type from_val = one;
    const real_type to_val = two;

    from.setToConstant(from_val);
    to.setToConstant(to_val);

    from.copyToStarting(to, start_idx);

    /*
     * Test that values at indices less than the start
     * index remain unchanged, and that values at indices
     * greater than or equal to the start idx are set
     * to the source value
     */
    const int fail = verifyAnswer(&to,
      [=] (local_ordinal_type i) -> real_type
      {
        return i < start_idx ? to_val : from_val;
      });

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &from);
  }

  bool vectorStartingAtCopyToStartingAt(
      hiop::hiopVector& to,
      hiop::hiopVector& from,
      const int rank)
  {
    const local_ordinal_type dest_size = getLocalSize(&to);
    const local_ordinal_type src_size = getLocalSize(&from);
    assert(dest_size == to.get_size()
        && "This test cannot be ran with distributed vectors");
    assert(dest_size > src_size
        && "Must pass in a destination vector larger than source vector");

    const int start_idx_src = 1;
    const int start_idx_dst = dest_size - (src_size - start_idx_src);
    const int num_elements_to_copy = src_size - start_idx_src;
    const real_type from_val = one;
    const real_type to_val = two;

    from.setToConstant(from_val);
    to.setToConstant(to_val);

    from.startingAtCopyToStartingAt(
        start_idx_src,
        to,
        start_idx_dst,
        num_elements_to_copy);

    const int fail = verifyAnswer(&to,
      [=] (local_ordinal_type i) -> real_type
      {
        const bool isValueCopied = (i >= start_idx_dst &&
          i < start_idx_dst + num_elements_to_copy);
        return isValueCopied ? from_val : to_val;
      });

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &to);
  }

  /*
   * this[i] = (pattern[i] == 0 ? 0 : this[i])
   */
  bool vectorSelectPattern(
      hiop::hiopVector& x,
      hiop::hiopVector& pattern,
      const int rank)
  {
    const local_ordinal_type N = getLocalSize(&x);
    assert(x.get_size() == pattern.get_size());
    assert(N == getLocalSize(&pattern));
    const real_type x_val = two;

    x.setToConstant(x_val);
    pattern.setToConstant(one);
    if (rank== 0)
      setLocalElement(&pattern, N - 1, zero);

    x.selectPattern(pattern);

    const int fail = verifyAnswer(&x,
      [=] (local_ordinal_type i) -> real_type
      {
        const bool isLastElementOnRank0 = (i == N-1 && rank == 0);
        return isLastElementOnRank0 ? zero : x_val;
      });

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &x);
  }

  /*
   * this[i] *= alpha
   */
  bool vectorScale(hiop::hiopVector& v, const int rank)
  {
    v.setToConstant(half);
    v.scale(half);

    int fail = verifyAnswer(&v, quarter);
    printMessage(fail, __func__, rank);

    return reduceReturn(fail, &v);
  }

  /*
   * this[i] *= x[i]
   */
  bool vectorComponentMult(hiop::hiopVector& v, hiop::hiopVector& x, const int rank)
  {
    assert(v.get_size() == x.get_size());
    assert(getLocalSize(&v) == getLocalSize(&x));

    v.setToConstant(half);
    x.setToConstant(half);

    v.componentMult(x);

    int fail = verifyAnswer(&v, quarter);
    printMessage(fail, __func__, rank);

    return reduceReturn(fail, &v);
  }

  /*
   * this[i] /= x[i]
   */
  bool vectorComponentDiv(hiop::hiopVector& v, hiop::hiopVector& x, const int rank)
  {
    assert(v.get_size() == x.get_size());
    assert(getLocalSize(&v) == getLocalSize(&x));

    v.setToConstant(one);
    x.setToConstant(two);

    v.componentDiv(x);

    int fail = verifyAnswer(&v, half);
    printMessage(fail, __func__, rank);

    return reduceReturn(fail, &v);
  }

  /*
   * this[i] = (pattern[i] == 0 ? 0 : this[i]/x[i])
   */
  bool vectorComponentDiv_p_selectPattern(
      hiop::hiopVector& v,
      hiop::hiopVector& x,
      hiop::hiopVector& pattern,
      const int rank)
  {
    const local_ordinal_type N = getLocalSize(&v);
    assert(v.get_size() == x.get_size());
    assert(v.get_size() == pattern.get_size());
    assert(N == getLocalSize(&x));
    assert(N == getLocalSize(&pattern));
    const real_type x_val = one;
    const real_type v_val = half;

    x.setToConstant(x_val);
    v.setToConstant(v_val);
    pattern.setToConstant(one);
    if (rank== 0)
      setLocalElement(&pattern, N - 1, zero);

    v.componentDiv_p_selectPattern(x, pattern);

    const int fail = verifyAnswer(&v,
      [=] (local_ordinal_type i) -> real_type
      {
        const bool isLastElementOnRank0 = (i == N-1 && rank == 0);
        return isLastElementOnRank0 ? zero : v_val / x_val;
      });

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &v);
  }

  /*
   * Test computing 1-norm ||v||  of vector v
   *                            1
   */
  bool vectorOnenorm(hiop::hiopVector& v, const int rank)
  {
    v.setToConstant(-one);
    real_type actual = v.onenorm();
    real_type expected = static_cast<real_type>(v.get_size());

    int fail = (actual != expected);
    printMessage(fail, __func__, rank);

    return reduceReturn(fail, &v);
  }

  /*
   * Test computing 2-norm ||v||  of vector v
   *                            2
   */
  bool vectorTwonorm(hiop::hiopVector& v, const int rank)
  {
    v.setToConstant(-one);
    real_type actual = v.twonorm();
    const real_type expected = sqrt(static_cast<real_type>(v.get_size()));

    int fail = !isEqual(expected, actual);
    printMessage(fail, __func__, rank);

    return reduceReturn(fail, &v);
  }

  /*
   * Test infinity-norm = max(abs(this[i]))
   *                       i
   */
  bool vectorInfnorm(hiop::hiopVector& v, const int rank)
  {
    const local_ordinal_type N = getLocalSize(&v);
    const real_type expected = two;

    v.setToConstant(one);
    if (rank== 0)
      setLocalElement(&v, N-1, -two);
    real_type actual = v.infnorm();

    int fail = (expected != actual);
    printMessage(fail, __func__, rank);

    return reduceReturn(fail, &v);
  }

  /*
   * this[i] += alpha * x[i]
   */
  bool vectorAxpy(hiop::hiopVector& v, hiop::hiopVector& x, const int rank)
  {
    const local_ordinal_type N = getLocalSize(&v);
    assert(v.get_size() == x.get_size());
    assert(N == getLocalSize(&x));

    const real_type alpha = half;
    const real_type x_val = two;
    const real_type v_val = two;

    x.setToConstant(x_val);
    v.setToConstant(v_val);

    v.axpy(alpha, x);

    const real_type expected = v_val + alpha * x_val;
    int fail = verifyAnswer(&v, expected);
    printMessage(fail, __func__, rank);

    return reduceReturn(fail, &v);
  }

  /*
   * this[i] += alpha * x[i] * z[i]
   */
  bool vectorAxzpy(
      hiop::hiopVector& v,
      hiop::hiopVector& x,
      hiop::hiopVector& z,
      const int rank)
  {
    const local_ordinal_type N = getLocalSize(&v);
    assert(v.get_size() == x.get_size());
    assert(N == getLocalSize(&x));

    const real_type alpha = half;
    const real_type x_val = two;
    const real_type v_val = two;
    const real_type z_val = two;

    x.setToConstant(x_val);
    z.setToConstant(z_val);
    v.setToConstant(v_val);

    v.axzpy(alpha, x, z);

    const real_type expected = v_val + (alpha * x_val * z_val);
    const int fail = verifyAnswer(&v, expected);

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &v);
  }

  /*
   * this[i] += alpha * x[i] / z[i]
   */
  bool vectorAxdzpy(
      hiop::hiopVector& v,
      hiop::hiopVector& x,
      hiop::hiopVector& z,
      const int rank)
  {
    const local_ordinal_type N = getLocalSize(&v);
    assert(v.get_size() == x.get_size());
    assert(N == getLocalSize(&x));

    const real_type alpha = three;
    const real_type x_val = half;
    const real_type v_val = two;
    const real_type z_val = half;

    x.setToConstant(x_val);
    z.setToConstant(z_val);
    v.setToConstant(v_val);

    v.axdzpy(alpha, x, z);

    const real_type expected = v_val + (alpha * x_val / z_val);
    const int fail = verifyAnswer(&v, expected);

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &v);
  }

  /*
   * this += C
   */
  bool vectorAddConstant(hiop::hiopVector& x, const int rank)
  {
    int fail = 0;

    x.setToConstant(zero);
    x.addConstant(two);

    fail = verifyAnswer(&x, two);
    printMessage(fail, __func__, rank);

    return reduceReturn(fail, &x);
  }

  /*
   * if (pattern[i] > 0.0) this[i] += C
   */
  bool vectorAddConstant_w_patternSelect(
      hiop::hiopVector& x,
      hiop::hiopVector& pattern,
      const int rank)
  {
    const local_ordinal_type N = getLocalSize(&x);
    assert(pattern.get_size() == x.get_size());
    assert(N == getLocalSize(&pattern));
    const real_type x_val = half;

    pattern.setToConstant(one);
    if (rank== 0)
      setLocalElement(&pattern, N - 1, zero);

    x.setToConstant(zero);
    x.addConstant_w_patternSelect(x_val, pattern);

    const int fail = verifyAnswer(&x,
      [=] (local_ordinal_type i) -> real_type
      {
        return (rank == 0 && i == N-1) ? zero : x_val;
      });

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &x);
  }

  /*
   * Dot product == \sum{this[i] * other[i]}
   */
  bool vectorDotProductWith(
      hiop::hiopVector& x,
      hiop::hiopVector& y,
      const int rank)
  {
    // Must use global size, as every rank will get global
    const global_ordinal_type N = x.get_size();
    assert(getLocalSize(&x) == getLocalSize(&y));

    x.setToConstant(one);
    y.setToConstant(two);

    const real_type expected = two * (real_type)N;
    const real_type actual = x.dotProductWith(y);
    const bool fail = !isEqual(actual, expected);

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &x);
  }

  /*
   * this[i] == -this_prev[i]
   */
  bool vectorNegate(hiop::hiopVector& x, const int rank)
  {
    x.setToConstant(one);
    x.negate();
    const bool fail = verifyAnswer(&x, -one);
    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &x);
  }

  bool vectorInvert(hiop::hiopVector& x, const int rank)
  {
    x.setToConstant(two);
    x.invert();
    const bool fail = verifyAnswer(&x, half);
    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &x);
  }

  /*
   * sum{ln(x_i):i=1,..,n}
   */
  bool vectorLogBarrier(
      hiop::hiopVector& x,
      hiop::hiopVector& pattern,
      const int rank)
  {
    printMessage(SKIP_TEST, __func__, rank);
    return 0;
    
    const local_ordinal_type N = getLocalSize(&x);
    assert(N == getLocalSize(&pattern));

    // Ensure that only N-1 elements of x are
    // used in the log calculation
    pattern.setToConstant(one);
    setLocalElement(&pattern, N - 1, zero);

    const real_type x_val = half;
    x.setToConstant(x_val);

    // No loops such that the test captures accumulation errors
    const real_type expected = (N-1) * std::log(x_val);
    const real_type result = x.logBarrier(pattern);
    printf("r %f e %f diff %.10e\n",
        result, expected,
        std::abs(result-expected));

    const bool fail = !isEqual(result, expected);

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &x);
  }

  /*
   * this += alpha / pattern(x)
   */
  bool vectorAddLogBarrierGrad(
      hiop::hiopVector& x,
      hiop::hiopVector& y,
      hiop::hiopVector& pattern,
      const int rank)
  {
    const local_ordinal_type N = getLocalSize(&x);
    assert(N == getLocalSize(&pattern));
    assert(N == getLocalSize(&y));
    static constexpr real_type alpha = half;
    static constexpr real_type x_val = two;
    static constexpr real_type y_val = two;

    pattern.setToConstant(one);
    x.setToConstant(x_val);
    y.setToConstant(y_val);

    if (rank == 0)
      setLocalElement(&pattern, N-1, zero);

    x.addLogBarrierGrad(alpha, y, pattern);

    const real_type logBarrierGradVal = x_val + (alpha / y_val);
    const int fail = verifyAnswer(&x,
      [=] (local_ordinal_type i) -> real_type
      {
        const bool isLastElementOnRank0 = (i == N-1 && rank == 0);
        return isLastElementOnRank0 ? x_val : logBarrierGradVal;
      });

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &x);
  }

  /*
   * term := 0.0
   * \forall n \in n_local
   *     if left[n] == 1.0 \land right[n] == 0.0
   *         term += this[n]
   * term *= mu * kappa
   * return term
   */
  bool vectorLinearDampingTerm(
      hiop::hiopVector& x,
      hiop::hiopVector& left,
      hiop::hiopVector& right,
      const int rank)
  {
    const local_ordinal_type N = getLocalSize(&x);
    assert(N == getLocalSize(&left));
    assert(N == getLocalSize(&right));
    static constexpr real_type mu = two;
    static constexpr real_type kappa_d = two;

    x.setToConstant(one);
    left.setToConstant(one);
    right.setToConstant(zero);

    if (rank == 0)
    {
      setLocalElement(&left, N-1, two);
      setLocalElement(&right, N-1, two);
    }

    real_type expected = zero;
    for (local_ordinal_type i=0; i<N-1; ++i)
    {
      expected += one;
    }
    if (rank != 0) expected += one;
    expected *= mu;
    expected *= kappa_d;

    const real_type term = x.linearDampingTerm(left, right, mu, kappa_d);

    const int fail = !isEqual(term, expected);

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &x);
  }

  /*
   * this[i] > 0
   */
  bool vectorAllPositive(hiop::hiopVector& x, const int rank)
  {
    const local_ordinal_type N = getLocalSize(&x);
    int fail = 0;
    x.setToConstant(one);
    if (!x.allPositive())
      fail++;

    x.setToConstant(one);
    if (rank == 0)
      setLocalElement(&x, N-1, -one);
    if (x.allPositive())
      fail++;

    printMessage(fail, __func__, rank);
    return fail;
  }

  /*
   * this[i] > 0 \lor pattern[i] != 1.0
   */
  bool vectorAllPositive_w_patternSelect(
      hiop::hiopVector& x,
      hiop::hiopVector& pattern,
      const int rank)
  {
    const local_ordinal_type N = getLocalSize(&x);
    assert(N == getLocalSize(&pattern));

    int fail = 0;

    x.setToConstant(one);
    pattern.setToConstant(one);
    if (!x.allPositive_w_patternSelect(pattern))
      fail++;

    x.setToConstant(-one);
    if (x.allPositive_w_patternSelect(pattern))
      fail++;

    x.setToConstant(one);
    if (rank == 0)
      setLocalElement(&x, N-1, -one);
    if (x.allPositive_w_patternSelect(pattern))
      fail++;

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &x);
  }

  /*
   * This method is not yet implemented in HIOP
   */
  bool vectorMin(const hiop::hiopVector& x, const int rank)
  {
    (void)x; (void) rank;
    printMessage(SKIP_TEST, __func__, rank);
    return 0;
  }

  /*
   * Project vector into bounds
   */
  bool vectorProjectIntoBounds(
      hiop::hiopVector& x,
      hiop::hiopVector& lower,
      hiop::hiopVector& upper,
      hiop::hiopVector& lower_pattern,
      hiop::hiopVector& upper_pattern,
      const int rank)
  {
    // setup constants and make assertions
    const local_ordinal_type N = getLocalSize(&x);
    assert(N == getLocalSize(&lower));
    assert(N == getLocalSize(&upper));
    assert(N == getLocalSize(&lower_pattern));
    assert(N == getLocalSize(&upper_pattern));
    static constexpr real_type kappa1 = half;
    static constexpr real_type kappa2 = half;
    int fail = 0;

    // Check that lower > upper returns false
    x.setToConstant(one);
    lower.setToConstant(one);
    upper.setToConstant(-one);
    lower_pattern.setToConstant(one);
    upper_pattern.setToConstant(one);
    if (x.projectIntoBounds(
          lower, lower_pattern,
          upper, upper_pattern,
          kappa1, kappa2))
      fail++;

    // check that patterns are correctly applied and
    // x[0] is left at 1
    x.setToConstant(one);
    lower.setToConstant(-one);
    upper.setToConstant(one);
    lower_pattern.setToConstant(one);
    setLocalElement(&lower_pattern, 0, zero);
    upper_pattern.setToConstant(one);
    setLocalElement(&upper_pattern, 0, zero);

    // Call should return true
    fail += !x.projectIntoBounds(
        lower, lower_pattern, upper,
        upper_pattern, kappa1, kappa2);

    // First element should be one
    fail += !isEqual(getLocalElement(&x, 0), one);

    // Testing when x is on a boundary:
    // Check that projection of 1 into (-1, 1)
    // returns `true' and x == half
    x.setToConstant(one);
    lower.setToConstant(-one);
    upper.setToConstant(one);
    lower_pattern.setToConstant(one);
    upper_pattern.setToConstant(one);
    x.projectIntoBounds(
        lower, lower_pattern, upper,
        upper_pattern, kappa1, kappa2);

    // x[i] == 1/2 \forall i \in [1, N)
    fail += verifyAnswer(&x, half);

    // testing when x is below boundaries
    // check that projection of -2 into (0, 2)
    // returns `true' and x == half
    x.setToConstant(-two);
    lower.setToConstant(zero);
    upper.setToConstant(two);
    lower_pattern.setToConstant(one);
    upper_pattern.setToConstant(one);

    // Call should return true
    fail += !x.projectIntoBounds(
        lower, lower_pattern, upper,
        upper_pattern, kappa1, kappa2);

    // x[i] == 1/2 \forall i \in [1, N)
    fail += verifyAnswer(&x, half);

    // testing when x is above boundaries
    // check that projection of -2 into (0, 2)
    // returns `true' and x == half
    x.setToConstant(two);
    lower.setToConstant(-two);
    upper.setToConstant(zero);
    lower_pattern.setToConstant(one);
    upper_pattern.setToConstant(one);

    // Call should return true
    fail += !x.projectIntoBounds(
        lower, lower_pattern, upper,
        upper_pattern, kappa1, kappa2);

    // x[i] == -1/2 \forall i \in [1, N)
    fail += verifyAnswer(&x, -half);

    printMessage(fail, __func__, rank);
    return 0;
  }

  /*
   * fractionToTheBdry psuedocode:
   *
   * \forall dxi \in dx, dxi >= 0 \implies
   *     return 1.0
   *
   * \exists dxi \in dx s.t. dxi < 0 \implies
   *     return_value := 1.0
   *     auxilary := 0.0
   *     \forall n \in n_local
   *         auxilary = compute_step_to_boundary(x[n], dx[n])
   *         if auxilary < return_value
   *             return_value = auxilary
   *     return auxilary
   */
  bool vectorFractionToTheBdry(
      hiop::hiopVector& x,
      hiop::hiopVector& dx,
      const int rank)
  {
    const local_ordinal_type N = getLocalSize(&x);
    assert(N == getLocalSize(&dx));
    static constexpr real_type tau = half;
    int fail = 0;

    x.setToConstant(one);

    dx.setToConstant(one);
    real_type result = x.fractionToTheBdry(dx, tau);

    real_type expected = one;
    fail += !isEqual(result, expected);

    dx.setToConstant(-one);
    result = x.fractionToTheBdry(dx, tau);
    real_type aux;
    expected = one;
    for (local_ordinal_type i=0; i<N; i++)
    {
      aux = -tau * getLocalElement(&x, i) / getLocalElement(&dx, i);
      if (aux<expected) expected=aux;
    }
    fail += !isEqual(result, expected);

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &x);
  }

  /*
   * Same as fractionToTheBdry, except that
   * no x[i] where pattern[i]==0 will be calculated
   */
  bool vectorFractionToTheBdry_w_pattern(
      hiop::hiopVector& x,
      hiop::hiopVector& dx,
      hiop::hiopVector& pattern,
      const int rank)
  {
    const local_ordinal_type N = getLocalSize(&x);
    assert(N == getLocalSize(&dx));
    assert(N == getLocalSize(&pattern));
    static constexpr real_type tau = half;
    int fail = 0;

    // Fraction to boundary is const, so no need to reset x after each test
    x.setToConstant(one);

    // Pattern all ones, X all ones, result should be
    // default (alpha == one)
    pattern.setToConstant(one);
    dx.setToConstant(one);
    real_type result = x.fractionToTheBdry_w_pattern(dx, tau, pattern);
    real_type expected = one;  // default value if dx >= 0
    fail += !isEqual(result, expected);

    // Pattern all ones except for one value, should still be default
    // value of one
    pattern.setToConstant(one);
    if (rank == 0)
      setLocalElement(&pattern, N-1, 0);
    dx.setToConstant(one);
    result = x.fractionToTheBdry_w_pattern(dx, tau, pattern);
    expected = one;  // default value if dx >= 0
    fail += !isEqual(result, expected);

    // Pattern all ones, dx will be <0
    pattern.setToConstant(one);
    dx.setToConstant(-one);
    result = x.fractionToTheBdry_w_pattern(dx, tau, pattern);
    real_type aux;
    expected = one;
    for (local_ordinal_type i=0; i<N; i++)
    {
      if (rank == 0 && i == N-1) continue;
      aux = -tau * getLocalElement(&x, i) / getLocalElement(&dx, i);
      if (aux<expected) expected=aux;
    }
    fail += !isEqual(result, expected);

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &x);
  }

  /*
   *  pattern != 0 \lor this == 0
   */
  bool vectorMatchesPattern(
      hiop::hiopVector& x,
      hiop::hiopVector& pattern,
      const int rank)
  {
    const local_ordinal_type N = getLocalSize(&x);
    assert(N == getLocalSize(&pattern));
    int fail = 0;

    x.setToConstant(one);
    pattern.setToConstant(one);
    if (!x.matchesPattern(pattern)) fail++;

    x.setToConstant(one);
    pattern.setToConstant(one);
    if (rank == 0) setLocalElement(&pattern, N-1, 0);
    if (x.matchesPattern(pattern)) fail++;

    x.setToConstant(one);
    pattern.setToConstant(one);
    if (rank == 0) setLocalElement(&x, N-1, 0);
    if (!x.matchesPattern(pattern)) fail++;

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &x);
  }

  /*
   * Checks that hiop correctly adjusts based on the
   * hessian of the duals function
   */
  bool vectorAdjustDuals_plh(
      hiop::hiopVector& z1,
      hiop::hiopVector& z2,
      hiop::hiopVector& x,
      hiop::hiopVector& pattern,
      const int rank)
  {
    const local_ordinal_type N = getLocalSize(&z1);
    assert(N == getLocalSize(&z2));
    assert(N == getLocalSize(&x));
    assert(N == getLocalSize(&pattern));

    // z1 will adjust duals with it's method
    z1.setToConstant(one);

    // z2's duals will be adjusted by hand
    z2.setToConstant(one);

    x.setToConstant(two);
    pattern.setToConstant(one);

    static constexpr real_type mu = half;
    static constexpr real_type kappa = half;
    z1.adjustDuals_plh(
        x,
        pattern,
        mu,
        kappa);

    real_type a, b;
    for (local_ordinal_type i=0; i<N; i++)
    {
      a = mu / getLocalElement(&x, i);
      b = a / kappa;
      a *= kappa;
      if      (getLocalElement(&x, i) < b)     setLocalElement(&z2, i, b);
      else if (a <= b)                    setLocalElement(&z2, i, b);
      else if (a < getLocalElement(&x, i))     setLocalElement(&z2, i, a);
    }

    // the method's adjustDuals_plh should yield
    // the same result as computing by hand
    int fail = 0;
    for (local_ordinal_type i=0; i<N; i++)
    {
      fail += !isEqual(
          getLocalElement(&z1, i),     // expected
          getLocalElement(&z2, i));    // actual
    }

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &x);
  }

  /*
   * \exists e \in this s.t. isnan(e)
   */
  bool vectorIsnan(hiop::hiopVector& x, const int rank)
  {
    const local_ordinal_type N = getLocalSize(&x);
    int fail = 0;
    x.setToConstant(zero);
    if (x.isnan())
      fail++;

    if (rank == 0)
      setLocalElement(&x, N-1, NAN);
    if (x.isnan() && rank != 0)
      fail++;

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &x);
  }

  /*
   * \exists e \in this s.t. isinf(e)
   */
  bool vectorIsinf(hiop::hiopVector& x, const int rank)
  {
    const local_ordinal_type N = getLocalSize(&x);
    int fail = 0;
    x.setToConstant(zero);
    if (x.isinf())
      fail++;

    if (rank == 0)
      setLocalElement(&x, N-1, INFINITY);
    if (x.isinf() && rank != 0)
      fail++;

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &x);
  }

  /*
   * \forall e \in this, isfinite(e)
   */
  bool vectorIsfinite(hiop::hiopVector& x, const int rank)
  {
    const local_ordinal_type N = getLocalSize(&x);
    int fail = 0;
    x.setToConstant(zero);
    if (!x.isfinite())
      fail++;

    if (rank == 0)
      setLocalElement(&x, N-1, INFINITY);
    if (!x.isfinite() && rank != 0)
      fail++;

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &x);
  }

protected:
  // Interface to methods specific to vector implementation
  virtual void setLocalElement(hiop::hiopVector* x, local_ordinal_type i, real_type val) = 0;
  virtual real_type getLocalElement(const hiop::hiopVector* x, local_ordinal_type i) = 0;
  virtual local_ordinal_type getLocalSize(const hiop::hiopVector* x) = 0;
  virtual real_type* getLocalData(hiop::hiopVector* x) = 0;
  virtual int verifyAnswer(hiop::hiopVector* x, real_type answer) = 0;
  virtual int verifyAnswer(
      hiop::hiopVector* x,
      std::function<real_type(local_ordinal_type)> expect) = 0;
  virtual bool reduceReturn(int failures, hiop::hiopVector* x) = 0;
};

} // namespace hiop::tests
