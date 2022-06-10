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

#pragma once

#include <cstdio>
#include <cassert>
#include "hiopInterface.hpp"
#include "hiopVectorInt.hpp"

namespace hiop
{

class hiopVector
{
public:
  hiopVector()
    : n_(0)
  {
  }
  virtual ~hiopVector() {};
  /// @brief Set all elements to zero.
  virtual void setToZero() = 0;
  /// @brief Set all elements to c
  virtual void setToConstant( double c ) = 0;
  /// @brief Set all elements to random values uniformly distributed between `minv` and `maxv`.  TODO: add unit test
  virtual void set_to_random_uniform(double minv, double maxv) = 0;
  /// @brief Set all elements that are not zero in ix to  c, and the rest to 0
  virtual void setToConstant_w_patternSelect( double c, const hiopVector& ix)=0;

  // TODO: names of copyTo/FromStarting methods are quite confusing 
  //maybe startingAtCopyFromStartingAt startingAtCopyToStartingAt ?
  /// @brief Copy the elements of v
  virtual void copyFrom(const hiopVector& v ) = 0;
  virtual void copyFrom(const double* v_local_data) = 0; //v should be of length at least n_local_
  virtual void copy_from_w_pattern(const hiopVector& src, const hiopVector& select) = 0;
  /// @brief Copy the 'n' elements of v starting at 'start_index_in_this' in 'this'
  virtual void copyFromStarting(int start_index_in_this, const double* v, int n) = 0;
  /// @brief Copy v in 'this' starting at start_index_in_src in  'this'. */
  virtual void copyFromStarting(int start_index_in_src, const hiopVector& v) = 0;
  /// @brief Copy the 'n' elements of v starting at 'start_index_in_v' into 'this'
  virtual void copy_from_starting_at(const double* v, int start_index_in_v, int n) = 0;

  /**
   * @brief Copy from src the elements specified by the indices in index_in_src. 
   *
   * @pre All vectors must reside in the same memory space. 
   * @pre Size of src must be greater or equal than size of this
   * @pre Size of index_in_src must be equal to size of this
   * @pre Elements of index_in_src must be valid (zero-based) indexes in src
   *
   */
  virtual void copy_from_indexes(const hiopVector& src, const hiopVectorInt& index_in_src) = 0;

  /**
   * @brief Copy from src the elements specified by the indices in index_in_src. 
   *
   * @pre All vectors and arrays must reside in the same memory space. 
   * @pre Size of src must be greater or equal than size of this
   * @pre Size of index_in_src must be equal to size of this
   * @pre Elements of index_in_src must be valid (zero-based) indexes in src
   *
   */
  virtual void copy_from_indexes(const double* src, const hiopVectorInt& index_in_src) = 0;
  
  /*
   * @brief Copy from 'v' starting at 'start_idx_src' to 'this' starting at 'start_idx_dest'
   *
   * Elements are copied into 'this' till the end of the 'this' is reached, more exactly a number 
   * of lenght(this) - start_idx_dest elements.
   *
   * Precondition: The method expects that in 'v' there are at least as many elements starting 
   * 'start_idx_src' as 'this' has starting at start_idx_dest, or in other words,
   * length(this) - start_idx_dest <= length(v) - start_idx_src
   */
  virtual void startingAtCopyFromStartingAt(int start_idx_dest, const hiopVector& v, int start_idx_src) = 0;

  /// @brief Copy 'this' to double array, which is assumed to be at least of 'n_local_' size.
  virtual void copyTo(double* dest) const = 0;
  /// @brief Copy 'this' to v starting at start_index in 'this'.
  virtual void copyToStarting(int start_index_in_src, hiopVector& v) const = 0;
  /// @brief Copy 'this' to v starting at start_index in 'v'.
  virtual void copyToStarting(hiopVector& v, int start_index_in_dest) const = 0;
  /// @brief Copy the entries in 'this' where corresponding 'ix' is nonzero, to v starting at start_index in 'v'.
  virtual void copyToStartingAt_w_pattern(hiopVector& v, int start_index_in_dest, const hiopVector& ix) const = 0;
  
  /// @brief Copy the entries in `c` and `d` to `this`, according to the mapping in `c_map` and `d_map`
  virtual void copy_from_two_vec_w_pattern(const hiopVector& c, 
                                           const hiopVectorInt& c_map, 
                                           const hiopVector& d, 
                                           const hiopVectorInt& d_map) = 0;

  /// @brief Copy the entries in `this` to `c` and `d`, according to the mapping `c_map` and `d_map`
  virtual void copy_to_two_vec_w_pattern(hiopVector& c, 
                                         const hiopVectorInt& c_map, 
                                         hiopVector& d, 
                                         const hiopVectorInt& d_map) const = 0;

  /**
   * copy 'this' (source) starting at 'start_idx_in_src' to 'dest' starting at index 'int start_idx_dest' 
   * If num_elems>=0, 'num_elems' will be copied; if num_elems<0, elements will be copied till the end of
   * either source ('this') or destination ('dest') is reached
   * if 'selec_dest' is given, the values are copy to 'dest' where the corresponding entry in 'selec_dest' is nonzero
   */
  virtual void startingAtCopyToStartingAt(int start_idx_in_src, hiopVector& dest, int start_idx_dest, int num_elems=-1) const = 0;
  virtual void startingAtCopyToStartingAt_w_pattern(int start_idx_in_src, hiopVector& dest, int start_idx_dest, const hiopVector& selec_dest, int num_elems=-1) const = 0;

  /** @brief Return the two norm */
  virtual double twonorm() const = 0;
  /** @brief Return the infinity norm */
  virtual double infnorm() const = 0;
  /**
   * @brief Linf norm on single rank
   */
  virtual double infnorm_local() const = 0;
  /** @brief Return the one norm */
  virtual double onenorm() const = 0;
  /**
   * @brief L1 norm on single rank
   */
  virtual double onenorm_local() const = 0;

  /** @brief Multiply the components of this by the components of v. */
  virtual void componentMult( const hiopVector& v ) = 0;
  /** @brief Divide the components of this hiopVector by the components of v. */
  virtual void componentDiv ( const hiopVector& v ) = 0;
  /**
   * @brief Elements of this that corespond to nonzeros in ix are divided by elements of v.
   * The rest of elements of this are set to zero.
   */
  virtual void componentDiv_w_selectPattern( const hiopVector& v, const hiopVector& ix) = 0;

  /** @brief Set each component of this hiopVector to the minimum of itself and the given constant. */
  virtual void component_min(const double constant) = 0;
  /** @brief Set each component of this hiopVector to the minimum of itself and the corresponding component of 'v'. */
  virtual void component_min(const hiopVector& v ) = 0;
  /** @brief Set each component of this hiopVector to the maximum of itself and the given constant. */
  virtual void component_max(const double constant) = 0;
  /** @brief Set each component of this hiopVector to the maximum of itself and the corresponding component of 'v'. */
  virtual void component_max(const hiopVector& v) = 0;
  /** @brief Set each component to its absolute value */
  virtual void component_abs() = 0;
  /** @brief Apply sign function to each component */
  virtual void component_sgn() = 0;
  /** @brief compute sqrt of each component */
  virtual void component_sqrt() = 0;

  /// @brief Scale each element of this  by the constant alpha
  virtual void scale(double alpha) = 0;
  /// @brief this += alpha * x
  virtual void axpy(double alpha, const hiopVector& x) = 0;
  /// @brief this += alpha * x, for the entries in 'this' where corresponding 'select' is nonzero.
  virtual void axpy_w_pattern(double alpha, const hiopVector& x, const hiopVector& select) = 0;

  /**
   * @brief Performs axpy, this += alpha*x, on the indexes in this specified by i.
   * 
   * @param alpha scaling factor 
   * @param x vector of doubles to be axpy-ed to this (size equal to size of i and less than or equal to size of this)
   * @param i vector of indexes in this to which the axpy operation is performed (size equal to size of x and less than 
   * or equal to size of this)
   *
   * @pre The entries of i must be valid (zero-based) indexes in this
   *
   */
  virtual void axpy(double alpha, const hiopVector& x, const hiopVectorInt& i) = 0;
  
  /// @brief this += alpha * x * z
  virtual void axzpy ( double alpha, const hiopVector& x, const hiopVector& z ) = 0;
  /// @brief this += alpha * x / z
  virtual void axdzpy( double alpha, const hiopVector& x, const hiopVector& z ) = 0;
  /// @brief this += alpha * x / z on entries 'i' for which select[i]==1.
  virtual void axdzpy_w_pattern( double alpha, const hiopVector& x, const hiopVector& z,
				 const hiopVector& select ) = 0; 
  /// @brief Add c to the elements of this
  virtual void addConstant( double c ) = 0;
  virtual void addConstant_w_patternSelect(double c, const hiopVector& ix) = 0;
  /// @brief Return the dot product of this hiopVector with v
  virtual double dotProductWith( const hiopVector& v ) const = 0;
  /// @brief Negate all the elements of this
  virtual void negate() = 0;
  /// @brief Invert (1/x) the elements of this
  virtual void invert() = 0;
  /// @brief compute log barrier term, that is sum{ln(x_i):i=1,..,n}
  virtual double logBarrier_local(const hiopVector& select) const = 0;
  /// @brief adds the gradient of the log barrier, namely this=this+alpha*1/select(x)
  virtual void addLogBarrierGrad(double alpha, const hiopVector& x, const hiopVector& select)=0;
  /// @brief compute sum{(x_i):i=1,..,n}
  virtual double sum_local() const = 0;

  /**
   * @brief Computes the log barrier's linear damping term of the Filter-IPM method of 
   * WaectherBiegler (see paper, section 3.7).
   * Essentially compute  kappa_d*mu* \sum { this[i] | ixleft[i]==1 and ixright[i]==0 }
   */
  virtual double linearDampingTerm_local(const hiopVector& ixleft, const hiopVector& ixright, 
					 const double& mu, const double& kappa_d) const=0;

  /** 
   * Performs `this[i] = alpha*this[i] + sign*ct` where sign=1 when EXACTLY one of 
   * ixleft[i] and ixright[i] is 1.0 and sign=0 otherwise. 
   *
   * Supports distributed/MPI vectors, but performs only elementwise operations and do not
   * require communication.
   *
   * This method is used to add gradient contributions from the (linear) damping term used
   * to handle unbounded problems. The damping terms are used for variables that are 
   * bounded on one side only. 
   */
  virtual void addLinearDampingTerm(const hiopVector& ixleft,
                                    const hiopVector& ixright,
                                    const double& alpha,
                                    const double& ct) = 0;

  /// @brief True if all elements of this are positive.
  virtual int allPositive() = 0;
  /// @brief True if elements corresponding to nonzeros in w are all positive
  virtual int allPositive_w_patternSelect(const hiopVector& w) = 0;
  /// @brief Return the minimum value in this vector
  virtual double min() const = 0;
  virtual double min_w_pattern(const hiopVector& select) const = 0;
  /// @brief Return the minimum value in this vector, and the index at which it occurs.
  virtual void min( double& m, int& index ) const = 0;
  /// @brief Project the vector into the bounds, used for shifting the ini pt in the bounds
  virtual bool projectIntoBounds_local(const hiopVector& xl, const hiopVector& ixl, 
				       const hiopVector& xu, const hiopVector& ixu,
				       double kappa1, double kappa2) = 0;
  /// @brief max{a\in(0,1]| x+ad >=(1-tau)x}
  virtual double fractionToTheBdry_local(const hiopVector& dx, const double& tau) const = 0;
  virtual double fractionToTheBdry_w_pattern_local(const hiopVector& dx,
						   const double& tau,
						   const hiopVector& ix) const = 0;
  /// @brief Entries corresponding to zeros in ix are set to zero
  virtual void selectPattern(const hiopVector& ix) = 0;
  /// @brief checks whether entries in this matches pattern in ix
  virtual bool matchesPattern(const hiopVector& ix) = 0;

  /// @brief dual adjustment -> see hiopIterate::adjustDuals_primalLogHessian
  virtual void adjustDuals_plh(const hiopVector& x, const hiopVector& ix,
			       const double& mu, const double& kappa)=0;

  /// @brief True if all elements of this are zero. TODO: add unit test
  virtual bool is_zero() const = 0;
  /// @brief check for nans in the local vector
  virtual bool isnan_local() const = 0;
  /// @brief check for infs in the local vector
  virtual bool isinf_local() const = 0;
  /// @brief check if all values are finite /well-defined floats. Returns false if nan or infs are present.
  virtual bool isfinite_local() const = 0;
  
  /// @brief prints up to max_elems (by default all), on rank 'rank' (by default on all)
  virtual void print(FILE*, const char* message=NULL,int max_elems=-1, int rank=-1) const = 0;
  virtual void print() const {assert(0);} 
  /// @brief allocates a vector that mirrors this, but doesn't copy the values
  virtual hiopVector* alloc_clone() const = 0;
  /// @brief allocates a vector that mirrors this, and copies the values
  virtual hiopVector* new_copy () const = 0;
  virtual size_type get_size() const { return n_; }
  virtual size_type get_local_size() const = 0;
  virtual double* local_data() = 0;
  virtual const double* local_data_const() const = 0;
  virtual double* local_data_host() = 0;
  virtual const double* local_data_host_const() const = 0;

  virtual void copyToDev() = 0;
  virtual void copyFromDev() = 0;
  virtual void copyToDev() const = 0;
  virtual void copyFromDev() const = 0;
  
  /// @brief get number of values that are less than the given value 'val'. TODO: add unit test
  virtual size_type numOfElemsLessThan(const double &val) const = 0;
  /// @brief get number of values whose absolute value are less than the given value 'val'. TODO: add unit test
  virtual size_type numOfElemsAbsLessThan(const double &val) const = 0;  

  /// @brief set int array 'arr', starting at `start` and ending at `end`, to the values in `arr_src` from 'start_src`
  /// TODO: add unit test
  virtual void set_array_from_to(hiopInterfaceBase::NonlinearityType* arr, 
                                 const int start, 
                                 const int end, 
                                 const hiopInterfaceBase::NonlinearityType* arr_src,
                                 const int start_src) const = 0;
  virtual void set_array_from_to(hiopInterfaceBase::NonlinearityType* arr, 
                                 const int start, 
                                 const int end, 
                                 const hiopInterfaceBase::NonlinearityType arr_src) const = 0;

  /// @brief check if `this` vector is identical to `vec`
  virtual bool is_equal(const hiopVector& vec) const = 0;

protected:
  size_type n_; //we assume sequential data
protected:
  /// for internal use only; derived classes may use copy constructor and always allocate data_
  hiopVector(const hiopVector& v)
    : n_(v.n_)
  {
  };
};

}
