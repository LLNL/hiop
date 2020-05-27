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

#ifndef HIOP_VECTOR
#define HIOP_VECTOR

#include "hiop_defs.hpp"

#ifdef HIOP_USE_MPI
#include "mpi.h"
#else 

#ifndef MPI_COMM
#define MPI_Comm int
#endif
#ifndef MPI_COMM_NULL
#define MPI_COMM_NULL 0
#endif
#include <cstddef>

#endif 

#include <cstdio>

namespace hiop
{

class hiopVector
{
public:
  hiopVector() { n=0;}
  virtual ~hiopVector() {};
  /** Set all elements to zero. */
  virtual void setToZero() = 0;
  /** Set all elements  to  c */
  virtual void setToConstant( double c ) = 0;
  /** Set all elements that are not zero in ix to  c, and the rest to 0 */
  virtual void setToConstant_w_patternSelect( double c, const hiopVector& ix)=0;
  //TO DO: names of copyTo/FromStarting methods are quite confusing 
  //maybe startingAtCopyFromStartingAt startingAtCopyToStartingAt ?
  /** Copy the elements of v */
  virtual void copyFrom(const hiopVector& v ) = 0;
  /** Copy the 'n' elements of v starting at 'start_index_in_src' in 'this' */
  virtual void copyFromStarting(int start_index_in_src, const double* v, int n) = 0;
  /* Copy v in 'this' starting at start_index_in_src in  'this'. */
  virtual void copyFromStarting(int start_index_in_src, const hiopVector& v) = 0;

  /* copy from 'dest' starting at 'start_idx_dest' to 'this' starting at 'start_idx_src' */
   
  virtual void startingAtCopyFromStartingAt(int start_idx_src, const hiopVector& dest, int start_idx_dest) = 0;

  /* Copy 'this' to double array, which is assumed to be at least of 'n_local' size.*/
  virtual void copyTo(double* dest) const = 0;
  /* Copy 'this' to v starting at start_index in 'this'. */
  virtual void copyToStarting(int start_index_in_src, hiopVector& v) = 0;
  /* Copy 'this' to v starting at start_index in 'v'. */
  virtual void copyToStarting(hiopVector& v, int start_index_in_dest) = 0;

  /* copy 'this' (source) starting at 'start_idx_in_src' to 'dest' starting at index 'int start_idx_dest' 
   * If num_elems>=0, 'num_elems' will be copied; if num_elems<0, elements will be copied till the end of
   * either source ('this') or destination ('dest') is reached */
  virtual void startingAtCopyToStartingAt(int start_idx_in_src, hiopVector& dest, int start_idx_dest, int num_elems=-1) const = 0;

  /** Return the two norm */
  virtual double twonorm() const = 0;
  /** Return the infinity norm */
  virtual double infnorm() const = 0;
  /** Return the one norm */
  virtual double onenorm() const = 0;
  /** Multiply the components of this by the components of v. */
  virtual void componentMult( const hiopVector& v ) = 0;
  /** Divide the components of this hiopVector by the components of v. */
  virtual void componentDiv ( const hiopVector& v ) = 0;
  /* Elements of this that corespond to nonzeros in ix are divided by elements of v.
   * The rest of elements of this are set to zero.
   */
  virtual void componentDiv_p_selectPattern( const hiopVector& v, const hiopVector& ix) = 0;
  /** Scale each element of this  by the constant alpha */
  virtual void scale( double alpha ) = 0;
  /** this += alpha * x */
  virtual void axpy  ( double alpha, const hiopVector& x ) = 0;
  /** this += alpha * x * z */
  virtual void axzpy ( double alpha, const hiopVector& x, const hiopVector& z ) = 0;
  /** this += alpha * x / z */
  virtual void axdzpy( double alpha, const hiopVector& x, const hiopVector& z ) = 0;
  /** this += alpha * x / z on entries 'i' for which select[i]==1. */
  virtual void axdzpy_w_pattern( double alpha, const hiopVector& x, const hiopVector& z, const hiopVector& select ) = 0; 
  /** Add c to the elements of this */
  virtual void addConstant( double c ) = 0;
  virtual void addConstant_w_patternSelect(double c, const hiopVector& ix) = 0;
  /** Return the dot product of this hiopVector with v */
  virtual double dotProductWith( const hiopVector& v ) const = 0;
  /** Negate all the elements of this */
  virtual void negate() = 0;
  /** Invert (1/x) the elements of this */
  virtual void invert() = 0;
  /* compute log barrier term, that is sum{ln(x_i):i=1,..,n} */
  virtual double logBarrier(const hiopVector& select) const = 0;
  /* adds the gradient of the log barrier, namely this=this+alpha*1/select(x) */
  virtual void addLogBarrierGrad(double alpha, const hiopVector& x, const hiopVector& select)=0;

  /* computes the log barrier's linear damping term of the Filter-IPM method of WaectherBiegler (see paper, section 3.7).
   * Essentially compute  kappa_d*mu* \sum { this[i] | ixleft[i]==1 and ixright[i]==0 } */
  virtual double linearDampingTerm(const hiopVector& ixleft, const hiopVector& ixright, 
				   const double& mu, const double& kappa_d)const=0;
  /** True if all elements of this are positive. */
  virtual int allPositive() = 0;
  /** True if elements corresponding to nonzeros in w are all positive */
  virtual int allPositive_w_patternSelect(const hiopVector& w) = 0;
  /** Return the minimum value in this vector, and the index at which it occurs. */
  virtual void min( double& m, int& index ) const = 0;
  /** Project the vector into the bounds, used for shifting the ini pt in the bounds */
  virtual bool projectIntoBounds(const hiopVector& xl, const hiopVector& ixl, 
				 const hiopVector& xu, const hiopVector& ixu,
				 double kappa1, double kappa2) = 0;
  /* max{a\in(0,1]| x+ad >=(1-tau)x} */
  virtual double fractionToTheBdry(const hiopVector& dx, const double& tau) const = 0;
  virtual double fractionToTheBdry_w_pattern(const hiopVector& dx, const double& tau, const hiopVector& ix) const = 0;
  /** Entries corresponding to zeros in ix are set to zero */
  virtual void selectPattern(const hiopVector& ix) = 0;
  /** checks whether entries in this matches pattern in ix */
  virtual bool matchesPattern(const hiopVector& ix) = 0;

  /** allocates a vector that mirrors this, but doesn't copy the values  */
  //virtual hiopVector* new_alloc() const = 0;
  /** allocates a vector that mirrors this, and copies the values  */
  //virtual hiopVector* new_copy() const = 0;

  /* dual adjustment -> see hiopIterate::adjustDuals_primalLogHessian */
  virtual void adjustDuals_plh(const hiopVector& x, const hiopVector& ix, const double& mu, const double& kappa)=0;

  /* check for nans in the local vector */
  virtual bool isnan() const = 0;
  /* check for infs in the local vector */
  virtual bool isinf() const = 0;
  /* check if all values are finite /well-defined floats. Returns false is nan or infs are present. */
  virtual bool isfinite() const = 0;
  
  /* prints up to max_elems (by default all), on rank 'rank' (by default on all) */
  virtual void print(FILE*, const char* message=NULL,int max_elems=-1, int rank=-1) const = 0;
  
  inline long long get_size() const { return n; }
protected:
  long long n; //we assume sequential data

protected:
  hiopVector(const hiopVector& v) : n(v.n) {};
};

class hiopVectorPar : public hiopVector
{
public:
  hiopVectorPar(const long long& glob_n, long long* col_part=NULL, MPI_Comm comm=MPI_COMM_NULL);
  virtual ~hiopVectorPar();

  virtual void setToZero();
  virtual void setToConstant( double c );
  virtual void setToConstant_w_patternSelect(double c, const hiopVector& select);
  virtual void copyFrom(const hiopVector& v );
  virtual void copyFrom(const double* v_local_data); //v should be of length at least n_local
  /** Copy the 'n' elements of v starting at 'start_index_in_src' in 'this' */
  virtual void copyFromStarting(int start_index_in_src, const double* v, int n);
  virtual void copyFromStarting(int start_index_in_src, const hiopVector& v);
  /* copy 'dest' starting at 'start_idx_dest' to 'this' starting at 'start_idx_src' */
  virtual void startingAtCopyFromStartingAt(int start_idx_src, const hiopVector& v, int start_idx_dest);

  virtual void copyTo(double* dest) const;
  virtual void copyToStarting(int start_index_in_src, hiopVector& v);
  /* Copy 'this' to v starting at start_index in 'v'. */
  virtual void copyToStarting(hiopVector& v, int start_index_in_dest);
  /* copy 'this' (source) starting at 'start_idx_in_src' to 'dest' starting at index 'int start_idx_dest' 
   * If num_elems>=0, 'num_elems' will be copied; if num_elems<0, elements will be copied till the end of
   * either source ('this') or destination ('dest') is reached */
  virtual void startingAtCopyToStartingAt(int start_idx_in_src, hiopVector& dest, int start_idx_dest, int num_elems=-1) const;

  virtual double twonorm() const;
  virtual double dotProductWith( const hiopVector& v ) const;
  virtual double infnorm() const;
  virtual double infnorm_local() const;
  virtual double onenorm() const;
  virtual double onenorm_local() const; 
  virtual void componentMult( const hiopVector& v );
  virtual void componentDiv ( const hiopVector& v );
  virtual void componentDiv_p_selectPattern( const hiopVector& v, const hiopVector& ix);
  virtual void scale( double alpha );
  /** this += alpha * x */
  virtual void axpy  ( double alpha, const hiopVector& x );
  /** this += alpha * x * z */
  virtual void axzpy ( double alpha, const hiopVector& x, const hiopVector& z );
  /** this += alpha * x / z */
  virtual void axdzpy( double alpha, const hiopVector& x, const hiopVector& z );
  virtual void axdzpy_w_pattern( double alpha, const hiopVector& x, const hiopVector& z, const hiopVector& select ); 
  /** Add c to the elements of this */
  virtual void addConstant( double c );
  virtual void addConstant_w_patternSelect(double c, const hiopVector& ix);
  virtual void min( double& m, int& index ) const;
  virtual void negate();
  virtual void invert();
  virtual double logBarrier(const hiopVector& select) const;
  virtual void addLogBarrierGrad(double alpha, const hiopVector& x, const hiopVector& select);

  virtual double linearDampingTerm(const hiopVector& ixl_select, const hiopVector& ixu_select, 
				   const double& mu, const double& kappa_d) const;
  virtual int allPositive();
  virtual int allPositive_w_patternSelect(const hiopVector& w);
  virtual bool projectIntoBounds(const hiopVector& xl, const hiopVector& ixl, 
				 const hiopVector& xu, const hiopVector& ixu,
				 double kappa1, double kappa2);
  virtual double fractionToTheBdry(const hiopVector& dx, const double& tau) const;
  virtual double fractionToTheBdry_w_pattern(const hiopVector& dx, const double& tau, const hiopVector& ix) const;
  virtual void selectPattern(const hiopVector& ix);
  virtual bool matchesPattern(const hiopVector& ix);

  virtual hiopVectorPar* alloc_clone() const;
  virtual hiopVectorPar* new_copy () const;

  virtual void adjustDuals_plh(const hiopVector& x, const hiopVector& ix, const double& mu, const double& kappa);

  virtual bool isnan() const;
  virtual bool isinf() const;
  virtual bool isfinite() const;
  
  virtual void print(FILE*, const char* withMessage=NULL, int max_elems=-1, int rank=-1) const;

  /* more accessers */
  inline long long get_local_size() const { return n_local; }
  inline double* local_data() { return data; }
  inline const double* local_data_const() const { return data; }

protected:
  MPI_Comm comm;
  double* data;
  long long glob_il, glob_iu;
  long long n_local;
private:
  /** copy constructor, for internal/private use only (it doesn't copy the elements.) */
  hiopVectorPar(const hiopVectorPar&);

};

}
#endif
