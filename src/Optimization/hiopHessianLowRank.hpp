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

#ifndef HIOP_HESSIANLOWRANK
#define HIOP_HESSIANLOWRANK

#include "hiopNlpFormulation.hpp"
#include "hiopIterate.hpp"

#include <cassert>

namespace hiop
{

/* Class for storing and solving with the low-rank Hessian 
 *
 * Stores the Hessian wrt x as Hk=Dk+Bk, where 
 *  - Dk is the log barrier diagonal
 *  - Bk=B0 - M*N^{-1}*M^T is the secant approximation for the Hessian of the Lagrangian
 * in the compact representation of  Byrd, Nocedal, and Schnabel (1994)
 * Reference: Byrd, Nocedal, and Schnabel, "Representations of quasi-Newton matrices and
 * and there use in limited memory methods", Math. Programming 63 (1994), p. 129-156.
 * 
 * M=[B0*Sk Yk] is nx2l, and N is 2lx2l, where n=dim(x) and l is the length of the memory of secant 
 * approximation. This class is for when n>>k (l=O(10)).
 * 
 * This class provides functionality to KKT linear system class for updating the secant approximation 
 * and solving with Hk=Dk+Bk.
 * 
 * Solving with Hk is performed by 
 *  1. computing the inverse as
 * Hk{-1} = (Dk+B0)^{-1} - (Dk+B0)^{-1}*[B0*Sk Yk]*( -N + [Sk'*B0]*(Dk+B0)^{-1}*[B0*Sk Yk] )^{-1} *[Sk'*B0]*(Dk+B0)^{-1}
 *                                                 (      [Yk'   ]                         )       [Yk'   ]
 *  2. multiplying with the above expression. The inner 2lx2l inverse matrix is not explicitly computed; instead 
 * V=(-N + [Sk'*B0]*(Dk+B0)^{-1}*[B0*Sk Yk] ) is stored, factorized, and solved with.
 *   (     [Yk'   ]              [Yk'     ] ) 
 *
 * Notation used in the implementation provided by this class
 *  - DhInv:=(Dk+B0)^{-1}, thus Hk{-1}=DhInv-DhInv*U*V^{-1}*U'*DhInv with
 *  - U:=[B0*St' Yt'] and  V is defined above
 *  - 
 *  
 * Parallel computations: Dk, B0 are distributed vectors, M is distributed 
 * column-wise, and N is local (stored on all processors).
 */
class hiopHessianLowRank : public hiopMatrix
{
public:
  hiopHessianLowRank(hiopNlpDenseConstraints* nlp_, int max_memory_length);
  virtual ~hiopHessianLowRank();

  /* return false if the update destroys hereditary positive definitness and the BFGS update is not taken*/
  virtual bool update(const hiopIterate& x_curr, const hiopVector& grad_f_curr,
		      const hiopMatrix& Jac_c_curr, const hiopMatrix& Jac_d_curr);

  /* updates the logBar diagonal term from the representation */
  virtual bool updateLogBarrierDiagonal(const hiopVector& Dx);

  /* solves this*x=res */
  virtual void solve(const hiopVector& rhs, hiopVector& x);
  /* W = beta*W + alpha*X*inverse(this)*X^T (a more efficient version of solve)
   * This is performed as W = beta*W + alpha*X*(this\X^T)
   */ 
  virtual void symMatTimesInverseTimesMatTrans(double beta, hiopMatrixDense& W_, 
					       double alpha, const hiopMatrixDense& X);
#ifdef HIOP_DEEPCHECKS
  /* computes the product of the Hessian with a vector: y=beta*y+alpha*H*x.
   * The function is supposed to use the underlying ***recursive*** definition of the 
   * quasi-Newton Hessian and is used for checking/testing/error calculation.
   */
  virtual void timesVec(double beta, hiopVector& y, double alpha, const hiopVector&x);

  /* same as above but without the Dx term in H */
  virtual void timesVec_noLogBarrierTerm(double beta, hiopVector& y, double alpha, const hiopVector&x);
  /* code shared by the above two methods*/
  virtual void timesVecCmn(double beta, hiopVector& y, double alpha, const hiopVector&x, bool addLogBarTerm);

  virtual void print(FILE* f, hiopOutVerbosity v, const char* msg) const;
#endif

protected:
  int l_max; //max memory size
  int l_curr; //number of pairs currently stored
  double sigma; //initial scaling factor of identity
  double sigma0; //default scaling factor of identity
  int sigma_update_strategy;
  double sigma_safe_min, sigma_safe_max; //min and max safety thresholds for sigma
  hiopNlpDenseConstraints* nlp;
private:
  hiopVector* DhInv; //(B0+Dk)^{-1}
#ifdef HIOP_DEEPCHECKS
  // needed in timesVec (for residual checking in solveCompressed, only HIOP_DEEPCHECKS mode).
  // can be recomputed from DhInv decided to store it instead to avoid round-off errors
  hiopVector* _Dx; 
#endif
  bool matrixChanged;
  //these are matrices from the compact representation; they are updated at each iteration.
  // more exactly Bk=B0-[B0*St' Yt']*[St*B0*St'  L]*[St*B0]
  //                                 [  L'      -D] [Yt   ]                   
  hiopMatrixDense *St,*Yt; //we store the transpose to easily access columns in S and T
  hiopMatrixDense *L;     //lower triangular from the compact representation
  hiopVector* D;       //diag 
  //these are matrices from the representation of the inverse
  hiopMatrixDense* V;    
#ifdef HIOP_DEEPCHECKS
  //copy of the matrix - needed to check the residual
   hiopMatrixDense* _Vmat; 
#endif
  void growL(const int& lmem_curr, const int& lmem_max, const hiopVector& YTs);
  void growD(const int& l_curr, const int& l_max, const double& sTy);
  void updateL(const hiopVector& STy, const double& sTy);
  void updateD(const double& sTy);
  //also stored are the iterate, gradient obj, and Jacobians at the previous optimization iteration
  hiopIterate *_it_prev;
  hiopVector *_grad_f_prev;
  hiopMatrixDense *_Jac_c_prev, *_Jac_d_prev;

  //internal helpers
  void updateInternalBFGSRepresentation();

  //internals buffers, mostly for MPIAll_reduce
  double* _buff_kxk; // size = num_constraints^2 
  double* _buff_2lxk; // size = 2 x q-Newton mem size x num_constraints
  double *_buff1_lxlx3, *_buff2_lxlx3;
  //auxiliary objects
  hiopMatrixDense *_S1, *_Y1, *_lxl_mat1, *_kx2l_mat1, *_kxl_mat1; //preallocated matrices 
  //holds X*D*S
  hiopMatrixDense& new_S1(const hiopMatrixDense& X, const hiopMatrixDense& St);
  //holds X*D*Y
  hiopMatrixDense& new_Y1(const hiopMatrixDense& X, const hiopMatrixDense& Yt);
  hiopMatrixDense& new_lxl_mat1 (int l);
  hiopMatrixDense& new_kxl_mat1 (int k, int l);
  hiopMatrixDense& new_kx2l_mat1(int k, int l);
  
  hiopVector *_l_vec1, *_l_vec2, *_n_vec1, *_n_vec2, *_2l_vec1;
  hiopVector& new_l_vec1(int l);
  hiopVector& new_l_vec2(int l);
  inline hiopVector& new_n_vec1(long long n)
  {
#ifdef HIOP_DEEPCHECKS
    assert(_n_vec1!=NULL);
    assert(_n_vec1->get_size()==n);
#endif
    return *_n_vec1;
  }
  inline hiopVector& new_n_vec2(long long n)
  {
#ifdef HIOP_DEEPCHECKS
    assert(_n_vec2!=NULL);
    assert(_n_vec2->get_size()==n);
#endif
    return *_n_vec2;
  }
  inline hiopVector& new_2l_vec1(int l) {
    if(_2l_vec1!=NULL && _2l_vec1->get_size()==2*l) return *_2l_vec1;
    if(_2l_vec1!=NULL) delete _2l_vec1;
    _2l_vec1=LinearAlgebraFactory::createVector(2*l);
    return *_2l_vec1;
  }
private:
  //utilities
  /* symmetric multiplication W = beta*W + alpha*X*Diag*X^T */
  static void symmMatTimesDiagTimesMatTrans_local(double beta, hiopMatrixDense& W_,
					   double alpha, const hiopMatrixDense& X_,
					   const hiopVector& d);
  /* W=S*Diag*X^T */
  static void matTimesDiagTimesMatTrans_local(hiopMatrixDense& W, const hiopMatrixDense& S, 
					      const hiopVector& d, const hiopMatrixDense& X);
  /* members and utilities related to V matrix: factorization and solve */
  hiopVector *_V_work_vec;
  int _V_ipiv_size; int* _V_ipiv_vec;
  void factorizeV();
  void solveWithV(hiopVector& rhs_s, hiopVector& rhs_y);
  void solveWithV(hiopMatrixDense& rhs);
private:
  hiopHessianLowRank() {};
  hiopHessianLowRank(const hiopHessianLowRank&) {};
  hiopHessianLowRank& operator=(const hiopHessianLowRank&) {return *this;};
public:
  /* methods that need to be implemented as the class inherits from hiopMatrix*/
public:
  virtual hiopMatrix* alloc_clone() const
  {
    assert(false && "not provided because it is not needed");
    return NULL;
  }
  virtual hiopMatrix* new_copy() const
  {
    assert(false && "not provided because it is not needed");
    return NULL;
  }

  virtual void setToZero()
  {
    assert(false && "not provided because it is not needed");
  }
  virtual void setToConstant(double c)
  {
    assert(false && "not provided because it is not needed");
  }

  void timesVec(double beta, hiopVector& y, double alpha, const hiopVector&x) const
  {
    assert(false && "not provided because it is not needed");
  }
  /** y = beta * y + alpha * this^T * x */
  virtual void transTimesVec(double beta,   hiopVector& y,
			     double alpha,  const hiopVector& x ) const
  {
    assert(false && "not provided because it is not needed");
  }

  /* W = beta*W + alpha*this*X */  
  virtual void timesMat(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const
  {
    assert(false && "not provided because it is not needed");
  }
  /* W = beta*W + alpha*this^T*X */
  virtual void transTimesMat(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const
  {
    assert(false && "not provided because it is not needed");
  }
  /* W = beta*W + alpha*this*X^T */
  virtual void timesMatTrans(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const
  {
    assert(false && "not provided because it is not needed");
  }
  virtual void addDiagonal(const double& alpha, const hiopVector& d_)
  {
    assert(false && "not provided because it is not needed");
  }
  virtual void addDiagonal(const double& value)
  {
    assert(false && "not provided because it is not needed");
  }
  virtual void addSubDiagonal(const double& alpha, long long start, const hiopVector& d_)
  {
    assert(false && "not provided because it is not needed");
  }
  /* add to the diagonal of 'this' (destination) starting at 'start_on_dest_diag' elements of
   * 'd_' (source) starting at index 'start_on_src_vec'. The number of elements added is 'num_elems' 
   * when num_elems>=0, or the remaining elems on 'd_' starting at 'start_on_src_vec'. */
  virtual void addSubDiagonal(int start_on_dest_diag, const double& alpha, 
			      const hiopVector& d_, int start_on_src_vec, int num_elems=-1)
  {
    assert(false && "not needed / implemented");
  }
  virtual void addSubDiagonal(int start_on_dest_diag, int num_elems, const double& c) 
  {
    assert(false && "not needed / implemented");
  }

  
  /* this += alpha*X */
  virtual void addMatrix(double alpah, const hiopMatrix& X)
  {
    assert(false && "not provided because it is not needed");
  }

  void addToSymDenseMatrixUpperTriangle(int row_start, int col_start, double alpha, hiopMatrixDense& W) const
  {
    assert(false && "not needed; should not be used");
  }
  void transAddToSymDenseMatrixUpperTriangle(int row_start, int col_start, double alpha, hiopMatrixDense& W) const
  {
    assert(false && "not needed; should not be used");
  }
  void addUpperTriangleToSymDenseMatrixUpperTriangle(int diag_start, 
						     double alpha, hiopMatrixDense& W) const
  {
    assert(false && "not needed; should not be used");
  }
  virtual double max_abs_value()
  {
    assert(false && "not provided because it is not needed");
    return 0.;
  }

  virtual void row_max_abs_value(hiopVector &ret_vec)
  {
    assert(false && "not provided because it is not needed");
  }
  
  virtual void scale_row(hiopVector &vec_scal, const bool inv_scale)
  {
    assert(false && "not provided because it is not needed");
  }

  void copyRowsFrom(const hiopMatrix& src_in, const long long* rows_idxs, long long n_rows)
  {
    assert(false && "not needed / should not be used");
  }
  
  /* return false is any of the entry is a nan, inf, or denormalized */
  virtual bool isfinite() const
  {
    assert(false && "not provided because it is not needed");
    return false;
  }
  
  /* call with -1 to print all rows, all columns, or on all ranks; otherwise will
  *  will print the first rows and/or columns on the specified rank.
  * 
  * If the underlying matrix is sparse, maxCols is ignored and a max number elements 
  * given by the value of 'maxRows' will be printed. If this value is negative, all
  * elements will be printed.
  */
  virtual void print(FILE* f=NULL, const char* msg=NULL, int maxRows=-1, int maxCols=-1, int rank=-1) const
  {
    assert(false && "not provided because it is not needed");
  }

  /* number of rows */
  virtual long long m() const { assert(false && "not provided because it is not needed"); return 0; }
  /* number of columns */
  virtual long long n() const { assert(false && "not provided because it is not needed"); return 0; }
#ifdef HIOP_DEEPCHECKS
  /* check symmetry */
  virtual bool assertSymmetry(double tol=1e-16) const { return true; }
#endif

};

/* Low-rank representation for the inverse of a low-rank matrix (Hessian).
 * In HIOP we need to solve with the BFGS quasi-Newton Hessian given by
 * the matrix M=B0+[B0*S Y] [S^T*B0*S  L] [ S^T*B0 ]
 *                          [ L^T     -D] [   Y   ]
 * Reference: Byrd, Nocedal, Schnabel, "Representations of quasi-Newton matrices and
 * and there use in limited memory methods", Math. Programming 63 (1994), p. 129-156.
 * 
 * To save on computations, we maintain a direct representaton of its inverse 
 * M^{-1} = H0 + [S HO*Y] [ R^{-T}*(D+Y^T*H0*Y)*R^{-1}    -R^{-T} ] [ S^T   ]
 *                        [          -R^{-1}                 0    ] [ Y^T*H0]
 * Here, H0=inv(B0). Check the above reference to see what each matrix represents. 
 */
class hiopHessianInvLowRank_obsolette : public hiopHessianLowRank
{
public:
  hiopHessianInvLowRank_obsolette(hiopNlpDenseConstraints* nlp, int max_memory_length);
  virtual bool update(const hiopIterate& x_curr, const hiopVector& grad_f_curr,
		      const hiopMatrix& Jac_c_curr, const hiopMatrix& Jac_d_curr);

  virtual bool updateLogBarrierDiagonal(const hiopVector& Dx);

  /* ! 
   * these methods use quantities computed in symmetricTimesMat. They should be called
   * after symmetricTimesMat
   */
  virtual void apply(double beta, hiopVector& y, double alpha, const hiopVector& x);
  virtual void apply(double beta, hiopMatrix& Y, double alpha, const hiopMatrix& X);

  /* W = beta*W + alpha*X*this*X^T
   * ! make sure this is called before 'apply'
   */
  virtual void symmetricTimesMat(double beta, hiopMatrixDense& W,
				 double alpha, const hiopMatrixDense& X);

  virtual ~hiopHessianInvLowRank_obsolette();

private: //internal methods
  
  /* symmetric multiplication W = beta*W + alpha*X*Diag*X^T */
  static void symmMatTimesDiagTimesMatTrans_local(double beta, hiopMatrixDense& W_,
					   double alpha, const hiopMatrixDense& X_,
					   const hiopVector& d);
  /* W=S*Diag*X^T */
  static void matTimesDiagTimesMatTrans_local(hiopMatrixDense& W, const hiopMatrixDense& S, const hiopVector& d, const hiopMatrixDense& X);

  /* rhs = R \ rhs, where R is upper triangular lxl and rhs is lx */
  static void triangularSolve(const hiopMatrixDense& R, hiopMatrixDense& rhs);
  static void triangularSolve(const hiopMatrixDense& R, hiopVector& rhs);
  static void triangularSolveTrans(const hiopMatrixDense& R, hiopVector& rhs);

  //grows R when the number of BFGS updates is less than the max memory
  void growR(const int& l_curr, const int& l_max, const hiopVector& STy, const double& sTy);
  void growD(const int& l_curr, const int& l_max, const double& sTy);
  void updateR(const hiopVector& STy, const double& sTy);
  void updateD(const double& sTy);
private:
  hiopVector* H0;
  hiopMatrixDense *St,*Yt; //we store the transpose to easily access columns in S and T
  hiopMatrixDense *R;
  hiopVector* D;

  int sigma_update_strategy;
  double sigma_safe_min, sigma_safe_max;

  //also stored are the iterate, gradient obj, and Jacobians at the previous iterations
  hiopIterate *_it_prev;
  hiopVector *_grad_f_prev;
  hiopMatrixDense *_Jac_c_prev, *_Jac_d_prev;

  //internals buffers
  double* _buff_kxk; // size = num_constraints^2 
  double* _buff_lxk; // size = q-Newton mem size x num_constraints
  double* _buff_lxl;
  //auxiliary objects
  hiopMatrixDense *_S1, *_Y1, *_DpYtH0Y; //aux matrices to hold St*X, Yt*H0*X, and D+Y^T*H0*Y in symmetricTimesMat
  hiopMatrixDense& new_S1(const hiopMatrixDense& St, const hiopMatrixDense& X);
  hiopMatrixDense& new_Y1(const hiopMatrixDense& Yt, const hiopMatrixDense& X);
  hiopMatrixDense& new_DpYtH0Y(const hiopMatrixDense& Yt);
  //similar for S3=DpYtH0Y*S2
  hiopMatrixDense *_S3;
  hiopMatrixDense& new_S3(const hiopMatrixDense& Left, const hiopMatrixDense& Right);
  hiopVector *_l_vec1, *_l_vec2, *_l_vec3, *_n_vec1, *_n_vec2;
  hiopVector& new_l_vec1(int l);
  hiopVector& new_l_vec2(int l);
  hiopVector& new_l_vec3(int l);
  inline hiopVector& new_n_vec1(long long n)
  {
#ifdef HIOP_DEEPCHECKS
    assert(_n_vec1!=NULL);
    assert(_n_vec1->get_size()==n);
#endif
    return *_n_vec1;
  }
  inline hiopVector& new_n_vec2(long long n)
  {
#ifdef HIOP_DEEPCHECKS
    assert(_n_vec2!=NULL);
    assert(_n_vec2->get_size()==n);
#endif
    return *_n_vec2;
  }
};

} //~namespace
#endif
