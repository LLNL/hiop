// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory (LLNL).
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
 * @file hiopKKTLinSysSparseCondensed.hpp
 *
 * @author Cosmin G. Petra <petra1@llnl.gov>, LLNL
 */

#ifndef HIOP_KKTLINSYSSPARSECONDENSED
#define HIOP_KKTLINSYSSPARSECONDENSED

#include "hiopKKTLinSysSparse.hpp"
#include "hiopMatrixSparseTriplet.hpp"
#include "hiopMatrixSparseCSR.hpp"
#include "hiopKrylovSolver.hpp"

namespace hiop
{

/**
 * Wrapper class for CSR (to be moved and "upgraded" to hiopMatrixSparseCSR)
 */
class hiopMatrixSparseCSRStorage
{
public:
  hiopMatrixSparseCSRStorage();
  hiopMatrixSparseCSRStorage(size_type m, size_type n, size_type nnz);
  virtual ~hiopMatrixSparseCSRStorage();

  /**
   * Forms a CSR matrix from a sparse matrix in triplet format. Assumes input is ordered by
   * rows then by columns.
   */
  bool form_from(const hiopMatrixSparseTriplet& M);

  /// Same as above but does not updates values
  bool form_from(const size_type m,
                 const size_type n,
                 const size_type nnz,
                 const index_type* irow,
                 const index_type* jcol);
  /**
   * Forms a CSR matrix representing the transpose of the input sparse matrix in triplet 
   * format passed as argument. Assumes triplet input is ordered by rows then by columns.
   */
  bool form_transpose_from(const hiopMatrixSparseTriplet& M);

  /**
   * Computes M = X*D*Y, where X is the calling matrix class and D is a diagonal specified by 
   * a vector.
   */
  void times_diag_times_mat(const hiopVector& diag,
                            const hiopMatrixSparseCSRStorage& Y,
                            hiopMatrixSparseCSRStorage& M);
//protected:
  hiopMatrixSparseCSRStorage* times_diag_times_mat_init(const hiopMatrixSparseCSRStorage& Y);

  void times_diag_times_mat_numeric(const hiopVector& diag,
                                    const hiopMatrixSparseCSRStorage& Y,
                                    hiopMatrixSparseCSRStorage& M);

  /// Extract the diagonal 
  void get_diagonal(hiopVector& diag) const;
public:

  inline index_type* irowptr() 
  {
    return irowptr_;
  }
  inline index_type* jcolind()
  {
    return jcolind_;
  }
  inline double* values()
  {
    return values_;
  }

  inline index_type* irowptr() const
  {
    return irowptr_;
  }
  inline index_type* jcolind() const
  {
    return jcolind_;
  }
  inline double* values() const
  {
    return values_;
  }
  inline size_type m() const
  {
    return nrows_;
  }
  inline size_type n() const
  {
    return ncols_;
  }
  inline size_type nnz() const
  {
    return nnz_;
  }
  void print(FILE* file,
             const char* msg=NULL,
             int maxRows=-1,
             int maxCols=-1,
             int rank=-1) const;
protected:
  void alloc();
  void dealloc();
protected:
  size_type nrows_;
  size_type ncols_;
  size_type nnz_;
  index_type* irowptr_;
  index_type* jcolind_;
  double* values_;
};

/**
 * Solves a sparse KKT linear system by exploiting the sparse structure, namely reduces 
 * the so-called XDYcYd KKT system 
 * [  H  +  Dx    0    Jd^T ] [ dx]   [ rx_tilde ]
 * [    0         Dd   -I   ] [ dd] = [ rd_tilde ]
 * [    Jd       -I     0   ] [dyd]   [   ryd    ]
 * into the condensed KKT system
 * (H+Dx+Jd^T*Dd*Jd)dx = rx_tilde + Jd^T*Dd*ryd + Jd^T*rd_tilde
 * dd = Jd*dx - ryd
 * dyd = Dd*dd - rd_tilde = Dd*Jd*dx - Dd*ryd - rd_tilde

 * Here Jd is sparse Jacobians for inequalities, H is a sparse Hessian matrix, Dx is 
 * log-barrier diagonal corresponding to x variables, Dd is the log-barrier diagonal 
 * corresponding to the inequality slacks, and I is the identity matrix. 
 *
 * @note: the NLP is assumed to have no equality constraints (or have been relaxed to 
 * two-sided inequality constraints).
 *
 * Dual regularization may be not enforced as it requires repeated divisions that are 
 * prone to round-off error accumulation. When/If the class is going to be updated to
 * use dual regularization, the regularized XDYcYd KKT system reads:
 * [  H+Dx+delta_wx*I         0         Jd^T     ] [ dx]   [ rx_tilde ]
 * [          0         Dd+delta_wd*I   -I       ] [ dd] = [ rd_tilde ]
 * [          Jd             -I         -delta_cd] [dyd]   [   ryd    ]
 *
 * (notation) Dd2 = [ I+delta_cd*(Dd+delta_wd*I) ]^{-1}
 * (notation) Dd3 = Dd2*(Dd+delta_wd*I)
 *
 * dd = Jd*dx - delta_cd*dyd - ryd
 *
 * From (Dd+delta_wd*I)*dd - dyd = rd_tilde one can write
 *   ->   (Dd+delta_wd*I)*(Jd*dx - delta_cd*dyd - ryd) - dyd = rd_tilde
 *   ->   [I+delta_cd*(Dd+delta_wd*I)] dyd = (Dd+delta_wd*I)*(Jd*dx - ryd) - rd_tilde 
 * dyd = (I+delta_cd*(Dd+delta_wd*I))^{-1} [ (Dd+delta_wd*I)*(Jd*dx - ryd) - rd_tilde ]
 * dyd =               Dd2                 [ (Dd+delta_wd*I)*(Jd*dx - ryd) - rd_tilde ]
 * dyd = Dd3*Jd*dx - Dd3*ryd - Dd2 rd_tilde 
 *
 * (H+Dx+delta_wx*I + Jd^T * Dd3 * Jd) dx = rx_tilde + Jd^T*Dd3*ryd +  Jd^T*Dd2*rd_tilde
 */

// forward decls
class hiopKKTMatVecOpr;
class hiopKKTPrecondOpr;
  
class hiopKKTLinSysCondensedSparse : public hiopKKTLinSysCompressedSparseXDYcYd
{
public:
  hiopKKTLinSysCondensedSparse(hiopNlpFormulation* nlp);
  virtual ~hiopKKTLinSysCondensedSparse();

  virtual bool build_kkt_matrix(const double& delta_wx,
                                const double& delta_wd,
                                const double& delta_cc,
                                const double& delta_cd);

  virtual bool solveCompressed(hiopVector& rx,
                               hiopVector& rd,
                               hiopVector& ryc,
                               hiopVector& ryd,
                               hiopVector& dx,
                               hiopVector& dd,
                               hiopVector& dyc,
                               hiopVector& dyd);
protected:
  /**
   * Solves the compressed XDYcYd system by using direct solves with Cholesky factors of the 
   * condensed linear system and appropriately manipulate the XDYcYD rhs/sol to condensed rhs/sol.
   * 
   * The method is used as a preconditioner solve in the Krylov-based iterative refinement from
   * solve_compressed method.
   */
  virtual bool solve_compressed_direct(hiopVector& rx,
                                       hiopVector& rd,
                                       hiopVector& ryc,
                                       hiopVector& ryd,
                                       hiopVector& dx,
                                       hiopVector& dd,
                                       hiopVector& dyc,
                                       hiopVector& dyd);
  
  /// Helper method for allocating and precomputing symbolically JacD'*Dd*JacD + H + Dx + delta_wx*I
  hiopMatrixSparseCSRStorage* add_matrices_init(hiopMatrixSparseCSRStorage& JtDiagJ,
                                                hiopMatrixSymSparseTriplet& HessSp,
                                                hiopVector& Dx,
                                                double delta_wx);
  /// Helper method for fast computation of JacD'*Dd*JacD + H + Dx + delta_wx*I
  void add_matrices(hiopMatrixSparseCSRStorage& JtDiagJ,
                    hiopMatrixSymSparseTriplet& HessSp,
                    hiopVector& Dx,
                    double delta_wx,
                    hiopMatrixSparseCSRStorage& M);
protected:
  ////
  ////from the parent class and its parents we also use
  ////
  //right-hand side [rx_tilde, rd_tilde, ((ryc->empty)), ryd]
  //  hiopVector *rhs_; 

  
  //  hiopVectorPar *Dd;
  //  hiopVectorPar *ryd_tilde;

  //from the parent's parent class (hiopKKTLinSysCompressed) we also use
  //  hiopVectorPar *Dx;
  //  hiopVectorPar *rx_tilde;

  //keep Hx = Dx (Dx=log-barrier diagonal for x) + regularization
  //keep Hd = Dd (Dd=log-barrier diagonal for slack variable) + regularization
  //  hiopVector *Hx_, *Hd_;

  //
  //  hiopNlpSparse* nlpSp_;
  //  hiopMatrixSparse* HessSp_;
  //  const hiopMatrixSparse* Jac_cSp_;
  //  const hiopMatrixSparse* Jac_dSp_;

  // int write_linsys_counter_;
  //  hiopCSR_IO csr_writer_;

  /// Member for JacD in CSR format
  hiopMatrixSparseCSR* JacD_;
  
  /// Member for JacD' in CSR format
  hiopMatrixSparseCSR* JacDt_;

  /// Member for lower triangular part of Hess in CSR
  hiopMatrixSparseCSR* Hess_lower_csr_;

  /// Member for upper triangular part of Hess
  hiopMatrixSparseCSR* Hess_upper_csr_;
  
  /// Member for Hess
  hiopMatrixSparseCSR* Hess_csr_;
  
  /// Member for JacD'*Dd*JacD
  hiopMatrixSparseCSRStorage* JtDiagJsto_;
  hiopMatrixSparseCSR* JtDiagJ_;

  /// Member for JacD'*Dd*JacD + H + Dx + delta_wx*I
  hiopMatrixSparseCSRStorage* M_condensedsto_;
  hiopMatrixSparseCSR* M_condensed_;

  /**
   * Maps indexes of the nonzeros in the union of the sparsity patterns of JacD'*Dd*JacD, H, and
   * Dx + delta_wx*I, into the sorted CSR arrays of M_condensed_. Allows fast computation of
   * the nonzeros of M_condensed_ from the nonzeros of JacD'*Dd*JacD, H, and
   * Dx + delta_wx*I via add_matrices method. It is computed in add_matrices_init.
   */
  std::vector<index_type> map_idxs_in_sorted_;

  /**
   * Keeps the indexes of the nonzeros of the strictly lower triangle of H in the values array of H.
   * It is built in add_matrices_init and reused in add_matrices to allow fast numerical computation
   * of M_condensed_.
   */
  std::vector<index_type> map_H_lowtr_idxs_;

  /// Inertia correction perturbations used in the (last) factorization
  double delta_wx_;

  /// Matrix operator performing mat-vec with the XDYcYd KKT linear system matrix
  hiopKKTMatVecOpr* krylov_mat_opr_;

  /// Preconditioner operator that solves with the factors of the condensed system via the
  /// solve_compressed_direct method
  hiopKKTPrecondOpr* krylov_prec_opr_;

  /// Temporary vector to be used in the Krylov solve;
  hiopVector* krylov_rhs_xdycyd_;

  /// BiCGStab solver
  hiopBiCGStabSolver* bicgstab_;
private:
  //placeholder for the code that decides which linear solver to used based on safe_mode_
  hiopLinSolverSymSparse* determine_and_create_linsys(size_type nxd, size_type nineq, size_type nnz);

  friend class hiopKKTMatVecOpr;
  friend class hiopKKTPrecondOpr;
};

/** 
 * Krylov linear operator and preconditioner for the condensed KKT linsys
 */

class hiopKKTMatVecOpr : public hiopLinearOperator
{
public:
  hiopKKTMatVecOpr(hiopKKTLinSysCondensedSparse* kkt)
    : kkt_(kkt)
  {
    const hiopMatrixSparse* Jac_d = kkt_->Jac_dSp_;
    const hiopMatrixSparse* Hess = kkt_->HessSp_;
    const hiopVector* Dd_pert = kkt_->Hd_;
    const hiopVector* Dx = kkt_->Dx_;
    
    xdx_ = Dx->alloc_clone();
    xdd_ = Dd_pert->alloc_clone();
    xdyd_ = Dd_pert->alloc_clone();

    yrx_ = xdx_->alloc_clone();
    yrd_ = xdd_->alloc_clone();
    yryd_ = xdyd_->alloc_clone();
  }
  virtual ~hiopKKTMatVecOpr()
  {
    delete xdx_;
    delete xdd_;
    delete xdyd_;
    delete yrx_;
    delete yrd_;
    delete yryd_;
  }
  /** y = KKT * x */
  virtual bool times_vec(hiopVector& y, const hiopVector& x)
  {
    const hiopMatrixSparse* Jac_d = kkt_->Jac_dSp_;
    const hiopMatrixSparse* Hess = kkt_->HessSp_;
    const hiopVector* Dd_pert = kkt_->Hd_;
    const hiopVector* Dx = kkt_->Dx_;
    const double& delta_wx = kkt_->delta_wx_;

    const size_type nx = Hess->n();
    const size_type nineq = Jac_d->m();

    assert(x.get_size() == y.get_size());
    assert(nx+2*nineq == x.get_size());
    
    xdx_->startingAtCopyFromStartingAt(0, x, 0);
    xdd_->startingAtCopyFromStartingAt(0, x, nx);
    xdyd_->startingAtCopyFromStartingAt(0, x, nx+nineq);

    // yrx = (H+Dx+delta_wx*I)*xdx + Jd^T*xdyd
    yrx_->copyFrom(*xdx_);
    yrx_->componentMult(*Dx);
    yrx_->axpy(delta_wx, *xdx_);
    Hess->timesVec(1.0, *yrx_, 1.0, *xdx_);
    Jac_d->transTimesVec(1.0, *yrx_, 1.0, *xdyd_);
    yrx_->copyToStarting(y, 0);

    // yrd = (Dd+delta_wd*I)*xdd - dyd
    yrd_->copyFrom(*xdd_);
    yrd_->componentMult(*Dd_pert);
    yrd_->axpy(-1.0, *xdyd_);
    yrd_->copyToStarting(y, nx);

    // yryd = Jd*xdx - xdd 
    yryd_->copyFrom(*xdd_);
    Jac_d->timesVec(-1.0, *yryd_, 1.0, *xdx_);
    yryd_->copyToStarting(y, nx+nineq);

    return true;
  }

  /** y = KKT' * x */
  virtual bool trans_times_vec(hiopVector& y, const hiopVector& x)
  {
    //KKT has a symmetric linear system matrix
    return times_vec(y, x);
  }
private:
  hiopKKTLinSysCondensedSparse* kkt_;
  hiopKKTMatVecOpr()
    : kkt_(nullptr),
      xdx_(nullptr),
      xdd_(nullptr),
      xdyd_(nullptr),
      yrx_(nullptr),
      yrd_(nullptr),
      yryd_(nullptr)
  {
    assert(false && "this constructor should not be used");
  }
  hiopVector* xdx_;
  hiopVector* xdd_;
  hiopVector* xdyd_;
  hiopVector* yrx_;
  hiopVector* yrd_;
  hiopVector* yryd_;  
};

class hiopKKTPrecondOpr : public hiopLinearOperator
{
public:
  hiopKKTPrecondOpr(hiopKKTLinSysCondensedSparse* kkt)
    : kkt_(kkt),
      xyc_(nullptr),
      xdx_(nullptr),
      xdd_(nullptr),
      xdyd_(nullptr),
      yrx_(nullptr),
      yrd_(nullptr),
      yryd_(nullptr)
  {
    const hiopMatrixSparse* Jac_d = kkt_->Jac_dSp_;
    const hiopMatrixSparse* Hess = kkt_->HessSp_;
    const hiopVector* Dd_pert = kkt_->Hd_;
    const hiopVector* Dx = kkt_->Dx_;
    const size_type nx = Hess->n();
    const size_type nineq = Jac_d->m();

    xdx_ = Dx->alloc_clone();
    //dummy xyc (also used for ryc)
    xyc_ = new hiopVectorPar(0);
    xdd_ = Dd_pert->alloc_clone();
    xdyd_ = Dd_pert->alloc_clone();

    yrx_ = xdx_->alloc_clone();
    yrd_ = xdd_->alloc_clone();
    yryd_ = xdyd_->alloc_clone();
  }
  virtual ~hiopKKTPrecondOpr()
  {
    delete xyc_;
    delete xdx_;
    delete xdd_;
    delete xdyd_;
    delete yrx_;
    delete yrd_;
    delete yryd_;
  }
  /** y = KKT * x */
  virtual bool times_vec(hiopVector& y, const hiopVector& x)
  {
    bool bret;
    const hiopMatrixSparse* Jac_d = kkt_->Jac_dSp_;
    const hiopMatrixSparse* Hess = kkt_->HessSp_;
    const hiopVector* Dd_pert = kkt_->Hd_;
    const hiopVector* Dx = kkt_->Dx_;
    const size_type nx = Hess->n();
    const size_type nineq = Jac_d->m();

    assert(x.get_size() == y.get_size());
    assert(nx+2*nineq == x.get_size());
    
    xdx_->startingAtCopyFromStartingAt(0, x, 0);
    xdd_->startingAtCopyFromStartingAt(0, x, nx);
    xdyd_->startingAtCopyFromStartingAt(0, x, nx+nineq);

    bret = kkt_->solve_compressed_direct(*xdx_, *xdd_, *xyc_, *xdyd_, *yrx_, *yrd_, *xyc_, *yryd_); 
    
    yrx_->copyToStarting(y, 0);
    yrd_->copyToStarting(y, nx);
    yryd_->copyToStarting(y, nx+nineq);

    return bret;
  }

  /** y = KKT' * x */
  virtual bool trans_times_vec(hiopVector& y, const hiopVector& x)
  {
    //KKT has a symmetric linear system matrix
    return times_vec(y, x);
  }
private:
  hiopKKTLinSysCondensedSparse* kkt_;
  hiopKKTPrecondOpr()
    : kkt_(nullptr)
  {
    assert(false);
  }
  hiopVector* xdx_;
  hiopVector* xdd_;
  hiopVector* xyc_;
  hiopVector* xdyd_;
  hiopVector* yrx_;
  hiopVector* yrd_;
  hiopVector* yryd_;
};  
  
} // end of namespace

#endif
