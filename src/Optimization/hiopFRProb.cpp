// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
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
 * @file hiopFRProb.cpp
 *
 * @author Nai-Yuan Chiang <chiang7@llnl.gov>, LLNL
 *
 */

#include "hiopFRProb.hpp"

#include "hiopVector.hpp"

#include <cmath>
#include <cstring> //for memcpy
#include <cstdio>
#include <math.h>

namespace hiop
{

/* 
*  Specialized interface for feasibility restoration problem with sparse blocks in the Jacobian and Hessian.
*/
hiopFRProbSparse::hiopFRProbSparse(hiopAlgFilterIPMBase& solver_base)
  : solver_base_(solver_base)
{
  nlp_base_ = dynamic_cast<hiopNlpSparse*>(solver_base.get_nlp());
  n_x_ = nlp_base_->n();
  m_eq_ = nlp_base_->m_eq();
  m_ineq_ = nlp_base_->m_ineq();

  n_ = n_x_ + 2*m_eq_ + 2*m_ineq_;
  m_ = m_eq_ + m_ineq_;

  pe_st_ = n_x_;
  ne_st_ = pe_st_ + m_eq_;
  pi_st_ = ne_st_ + m_eq_;
  ni_st_ = pi_st_ + m_ineq_;

  x_ref_ = solver_base.get_it_curr()->get_x();

  // build vector VR
  DR_ = x_ref_->new_copy();
  DR_->component_abs();
  DR_->invert();
  DR_->component_min(1.0);

  wrk_x_ = x_ref_->alloc_clone();
  wrk_c_ = LinearAlgebraFactory::createVector(m_eq_);
  wrk_d_ = LinearAlgebraFactory::createVector(m_ineq_);
  wrk_eq_ = LinearAlgebraFactory::createVector(m_eq_);
  wrk_ineq_ = LinearAlgebraFactory::createVector(m_ineq_);
  wrk_cbody_ = LinearAlgebraFactory::createVector(m_eq_);
  wrk_dbody_ = LinearAlgebraFactory::createVector(m_ineq_);
  wrk_primal_ = LinearAlgebraFactory::createVector(n_);
  wrk_dual_ = LinearAlgebraFactory::createVector(m_);

  // nnz for sparse matrices;
  nnz_Jac_c_ = nlp_base_->get_nnz_Jaceq() + 2 * m_eq_;
  nnz_Jac_d_ = nlp_base_->get_nnz_Jacineq() + 2 * m_ineq_;
  
  // not sure i Hess has diagonal terms, compute nnz_hess here
  // assuming hess is in upper_triangular form
  hiopMatrixSparse* Hess_base = dynamic_cast<hiopMatrixSparse*>(solver_base_.get_Hess_Lagr());
  nnz_Hess_Lag_ = n_x_ + Hess_base->numberOfOffDiagNonzeros();
  
  Jac_cd_ = LinearAlgebraFactory::createMatrixSparse(m_, n_, nnz_Jac_c_ + nnz_Jac_d_);
  Hess_cd_ = LinearAlgebraFactory::createMatrixSymSparse(n_, nnz_Hess_Lag_);
  
  // set mu0 to be the maximun of the current barrier parameter mu and norm_inf(|c|)*/
  theta_ref_ = solver_base_.get_resid()->get_theta(); //at current point, i.e., reference point
  mu_ = solver_base.get_mu();
  mu_ = std::max(mu_, solver_base_.get_resid()->get_nrmInf_bar_feasib());

  zeta_ = std::sqrt(mu_);
  rho_ = 1000; // FIXME: make this as an user option
}

hiopFRProbSparse::~hiopFRProbSparse()
{
  delete wrk_x_;
  delete wrk_c_;
  delete wrk_d_;
  delete wrk_eq_;
  delete wrk_ineq_;
  delete wrk_cbody_;
  delete wrk_dbody_;
  delete wrk_primal_;
  delete wrk_dual_;
  delete DR_;
  delete Jac_cd_;
  delete Hess_cd_;
}

bool hiopFRProbSparse::get_prob_sizes(size_type& n, size_type& m)
{
  n = n_;
  m = m_;
  return true;
}

bool hiopFRProbSparse::get_vars_info(const size_type& n, double *xlow, double* xupp, NonlinearityType* type)
{
  assert(n == n_);

  const hiopVector& xl = nlp_base_->get_xl();
  const hiopVector& xu = nlp_base_->get_xu();
  const NonlinearityType* var_type = nlp_base_->get_var_type();

  // x, p and n
  wrk_primal_->setToConstant(0.0);
  xl.copyToStarting(*wrk_primal_,0);
  wrk_primal_->copyTo(xlow);

  wrk_primal_->setToConstant(1e+20);
  xu.copyToStarting(*wrk_primal_,0);
  wrk_primal_->copyTo(xupp);

  wrk_primal_->set_array_from_to(type, 0, n_x_, var_type, 0);
  wrk_primal_->set_array_from_to(type, n_x_, n_, hiopLinear);

  return true;
}

bool hiopFRProbSparse::get_cons_info(const size_type& m, double* clow, double* cupp, NonlinearityType* type)
{
  assert(m == m_);
  assert(m_eq_ + m_ineq_ == m_);
  const hiopVector& crhs = nlp_base_->get_crhs();
  const hiopVector& dl = nlp_base_->get_dl();
  const hiopVector& du = nlp_base_->get_du();
  const NonlinearityType* cons_eq_type = nlp_base_->get_cons_eq_type();
  const NonlinearityType* cons_ineq_type = nlp_base_->get_cons_ineq_type();

  wrk_dual_->setToConstant(0.0);

  // assemble wrk_dual_ = [crhs; dl] for lower bounds
  crhs.copyToStarting(*wrk_dual_, 0);
  dl.copyToStarting(*wrk_dual_, (int)m_eq_);
  wrk_dual_->copyTo(clow);

  // assemble wrk_dual_ = [crhs; du] for upper bounds
  du.copyToStarting(*wrk_dual_, (int)m_eq_);
  wrk_dual_->copyTo(cupp);

  wrk_dual_->set_array_from_to(type, 0, m_eq_, cons_eq_type, 0);
  wrk_dual_->set_array_from_to(type, m_eq_, m_, cons_ineq_type, 0);

  return true;
}

bool hiopFRProbSparse::get_sparse_blocks_info(int& nx,
                                              int& nnz_sparse_Jaceq,
                                              int& nnz_sparse_Jacineq,
                                              int& nnz_sparse_Hess_Lagr)
{
  nx = n_;
  nnz_sparse_Jaceq = nnz_Jac_c_;
  nnz_sparse_Jacineq = nnz_Jac_d_;
  nnz_sparse_Hess_Lagr = nnz_Hess_Lag_;
  return true;
}

bool hiopFRProbSparse::eval_f(const size_type& n, const double* x, bool new_x, double& obj_value)
{
  assert(n == n_);
  obj_value = 0.;

  wrk_primal_->copy_from_starting_at(x, 0, n_); // [x pe ne pi ni]
  wrk_x_->copy_from_starting_at(x, 0, n_x_);    // [x]
  
  // rho*sum(p+n)
  obj_value += rho_ * (wrk_primal_->sum_local() - wrk_x_->sum_local());

  // zeta/2*[DR*(x-x_ref)]^2
  wrk_x_->axpy(-1.0, *x_ref_);
  wrk_x_->componentMult(*DR_);
  double wrk_db = wrk_x_->twonorm();

  obj_value += 0.5 * zeta_ * wrk_db * wrk_db;

  return true;
}

bool hiopFRProbSparse::eval_grad_f(const size_type& n, const double* x, bool new_x, double* gradf)
{
  assert(n == n_);

  // p and n
  wrk_primal_->setToConstant(rho_);

  // x
  wrk_x_->copy_from_starting_at(x, 0, n_x_);
  wrk_x_->axpy(-1.0, *x_ref_);
  wrk_x_->componentMult(*DR_);
  wrk_x_->componentMult(*DR_);
  wrk_x_->scale(zeta_);
  wrk_x_->copyToStarting(*wrk_primal_,0);

  wrk_primal_->copyTo(gradf);

  return true;
}

bool hiopFRProbSparse::eval_cons(const size_type& n,
                                 const size_type& m,
                                 const size_type& num_cons,
                                 const index_type* idx_cons,
                                 const double* x,
                                 bool new_x, double* cons)
{
  return false;
}

bool hiopFRProbSparse::eval_cons(const size_type& n,
                                 const size_type& m,
                                 const double* x,
                                 bool new_x,
                                 double* cons)
{
  assert(n == n_);
  assert(m == m_);

  // evaluate c and d
  wrk_x_->copy_from_starting_at(x, 0, n_x_);
  nlp_base_->eval_c_d(*wrk_x_, new_x, *wrk_c_, *wrk_d_);

  wrk_eq_->copy_from_starting_at(x, pe_st_, m_eq_);     //pe
  wrk_eq_->copy_from_starting_at(x, ne_st_, m_eq_);     //ne
  wrk_c_->axpy(-1.0, *wrk_eq_);
  wrk_c_->axpy(1.0, *wrk_eq_);

  wrk_ineq_->copy_from_starting_at(x, pi_st_, m_ineq_); //pi
  wrk_ineq_->copy_from_starting_at(x, ni_st_, m_ineq_); //ni
  wrk_d_->axpy(-1.0, *wrk_ineq_);
  wrk_d_->axpy(1.0, *wrk_ineq_);

  // assemble the full vector
  wrk_c_->copyToStarting(*wrk_dual_, 0);
  wrk_d_->copyToStarting(*wrk_dual_, m_eq_);

  wrk_dual_->copyTo(cons);

  return true;
}

bool hiopFRProbSparse::eval_Jac_cons(const size_type& n, const size_type& m,
                                     const size_type& num_cons,
                                     const index_type* idx_cons,
                                     const double* x,
                                     bool new_x,
                                     const int& nnzJacS,
                                     int* iJacS,
                                     int* jJacS,
                                     double* MJacS)
{
  return false;
}

/// @pre assuming Jac of the original prob is sorted
bool hiopFRProbSparse::eval_Jac_cons(const size_type& n,
                                     const size_type& m,
                                     const double* x,
                                     bool new_x,
                                     const int& nnzJacS,
                                     int* iJacS,
                                     int* jJacS,
                                     double* MJacS)
{
  assert( n == n_);
  assert( m == m_);

  assert(nnzJacS == nlp_base_->get_nnz_Jaceq() + nlp_base_->get_nnz_Jacineq() + 2 * (m_));

  hiopMatrixSparse& Jac_c = dynamic_cast<hiopMatrixSparse&>(*solver_base_.get_Jac_c());
  hiopMatrixSparse& Jac_d = dynamic_cast<hiopMatrixSparse&>(*solver_base_.get_Jac_d());

  // extend Jac to the p and n parts
  if(MJacS != nullptr) {
    // get x for the original problem
    wrk_x_->copy_from_starting_at(x, 0, n_x_);

    // get Jac_c and Jac_d for the x part --- use original Jac_c/Jac_d as buffers
    nlp_base_->eval_Jac_c_d(*wrk_x_, new_x, Jac_c, Jac_d); 
  }

  Jac_cd_->set_Jac_FR(Jac_c, Jac_d, iJacS, jJacS, MJacS);
  
  return true;
}

bool hiopFRProbSparse::eval_Hess_Lagr(const size_type& n,
                                      const size_type& m,
                                      const double* x,
                                      bool new_x,
                                      const double& obj_factor,
                                      const double* lambda,
                                      bool new_lambda,
                                      const int& nnzHSS,
                                      int* iHSS,
                                      int* jHSS,
                                      double* MHSS)
{
  assert(nnzHSS == nnz_Hess_Lag_);

  // shortcut to the original Hess
  hiopMatrixSparse& Hess = dynamic_cast<hiopMatrixSparse&>(*solver_base_.get_Hess_Lagr());

  if(MHSS != nullptr) {
    // get x for the original problem
    wrk_x_->copy_from_starting_at(x, 0, n_x_);

    // split lambda
    wrk_eq_->copy_from_starting_at(lambda, 0, m_eq_);
    wrk_ineq_->copy_from_starting_at(lambda, m_eq_, m_ineq_);

    double obj_factor = 0.0;
    // get Hess for the x part --- use original Hess as buffers
    nlp_base_->eval_Hess_Lagr(*wrk_x_, new_x, obj_factor, *wrk_eq_, *wrk_ineq_, new_lambda, Hess);
    
    // additional diag Hess for x:  zeta*DR^2.*(x-x_ref)
    wrk_x_->axpy(-1.0, *x_ref_);
    wrk_x_->componentMult(*DR_);
    wrk_x_->componentMult(*DR_);
    wrk_x_->scale(zeta_);    
  }

  // extend Hes to the p and n parts
  Hess_cd_->set_Hess_FR(Hess, iHSS, jHSS, MHSS, *wrk_x_);
  
  return true;
}

bool hiopFRProbSparse::get_starting_point(const size_type& n,
                                          const size_type& m,
                                          double* x0,
                                          bool& duals_avail,
                                          double* z_bndL0,
                                          double* z_bndU0,
                                          double* lambda0,
                                          bool& slack_avail,
                                          double *ineq_slack)
{
  assert( n == n_);
  assert( m == m_);

  duals_avail = true;
  slack_avail = true;

  hiopVector* c = solver_base_.get_c();
  hiopVector* d = solver_base_.get_d();
  hiopVector* s = solver_base_.get_it_curr()->get_d();
  hiopVector* zl = solver_base_.get_it_curr()->get_zl();
  hiopVector* zu = solver_base_.get_it_curr()->get_zu();
  const hiopVector& crhs = nlp_base_->get_crhs();

  // x0 = x_ref
  wrk_x_->copyFrom(*x_ref_);

  // s = curr_s
  s->copyTo(ineq_slack);

  /*
  * compute pe (wrk_c_) and ne (wrk_eq_) rom equation (33)
  */
  // firstly use pe as a temp vec
  double tmp_db = mu_/(2*rho_);
  wrk_cbody_->copyFrom(*c);
  wrk_cbody_->axpy(-1.0, crhs);     // wrk_cbody_ = (c-crhs)
  wrk_c_->setToConstant(tmp_db);
  wrk_c_->axpy(-0.5, *wrk_cbody_);   // wrk_c_ = (mu-rho*(c-crhs))/(2*rho)

  // compute ne (wrk_eq_)
  wrk_eq_->copyFrom(*wrk_c_);
  wrk_eq_->componentMult(*wrk_c_);
  wrk_eq_->axpy(tmp_db, *wrk_cbody_);
  wrk_eq_->component_sqrt();
  wrk_eq_->axpy(1.0, *wrk_c_);

  // compute pe (wrk_c_)
  wrk_c_->copyFrom(*wrk_cbody_);
  wrk_c_->axpy(1.0, *wrk_eq_);

  /*
  * compute pi (wrk_d_) and ni (wrk_ineq_) rom equation (33)
  */
  // firstly use pi as a temp vec
  wrk_dbody_->copyFrom(*d);
  wrk_dbody_->axpy(-1.0, *s);        // wrk_dbody_ = (d-s)
  wrk_d_->setToConstant(tmp_db);
  wrk_d_->axpy(-0.5, *wrk_dbody_);   // wrk_c_ = (mu-rho*(d-s))/(2*rho)

  // compute ni (wrk_ineq_)
  wrk_ineq_->copyFrom(*wrk_d_);
  wrk_ineq_->componentMult(*wrk_d_);
  wrk_ineq_->axpy(tmp_db, *wrk_dbody_);
  wrk_ineq_->component_sqrt();
  wrk_ineq_->axpy(1.0, *wrk_d_);

  // compute pi (wrk_d_)
  wrk_d_->copyFrom(*wrk_dbody_);
  wrk_d_->axpy(1.0, *wrk_ineq_);

  /*
  * assemble x0
  */
  wrk_x_->copyToStarting(*wrk_primal_, 0);
  wrk_c_->copyToStarting(*wrk_primal_, n_x_);                         // pe
  wrk_eq_->copyToStarting(*wrk_primal_, n_x_ + m_eq_);                // ne
  wrk_d_->copyToStarting(*wrk_primal_, n_x_ + 2*m_eq_);               // pi
  wrk_ineq_->copyToStarting(*wrk_primal_, n_x_ + 2*m_eq_ + m_ineq_);  // ni

  wrk_primal_->copyTo(x0);

  /* initialize the dual variables for the variable bounds*/
  // get z = min(rho, z_base)
  wrk_x_->copyFrom(*zl);
  wrk_x_->component_min(rho_);

  // compute zl for p and n = mu*(p0)^-1
  wrk_c_->invert();
  wrk_c_->scale(mu_);
  wrk_eq_->invert();
  wrk_eq_->scale(mu_);
  wrk_d_->invert();
  wrk_d_->scale(mu_);
  wrk_ineq_->invert();
  wrk_ineq_->scale(mu_);

  // assemble zl
  wrk_x_->copyToStarting(*wrk_primal_, 0);
  wrk_c_->copyToStarting(*wrk_primal_, n_x_);                         // pe
  wrk_eq_->copyToStarting(*wrk_primal_, n_x_ + m_eq_);                // ne
  wrk_d_->copyToStarting(*wrk_primal_, n_x_ + 2*m_eq_);               // pi
  wrk_ineq_->copyToStarting(*wrk_primal_, n_x_ + 2*m_eq_ + m_ineq_);  // ni
  wrk_primal_->copyTo(z_bndL0);

  // get zu
  wrk_primal_->setToZero();
  wrk_x_->copyFrom(*zu);
  wrk_x_->component_min(rho_);
  wrk_x_->copyToStarting(*wrk_primal_, 0);
  wrk_primal_->copyTo(z_bndU0);

  // set lambda to 0 --- this will be updated by lsq later.
  // Need to have this since we set duals_avail to true, in order to initialize zl and zu
  wrk_dual_->setToZero();
  wrk_dual_->copyTo(lambda0);
  return true;
}

bool hiopFRProbSparse::iterate_callback(int iter,
                                        double obj_value,
                                        double logbar_obj_value,
                                        int n,
                                        const double* x,
                                        const double* z_L,
                                        const double* z_U,
                                        int m_ineq,
                                        const double* s,
                                        int m,
                                        const double* g,
                                        const double* lambda,
                                        double inf_pr,
                                        double inf_du,
                                        double onenorm_pr_,
                                        double mu,
                                        double alpha_du,
                                        double alpha_pr,
                                        int ls_trials)
{
  assert(n_ == n);
  assert(m_ineq_ == m_ineq);

  const hiopVector& crhs = nlp_base_->get_crhs();

  // evaluate c_body and d_body in base problem
  wrk_x_->copy_from_starting_at(x, 0, n_x_);
  wrk_d_->copy_from_starting_at(s, 0, m_ineq_);
  nlp_base_->eval_c_d(*wrk_x_, true, *wrk_cbody_, *wrk_dbody_);

  // compute theta for base problem
  wrk_cbody_->axpy(-1.0, crhs);           // wrk_cbody_ = (c-crhs)
  wrk_dbody_->axpy(-1.0, *wrk_d_);        // wrk_dbody_ = (d-s)

  double theta_ori = 0.0;
  theta_ori += wrk_cbody_->onenorm();
  theta_ori += wrk_dbody_->onenorm();

  // check if restoration phase should be discontinued

  // termination condition 1) theta_curr <= kappa_resto*theta_ref
  if(theta_ori <= nlp_base_->options->GetNumeric("kappa_resto")*theta_ref_ && iter>0) {
    // termination condition 2) (theta and logbar) are not in the original filter
    // check (original) filter condition
    if(!solver_base_.filter_contains(onenorm_pr_, logbar_obj_value)) {
      // terminate FR
      hiopIterate* it_next = solver_base_.get_it_trial();

      // set next iter->x to x from FR
      wrk_x_->copy_from_starting_at(x, 0, n_x_);
      it_next->get_x()->copyFrom(*wrk_x_);

      // set next iter->d (the slack for ineqaulities) to s from FR
      wrk_d_->copy_from_starting_at(s, 0, m_ineq_);
      it_next->get_d()->copyFrom(*wrk_d_);

      return false;
    }
  }

  mu_ = mu;
  zeta_ = std::sqrt(mu_);

  return true;
}

bool hiopFRProbSparse::force_update(double obj_value,
                                    const int n,
                                    double* x,
                                    double* z_L,
                                    double* z_U,
                                    const int m,
                                    double* g,
                                    double* lambda,
                                    double& mu,
                                    double& alpha_du,
                                    double& alpha_pr)
{
  // this function is used in FR in FR, see eq (33)
  assert( n == n_);

  hiopVector* c = solver_base_.get_c();
  hiopVector* d = solver_base_.get_d();
  hiopVector* s = solver_base_.get_it_curr()->get_d();
  const hiopVector& crhs = nlp_base_->get_crhs();

  // x is fixed
  wrk_x_->copy_from_starting_at(x, 0, n_x_);

  /*
  * compute pe (wrk_c_) and ne (wrk_eq_) rom equation (33)
  */
  // firstly use pe as a temp vec
  double tmp_db = mu_/(2*rho_);
  wrk_cbody_->copyFrom(*c);
  wrk_cbody_->axpy(-1.0, crhs);     // wrk_cbody_ = (c-crhs)
  wrk_c_->setToConstant(tmp_db);
  wrk_c_->axpy(-0.5, *wrk_cbody_);   // wrk_c_ = (mu-rho*(c-crhs))/(2*rho)

  // compute ne (wrk_eq_)
  wrk_eq_->copyFrom(*wrk_c_);
  wrk_eq_->componentMult(*wrk_c_);
  wrk_eq_->axpy(tmp_db, *wrk_cbody_);
  wrk_eq_->component_sqrt();
  wrk_eq_->axpy(1.0, *wrk_c_);

  // compute pe (wrk_c_)
  wrk_c_->copyFrom(*wrk_cbody_);
  wrk_c_->axpy(1.0, *wrk_eq_);

  /*
  * compute pi (wrk_d_) and ni (wrk_ineq_) rom equation (33)
  */
  // firstly use pi as a temp vec
  wrk_dbody_->copyFrom(*d);
  wrk_dbody_->axpy(-1.0, *s);        // wrk_dbody_ = (d-s)
  wrk_d_->setToConstant(tmp_db);
  wrk_d_->axpy(-0.5, *wrk_dbody_);   // wrk_c_ = (mu-rho*(d-s))/(2*rho)

  // compute ni (wrk_ineq_)
  wrk_ineq_->copyFrom(*wrk_d_);
  wrk_ineq_->componentMult(*wrk_d_);
  wrk_ineq_->axpy(tmp_db, *wrk_dbody_);
  wrk_ineq_->component_sqrt();
  wrk_ineq_->axpy(1.0, *wrk_d_);

  // compute pi (wrk_d_)
  wrk_d_->copyFrom(*wrk_dbody_);
  wrk_d_->axpy(1.0, *wrk_ineq_);

  /*
  * assemble x
  */
  wrk_x_->copyToStarting(*wrk_primal_, 0);
  wrk_c_->copyToStarting(*wrk_primal_, n_x_);
  wrk_eq_->copyToStarting(*wrk_primal_, n_x_ + m_eq_);
  wrk_d_->copyToStarting(*wrk_primal_, n_x_ + 2*m_eq_);
  wrk_ineq_->copyToStarting(*wrk_primal_, n_x_ + 2*m_eq_ + m_ineq_);

  wrk_primal_->copyTo(x);

  return true;
}






/* 
*  Specialized interface for feasibility restoration problem with MDS blocks in the Jacobian and Hessian.
*/
hiopFRProbMDS::hiopFRProbMDS(hiopAlgFilterIPMBase& solver_base)
  : solver_base_(solver_base)
{
  nlp_base_ = dynamic_cast<hiopNlpMDS*>(solver_base.get_nlp());
  n_x_ = nlp_base_->n();
  n_x_sp_ = nlp_base_->nx_sp();
  n_x_de_ = nlp_base_->nx_de();
  m_eq_ = nlp_base_->m_eq();
  m_ineq_ = nlp_base_->m_ineq();

  n_ = n_x_ + 2*m_eq_ + 2*m_ineq_;
  m_ = m_eq_ + m_ineq_;

  pe_st_ = n_x_;
  ne_st_ = pe_st_ + m_eq_;
  pi_st_ = ne_st_ + m_eq_;
  ni_st_ = pi_st_ + m_ineq_;

  x_ref_ = solver_base.get_it_curr()->get_x();

  // build vector VR
  DR_ = x_ref_->new_copy();
  DR_->component_abs();
  DR_->invert();
  DR_->component_min(1.0);

  wrk_x_ = x_ref_->alloc_clone();
  wrk_c_ = LinearAlgebraFactory::createVector(m_eq_);
  wrk_d_ = LinearAlgebraFactory::createVector(m_ineq_);
  wrk_eq_ = LinearAlgebraFactory::createVector(m_eq_);
  wrk_ineq_ = LinearAlgebraFactory::createVector(m_ineq_);
  wrk_cbody_ = LinearAlgebraFactory::createVector(m_eq_);
  wrk_dbody_ = LinearAlgebraFactory::createVector(m_ineq_);
  wrk_primal_ = LinearAlgebraFactory::createVector(n_);
  wrk_dual_ = LinearAlgebraFactory::createVector(m_);

  // nnz for sparse matrices;
  nnz_sp_Jac_c_ = nlp_base_->get_nnz_sp_Jaceq() + 2 * m_eq_;
  nnz_sp_Jac_d_ = nlp_base_->get_nnz_sp_Jacineq() + 2 * m_ineq_;

  // not sure i Hess has diagonal terms, compute nnz_hess here
  // assuming hess is in upper_triangular form
  hiopMatrixSparse* Hess_SS = dynamic_cast<hiopMatrixSparse*>(solver_base_.get_Hess_Lagr());
  nnz_sp_Hess_Lag_SS_ = n_x_ + Hess_SS->numberOfOffDiagNonzeros();

  // set mu0 to be the maximun of the current barrier parameter mu and norm_inf(|c|)*/
  theta_ref_ = solver_base_.get_resid()->get_theta(); //at current point, i.e., reference point
  mu_ = solver_base.get_mu();
  mu_ = std::max(mu_, solver_base_.get_resid()->get_nrmInf_bar_feasib());

  zeta_ = std::sqrt(mu_);
  rho_ = 1000; // FIXME: make this as an user option
}

hiopFRProbMDS::~hiopFRProbMDS()
{
  delete wrk_x_;
  delete wrk_c_;
  delete wrk_d_;
  delete wrk_eq_;
  delete wrk_ineq_;
  delete wrk_cbody_;
  delete wrk_dbody_;
  delete wrk_primal_;
  delete wrk_dual_;
  delete DR_;
}

bool hiopFRProbMDS::get_prob_sizes(long long& n, long long& m)
{
  n = n_;
  m = m_;
  return true;
}

bool hiopFRProbMDS::get_vars_info(const long long& n, double *xlow, double* xupp, NonlinearityType* type)
{
  assert(n == n_);

  const hiopVector& xl = nlp_base_->get_xl();
  const hiopVector& xu = nlp_base_->get_xu();
  const NonlinearityType* var_type = nlp_base_->get_var_type();

  // x, p and n
  wrk_primal_->setToConstant(0.0);
  xl.copyToStarting(*wrk_primal_,0);
  wrk_primal_->copyTo(xlow);

  wrk_primal_->setToConstant(1e+20);
  xu.copyToStarting(*wrk_primal_,0);
  wrk_primal_->copyTo(xupp);

  // x
  for(long long i = 0; i < n_x_; ++i) {
    type[i] = var_type[i];
  }
  // p and n
  for(long long i = n_x_; i < n_; ++i) {
    type[i] = hiopLinear;
  }
  return true;
}

bool hiopFRProbMDS::get_cons_info(const long long& m, double* clow, double* cupp, NonlinearityType* type)
{
  assert(m == m_);
  assert(m_eq_ + m_ineq_ == m_);
  const hiopVector& crhs = nlp_base_->get_crhs();
  const hiopVector& dl = nlp_base_->get_dl();
  const hiopVector& du = nlp_base_->get_du();
  const NonlinearityType* cons_eq_type = nlp_base_->get_cons_eq_type();
  const NonlinearityType* cons_ineq_type = nlp_base_->get_cons_ineq_type();

  wrk_dual_->setToConstant(0.0);

  // assemble wrk_dual_ = [crhs; dl] for lower bounds
  crhs.copyToStarting(*wrk_dual_, 0);
  dl.copyToStarting(*wrk_dual_, (int)m_eq_);
  wrk_dual_->copyTo(clow);

  // assemble wrk_dual_ = [crhs; du] for upper bounds
  du.copyToStarting(*wrk_dual_, (int)m_eq_);
  wrk_dual_->copyTo(cupp);

  for(long long i = 0; i < m_eq_; ++i) {
    type[i] = cons_eq_type[i];
  }
  for(long long i = m_eq_; i < m_; ++i) {
    type[i] = cons_ineq_type[i-m_eq_];
  }
  return true;
}

bool hiopFRProbMDS::get_sparse_blocks_info(int& nx,
                                                   int& nnz_sparse_Jaceq,
                                                   int& nnz_sparse_Jacineq,
                                                   int& nnz_sparse_Hess_Lagr)
{
  nx = n_;
  nnz_sparse_Jaceq = nnz_Jac_c_;
  nnz_sparse_Jacineq = nnz_Jac_d_;
  nnz_sparse_Hess_Lagr = nnz_Hess_Lag_;
  return true;
}

bool hiopFRProbMDS::eval_f(const long long& n, const double* x, bool new_x, double& obj_value)
{
  assert(n == n_);
  obj_value = 0.;

  // rho*sum(p+n)
  for(auto i = n_x_; i < n_; ++i) {
    obj_value += x[i];
  }
  obj_value *= rho_;

  // zeta/2*[DR*(x-x_ref)]^2
  wrk_x_->copy_from_starting_at(x, 0, n_x_);
  wrk_x_->axpy(-1.0, *x_ref_);
  wrk_x_->componentMult(*DR_);
  double wrk_db = wrk_x_->twonorm();

  obj_value += 0.5 * zeta_ * wrk_db * wrk_db;

  return true;
}

bool hiopFRProbMDS::eval_grad_f(const long long& n, const double* x, bool new_x, double* gradf)
{
  assert(n == n_);

  // p and n
  wrk_primal_->setToConstant(rho_);

  // x
  wrk_x_->copy_from_starting_at(x, 0, n_x_);
  wrk_x_->axpy(-1.0, *x_ref_);
  wrk_x_->componentMult(*DR_);
  wrk_x_->componentMult(*DR_);
  wrk_x_->scale(zeta_);
  wrk_x_->copyToStarting(*wrk_primal_,0);

  wrk_primal_->copyTo(gradf);

  return true;
}

bool hiopFRProbMDS::eval_cons(const long long& n,
                                      const long long& m,
                                      const long long& num_cons,
                                      const long long* idx_cons,
                                      const double* x,
                                      bool new_x, double* cons)
{
  return false;
}

bool hiopFRProbMDS::eval_cons(const long long& n,
                                      const long long& m,
                                      const double* x,
                                      bool new_x,
                                      double* cons)
{
  assert(n == n_);
  assert(m == m_);

  // evaluate c and d
  wrk_x_->copy_from_starting_at(x, 0, n_x_);
  nlp_base_->eval_c_d(*wrk_x_, new_x, *wrk_c_, *wrk_d_);

  wrk_eq_->copy_from_starting_at(x, pe_st_, m_eq_);     //pe
  wrk_eq_->copy_from_starting_at(x, ne_st_, m_eq_);     //ne
  wrk_c_->axpy(-1.0, *wrk_eq_);
  wrk_c_->axpy(1.0, *wrk_eq_);

  wrk_ineq_->copy_from_starting_at(x, pi_st_, m_ineq_); //pi
  wrk_ineq_->copy_from_starting_at(x, ni_st_, m_ineq_); //ni
  wrk_d_->axpy(-1.0, *wrk_ineq_);
  wrk_d_->axpy(1.0, *wrk_ineq_);

  // assemble the full vector
  wrk_c_->copyToStarting(*wrk_dual_, 0);
  wrk_d_->copyToStarting(*wrk_dual_, m_eq_);

  wrk_dual_->copyTo(cons);

  return true;
}

bool hiopFRProbMDS::eval_Jac_cons(const long long& n, const long long& m,
                                          const long long& num_cons,
                                          const long long* idx_cons,
                                          const double* x,
                                          bool new_x,
                                          const int& nnzJacS,
                                          int* iJacS,
                                          int* jJacS,
                                          double* MJacS)
{
  return false;
}

/// @pre assuming Jac of the original prob is sorted
bool hiopFRProbMDS::eval_Jac_cons(const long long& n,
                                          const long long& m,
                                          const double* x,
                                          bool new_x,
                                          const int& nnzJacS,
                                          int* iJacS,
                                          int* jJacS,
                                          double* MJacS)
{
  assert( n == n_);
  assert( m == m_);

  assert(nnzJacS == nlp_base_->get_nnz_Jaceq() + nlp_base_->get_nnz_Jacineq() + 2 * (m_));

  hiopMatrixSparse* Jac_c = dynamic_cast<hiopMatrixSparse*>(solver_base_.get_Jac_c());
  hiopMatrixSparse* Jac_d = dynamic_cast<hiopMatrixSparse*>(solver_base_.get_Jac_d());

  // shortcut to the original Jac
  int *irow_c = Jac_c->i_row();
  int *jcol_c = Jac_c->j_col();
  int *irow_d = Jac_d->i_row();
  int *jcol_d = Jac_d->j_col();

  // assuming Jac is sorted!
  int nnz_Jac_c_base = nlp_base_->get_nnz_Jaceq();
  int nnz_Jac_d_base = nlp_base_->get_nnz_Jacineq();

  int k;
  int k_base;
  int last_row_idx;

  // extend Jac to the p and n parts
  if(iJacS != NULL && jJacS != NULL){
    // Jac for d(x) - p + n
    last_row_idx = m_;
    k_base = nnz_Jac_d_base - 1;
    for(k = nnz_Jac_c_ + nnz_Jac_d_ - 1; k >= nnz_Jac_c_ && k_base >= 0; ) {
      int row_in_jac_d = irow_d[k_base];
      int row_idx = row_in_jac_d + m_eq_;
      if(row_idx != last_row_idx) {
        // n
        iJacS[k] = row_idx;
        jJacS[k] = ni_st_ + row_in_jac_d;
        // p
        iJacS[k-1] = row_idx;
        jJacS[k-1] = pi_st_ + row_in_jac_d;

        last_row_idx = row_idx;
        k = k-2;
      } else {
        // x
        iJacS[k] = row_idx;
        jJacS[k] = jcol_d[k_base];
        k--;
        k_base--;
      }
    }
    assert( k == nnz_Jac_c_ - 1 && k_base == -1 );

    // Jac for c(x) - p + n
    last_row_idx = m_eq_;
    k_base = nnz_Jac_c_base - 1;
    for(k = nnz_Jac_c_ - 1; k >= 0 && k_base >= 0; ) {
      int row_idx = irow_c[k_base];
      if(row_idx != last_row_idx) {
        // n
        iJacS[k] = row_idx;
        jJacS[k] = ne_st_ + row_idx;
        // p
        iJacS[k-1] = row_idx;
        jJacS[k-1] = pe_st_ + row_idx;

        last_row_idx = row_idx;
        k = k-2;
      } else {
        // x
        iJacS[k] = row_idx;
        jJacS[k] = jcol_c[k_base];
        k--;
        k_base--;
      }
    }
    assert( k == -1 && k_base == -1 );
  }

  //values for sparse Jacobian if requested by the solver
  if(MJacS != NULL) {
    // get x for the original problem
    wrk_x_->copy_from_starting_at(x, 0, n_x_);

    // get Jac_c and Jac_d for the x part --- use original Jac_c/Jac_d as buffers
    nlp_base_->eval_Jac_c_d(*wrk_x_, new_x, *Jac_c, *Jac_d);

    // shortcut to the original Jac
    double *M_c = Jac_c->M();
    double *M_d = Jac_d->M();

    // Jac for d(x) - p + n
    last_row_idx = m_;
    k_base = nnz_Jac_d_base - 1;
    for(k = nnz_Jac_c_ + nnz_Jac_d_ - 1; k >= nnz_Jac_c_ && k_base >= 0; ) {
      int row_idx = irow_d[k_base] + m_eq_;
      if(row_idx != last_row_idx) {
        // n
        MJacS[k] = 1.0;
        // p
        MJacS[k-1] = -1.0;

        last_row_idx = row_idx;
        k = k-2;
      } else {
        // x
        MJacS[k] = M_d[k_base];
        k--;
        k_base--;
      }
    }
    assert( k == nnz_Jac_c_ - 1 && k_base == -1 );

    // Jac for c(x) - p + n
    last_row_idx = m_eq_;
    k_base = nnz_Jac_c_base - 1;
    for(k = nnz_Jac_c_ - 1; k >= 0 && k_base >= 0; ) {
      int row_idx = irow_c[k_base];
      if(row_idx != last_row_idx) {
        // n
        MJacS[k] = 1.0;
        // p
        MJacS[k-1] = -1.0;

        last_row_idx = row_idx;
        k = k-2;
      } else {
        // x
        MJacS[k] = M_c[k_base];
        k--;
        k_base--;
      }
    }
    assert( k == -1 && k_base == -1 );
  }
  return true;
}

bool hiopFRProbMDS::eval_Hess_Lagr(const long long& n,
                                           const long long& m,
                                           const double* x,
                                           bool new_x,
                                           const double& obj_factor,
                                           const double* lambda,
                                           bool new_lambda,
                                           const int& nnzHSS,
                                           int* iHSS,
                                           int* jHSS,
                                           double* MHSS)
{
  assert(nnzHSS == nnz_Hess_Lag_);

  // shortcut to the original Hess
  hiopMatrixSparse* Hess = dynamic_cast<hiopMatrixSparse*>(solver_base_.get_Hess_Lagr());
  int *irow_h = Hess->i_row();
  int *jcol_h = Hess->j_col();

  // assuming Hess is sorted, and in upper-triangle format
  int nnz_Hess_base = nlp_base_->get_nnz_Hess_Lagr();

  // extend Jac to the p and n parts
  if(iHSS != NULL && jHSS != NULL) {
    int k = 0;
    int k_base = 0;
    int row_idx = 0;

    // Hess for x:  zeta*DR^2.*(x-x_ref)
    for(k_base = 0; k_base < nnz_Hess_base; ) {
      int row_base = irow_h[k_base];
      int col_base = jcol_h[k_base];
      if(row_idx < row_base) {
        // find empty row, insert diagonal entries
        iHSS[k] = row_idx;
        jHSS[k] = row_idx;
        k++;
        row_idx++;
        continue;
      }

      // now we are on a non-empty row
      if(col_base == row_base) {
        // find a diagonal nonzero in the original hess
        iHSS[k] = row_base;
        jHSS[k] = col_base;
        k++;
        k_base++;
        row_idx++;
      } else {
        if(row_idx == row_base) {
          // insert diagonal nonzero into the beginning of this row
          iHSS[k] = row_idx;
          jHSS[k] = row_idx;
          k++;
          row_idx++;
        }

        // copy original off-diag nonzero
        iHSS[k] = row_base;
        jHSS[k] = col_base;
        k++;
        k_base++;
      }
    }
    assert(row_idx == n_x_);
    assert(k_base == nnz_Hess_base);
    assert(k == nnz_Hess_Lag_);
  }

  if(MHSS != NULL) {
    int k = 0;
    int k_base = 0;
    int row_idx = 0;

    // get x for the original problem
    wrk_x_->copy_from_starting_at(x, 0, n_x_);

    // split lambda
    wrk_eq_->copy_from_starting_at(lambda, 0, m_eq_);
    wrk_ineq_->copy_from_starting_at(lambda, m_eq_, m_ineq_);

    double obj_factor = 0.0;
    // get Jac_c and Jac_d for the x part --- use original Jac_c/Jac_d as buffers
    nlp_base_->eval_Hess_Lagr(*wrk_x_, new_x, obj_factor, *wrk_eq_, *wrk_ineq_, new_lambda, *Hess);

    // shortcut to the original Jac
    double *M_h = Hess->M();

    // Hess for x:  zeta*DR^2.*(x-x_ref)
    wrk_x_->axpy(-1.0, *x_ref_);
    wrk_x_->componentMult(*DR_);
    wrk_x_->componentMult(*DR_);
    wrk_x_->scale(zeta_);

    double* x_db = wrk_x_->local_data();

    row_idx = 0;
    for(k_base = 0; k_base < nnz_Hess_base; ) {
      int row_base = irow_h[k_base];
      int col_base = jcol_h[k_base];
      if(row_idx < row_base) {
        // find empty row, insert diagonal entries
        MHSS[k] = x_db[row_idx];
        k++;
        row_idx++;
        continue;
      }

      // now we are on a non-empty row
      if(col_base == row_base) {
        // find a diagonal nonzero in the original hess
        MHSS[k] = M_h[k_base] + x_db[row_idx];
        k++;
        k_base++;
        row_idx++;
      } else {
        if(row_idx == row_base) {
          // insert diagonal nonzero into the beginning of this row
          MHSS[k] = x_db[row_idx];
          k++;
          row_idx++;
        }

        // copy original off-diag nonzero
        MHSS[k] = M_h[k_base];
        k++;
        k_base++;
      }
    }
    assert(row_idx == n_x_);
    assert(k_base == nnz_Hess_base);
    assert(k == nnz_Hess_Lag_);
  }
  return true;
}

bool hiopFRProbMDS::get_starting_point(const long long& n,
                                               const long long& m,
                                               double* x0,
                                               bool& duals_avail,
                                               double* z_bndL0,
                                               double* z_bndU0,
                                               double* lambda0,
                                               bool& slack_avail,
                                               double *ineq_slack)
{
  assert( n == n_);
  assert( m == m_);

  duals_avail = true;
  slack_avail = true;

  hiopVector* c = solver_base_.get_c();
  hiopVector* d = solver_base_.get_d();
  hiopVector* s = solver_base_.get_it_curr()->get_d();
  hiopVector* zl = solver_base_.get_it_curr()->get_zl();
  hiopVector* zu = solver_base_.get_it_curr()->get_zu();
  const hiopVector& crhs = nlp_base_->get_crhs();

  // x0 = x_ref
  wrk_x_->copyFrom(*x_ref_);

  // s = curr_s
  s->copyTo(ineq_slack);

  /*
  * compute pe (wrk_c_) and ne (wrk_eq_) rom equation (33)
  */
  // firstly use pe as a temp vec
  double tmp_db = mu_/(2*rho_);
  wrk_cbody_->copyFrom(*c);
  wrk_cbody_->axpy(-1.0, crhs);     // wrk_cbody_ = (c-crhs)
  wrk_c_->setToConstant(tmp_db);
  wrk_c_->axpy(-0.5, *wrk_cbody_);   // wrk_c_ = (mu-rho*(c-crhs))/(2*rho)

  // compute ne (wrk_eq_)
  wrk_eq_->copyFrom(*wrk_c_);
  wrk_eq_->componentMult(*wrk_c_);
  wrk_eq_->axpy(tmp_db, *wrk_cbody_);
  wrk_eq_->component_sqrt();
  wrk_eq_->axpy(1.0, *wrk_c_);

  // compute pe (wrk_c_)
  wrk_c_->copyFrom(*wrk_cbody_);
  wrk_c_->axpy(1.0, *wrk_eq_);

  /*
  * compute pi (wrk_d_) and ni (wrk_ineq_) rom equation (33)
  */
  // firstly use pi as a temp vec
  wrk_dbody_->copyFrom(*d);
  wrk_dbody_->axpy(-1.0, *s);        // wrk_dbody_ = (d-s)
  wrk_d_->setToConstant(tmp_db);
  wrk_d_->axpy(-0.5, *wrk_dbody_);   // wrk_c_ = (mu-rho*(d-s))/(2*rho)

  // compute ni (wrk_ineq_)
  wrk_ineq_->copyFrom(*wrk_d_);
  wrk_ineq_->componentMult(*wrk_d_);
  wrk_ineq_->axpy(tmp_db, *wrk_dbody_);
  wrk_ineq_->component_sqrt();
  wrk_ineq_->axpy(1.0, *wrk_d_);

  // compute pi (wrk_d_)
  wrk_d_->copyFrom(*wrk_dbody_);
  wrk_d_->axpy(1.0, *wrk_ineq_);

  /*
  * assemble x0
  */
  wrk_x_->copyToStarting(*wrk_primal_, 0);
  wrk_c_->copyToStarting(*wrk_primal_, n_x_);                         // pe
  wrk_eq_->copyToStarting(*wrk_primal_, n_x_ + m_eq_);                // ne
  wrk_d_->copyToStarting(*wrk_primal_, n_x_ + 2*m_eq_);               // pi
  wrk_ineq_->copyToStarting(*wrk_primal_, n_x_ + 2*m_eq_ + m_ineq_);  // ni

  wrk_primal_->copyTo(x0);

  /* initialize the dual variables for the variable bounds*/
  // get z = min(rho, z_base)
  wrk_x_->copyFrom(*zl);
  wrk_x_->component_min(rho_);

  // compute zl for p and n = mu*(p0)^-1
  wrk_c_->invert();
  wrk_c_->scale(mu_);
  wrk_eq_->invert();
  wrk_eq_->scale(mu_);
  wrk_d_->invert();
  wrk_d_->scale(mu_);
  wrk_ineq_->invert();
  wrk_ineq_->scale(mu_);

  // assemble zl
  wrk_x_->copyToStarting(*wrk_primal_, 0);
  wrk_c_->copyToStarting(*wrk_primal_, n_x_);                         // pe
  wrk_eq_->copyToStarting(*wrk_primal_, n_x_ + m_eq_);                // ne
  wrk_d_->copyToStarting(*wrk_primal_, n_x_ + 2*m_eq_);               // pi
  wrk_ineq_->copyToStarting(*wrk_primal_, n_x_ + 2*m_eq_ + m_ineq_);  // ni
  wrk_primal_->copyTo(z_bndL0);

  // get zu
  wrk_primal_->setToZero();
  wrk_x_->copyFrom(*zu);
  wrk_x_->component_min(rho_);
  wrk_x_->copyToStarting(*wrk_primal_, 0);
  wrk_primal_->copyTo(z_bndU0);

  // set lambda to 0 --- this will be updated by lsq later.
  // Need to have this since we set duals_avail to true, in order to initialize zl and zu
  wrk_dual_->setToZero();
  wrk_dual_->copyTo(lambda0);
  return true;
}

bool hiopFRProbMDS::iterate_callback(int iter,
                                             double obj_value,
                                             double logbar_obj_value,
                                             int n,
                                             const double* x,
                                             const double* z_L,
                                             const double* z_U,
                                             int m_ineq,
                                             const double* s,
                                             int m,
                                             const double* g,
                                             const double* lambda,
                                             double inf_pr,
                                             double inf_du,
                                             double onenorm_pr_,
                                             double mu,
                                             double alpha_du,
                                             double alpha_pr,
                                             int ls_trials)
{
  assert(n_ == n);
  assert(m_ineq_ == m_ineq);

  const hiopVector& crhs = nlp_base_->get_crhs();

  // evaluate c_body and d_body in base problem
  wrk_x_->copy_from_starting_at(x, 0, n_x_);
  wrk_d_->copy_from_starting_at(s, 0, m_ineq_);
  nlp_base_->eval_c_d(*wrk_x_, true, *wrk_cbody_, *wrk_dbody_);

  // compute theta for base problem
  wrk_cbody_->axpy(-1.0, crhs);           // wrk_cbody_ = (c-crhs)
  wrk_dbody_->axpy(-1.0, *wrk_d_);        // wrk_dbody_ = (d-s)

  double theta_ori = 0.0;
  theta_ori += wrk_cbody_->onenorm();
  theta_ori += wrk_dbody_->onenorm();

  // check if restoration phase should be discontinued

  // termination condition 1) theta_curr <= kappa_resto*theta_ref
  if(theta_ori <= nlp_base_->options->GetNumeric("kappa_resto")*theta_ref_ && iter>0) {
    // termination condition 2) (theta and logbar) are not in the original filter
    // check (original) filter condition
    if(!solver_base_.filter_contains(onenorm_pr_, logbar_obj_value)) {
      // terminate FR
      hiopIterate* it_next = solver_base_.get_it_trial();

      // set next iter->x to x from FR
      wrk_x_->copy_from_starting_at(x, 0, n_x_);
      it_next->get_x()->copyFrom(*wrk_x_);

      // set next iter->d (the slack for ineqaulities) to s from FR
      wrk_d_->copy_from_starting_at(s, 0, m_ineq_);
      it_next->get_d()->copyFrom(*wrk_d_);

      return false;
    }
  }

  mu_ = mu;
  zeta_ = std::sqrt(mu_);

  return true;
}

bool hiopFRProbMDS::force_update(double obj_value,
                                         const int n,
                                         double* x,
                                         double* z_L,
                                         double* z_U,
                                         const int m,
                                         double* g,
                                         double* lambda,
                                         double& mu,
                                         double& alpha_du,
                                         double& alpha_pr)
{
  // this function is used in FR in FR, see eq (33)
  assert( n == n_);

  hiopVector* c = solver_base_.get_c();
  hiopVector* d = solver_base_.get_d();
  hiopVector* s = solver_base_.get_it_curr()->get_d();
  const hiopVector& crhs = nlp_base_->get_crhs();

  // x is fixed
  wrk_x_->copy_from_starting_at(x, 0, n_x_);

  /*
  * compute pe (wrk_c_) and ne (wrk_eq_) rom equation (33)
  */
  // firstly use pe as a temp vec
  double tmp_db = mu_/(2*rho_);
  wrk_cbody_->copyFrom(*c);
  wrk_cbody_->axpy(-1.0, crhs);     // wrk_cbody_ = (c-crhs)
  wrk_c_->setToConstant(tmp_db);
  wrk_c_->axpy(-0.5, *wrk_cbody_);   // wrk_c_ = (mu-rho*(c-crhs))/(2*rho)

  // compute ne (wrk_eq_)
  wrk_eq_->copyFrom(*wrk_c_);
  wrk_eq_->componentMult(*wrk_c_);
  wrk_eq_->axpy(tmp_db, *wrk_cbody_);
  wrk_eq_->component_sqrt();
  wrk_eq_->axpy(1.0, *wrk_c_);

  // compute pe (wrk_c_)
  wrk_c_->copyFrom(*wrk_cbody_);
  wrk_c_->axpy(1.0, *wrk_eq_);

  /*
  * compute pi (wrk_d_) and ni (wrk_ineq_) rom equation (33)
  */
  // firstly use pi as a temp vec
  wrk_dbody_->copyFrom(*d);
  wrk_dbody_->axpy(-1.0, *s);        // wrk_dbody_ = (d-s)
  wrk_d_->setToConstant(tmp_db);
  wrk_d_->axpy(-0.5, *wrk_dbody_);   // wrk_c_ = (mu-rho*(d-s))/(2*rho)

  // compute ni (wrk_ineq_)
  wrk_ineq_->copyFrom(*wrk_d_);
  wrk_ineq_->componentMult(*wrk_d_);
  wrk_ineq_->axpy(tmp_db, *wrk_dbody_);
  wrk_ineq_->component_sqrt();
  wrk_ineq_->axpy(1.0, *wrk_d_);

  // compute pi (wrk_d_)
  wrk_d_->copyFrom(*wrk_dbody_);
  wrk_d_->axpy(1.0, *wrk_ineq_);

  /*
  * assemble x
  */
  wrk_x_->copyToStarting(*wrk_primal_, 0);
  wrk_c_->copyToStarting(*wrk_primal_, n_x_);
  wrk_eq_->copyToStarting(*wrk_primal_, n_x_ + m_eq_);
  wrk_d_->copyToStarting(*wrk_primal_, n_x_ + 2*m_eq_);
  wrk_ineq_->copyToStarting(*wrk_primal_, n_x_ + 2*m_eq_ + m_ineq_);

  wrk_primal_->copyTo(x);

  return true;
}







};
