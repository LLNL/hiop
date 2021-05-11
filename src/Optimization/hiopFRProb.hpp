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
 * @file hiopFRProb.hpp
 *
 * @author Cosmin G. Petra <petra1@llnl.gov>, LLNL
 * @author Nai-Yuan Chiang <chiang7@llnl.gov>, LLNL
 *
 */

#ifndef HIOP_FR_INTERFACE
#define HIOP_FR_INTERFACE

#include "hiopInterface.hpp"
#include "hiopAlgFilterIPM.hpp"

namespace hiop
{

/** Specialized interface for feasibility restoration problem with sparse blocks in the Jacobian and Hessian.
 *
 * More specifically, this interface is for specifying optimization problem:
 *
 * min f(x)
 *  s.t. g(x) <= or = 0, lb<=x<=ub
 *
 * such that Jacobian w.r.t. x and Hessian of the Lagrangian w.r.t. x are sparse
 *
 * @note this interface is 'local' in the sense that data is not assumed to be
 * distributed across MPI ranks ('get_vecdistrib_info' should return 'false').
 * Acceleration can be however obtained using OpenMP and CUDA via Raja
 * abstraction layer that HiOp uses and via linear solver.
 *
 */
class hiopFRProbSparse : public hiopInterfaceSparse
{
public:
  hiopFRProbSparse(hiopAlgFilterIPMBase& solver_base);
  virtual ~hiopFRProbSparse();

  virtual bool get_prob_sizes(long long& n, long long& m);
  virtual bool get_vars_info(const long long& n, double *xlow, double* xupp, NonlinearityType* type);
  virtual bool get_cons_info(const long long& m, double* clow, double* cupp, NonlinearityType* type);
  virtual bool get_sparse_blocks_info(int& nx,
                                      int& nnz_sparse_Jaceq,
                                      int& nnz_sparse_Jacineq,
                                      int& nnz_sparse_Hess_Lagr);

  virtual bool eval_f(const long long& n, const double* x, bool new_x, double& obj_value);
  virtual bool eval_cons(const long long& n,
                         const long long& m,
                         const long long& num_cons,
                         const long long* idx_cons,
                         const double* x,
                         bool new_x,
                         double* cons);
  virtual bool eval_cons(const long long& n,
                         const long long& m,
                         const double* x,
                         bool new_x,
                         double* cons);
  virtual bool eval_grad_f(const long long& n, const double* x, bool new_x, double* gradf);
  virtual bool eval_Jac_cons(const long long& n, const long long& m,
                             const long long& num_cons,
                             const long long* idx_cons,
                             const double* x,
                             bool new_x,
                             const int& nnzJacS,
                             int* iJacS,
                             int* jJacS,
                             double* MJacS);
  virtual bool eval_Jac_cons(const long long& n,
                             const long long& m,
                             const double* x,
                             bool new_x,
                             const int& nnzJacS,
                             int* iJacS,
                             int* jJacS,
                             double* MJacS);

  virtual bool get_starting_point(const long long& n,
                                  const long long& m,
                                  double* x0,
                                  bool& duals_avail,
                                  double* z_bndL0,
                                  double* z_bndU0,
                                  double* lambda0,
                                  bool& slack_avail,
                                  double* ineq_slack = nullptr);

  virtual bool eval_Hess_Lagr(const long long& n,
                              const long long& m,
                              const double* x,
                              bool new_x,
                              const double& obj_factor,
                              const double* lambda,
                              bool new_lambda,
                              const int& nnzHSS,
                              int* iHSS,
                              int* jHSS,
                              double* MHSS);

  virtual bool iterate_callback(int iter,
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
                                int ls_trials);

  virtual bool force_update(double obj_value,
                            const int n,
                            double* x,
                            double* z_L,
                            double* z_U,
                            const int m,
                            double* g,
                            double* lambda,
                            double& mu,
                            double& alpha_du,
                            double& alpha_pr);
private:
  long long n_;
  long long m_;

  long long n_x_;
  long long m_eq_;
  long long m_ineq_;

  long long nnz_Jac_c_;
  long long nnz_Jac_d_;
  long long nnz_Hess_Lag_;

  hiopAlgFilterIPMBase& solver_base_;
  hiopNlpSparse* nlp_base_;

  hiopVector* x_ref_;
  hiopVector* DR_;
  hiopVector* wrk_x_;
  hiopVector* wrk_c_;
  hiopVector* wrk_d_;
  hiopVector* wrk_eq_;
  hiopVector* wrk_ineq_;
  hiopVector* wrk_cbody_;
  hiopVector* wrk_dbody_;
  hiopVector* wrk_primal_;  // [x pe ne pi ni]
  hiopVector* wrk_dual_;  // [c d]

  double zeta_;
  double theta_ref_;
  double mu_;
  double rho_;

  int pe_st_; // the 1st index of pe in the full primal space
  int ne_st_; // the 1st index of ne in the full primal space
  int pi_st_; // the 1st index of pi in the full primal space
  int ni_st_; // the 1st index of ni in the full primal space
};


} //end of namespace
#endif
