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

#ifndef HIOP_NLP_FORMULATION
#define HIOP_NLP_FORMULATION

#include "hiopInterface.hpp"
#include "hiopVector.hpp"
#include "hiopMatrix.hpp"
#include "hiopMatrixMDS.hpp"

#ifdef HIOP_USE_MPI
#include "mpi.h"  
#endif

#include "hiopNlpTransforms.hpp"

#include "hiopRunStats.hpp"
#include "hiopLogger.hpp"
#include "hiopOptions.hpp"

#include <cstring>

namespace hiop
{

/** Class for a general NlpFormulation with general constraints and bounds on the variables. 
 * This class also  acts as a factory for linear algebra objects (derivative 
 * matrices, KKT system) whose types are decided based on the hiopInterfaceXXX object passed in the
 * constructor.
 * 
 * This formulation assumes that optimiz variables, rhs, and gradient are VECTORS: contiguous 
 * double arrays for which only local part is accessed (no inter-process comm).
 * Derivatives are generic MATRICES, whose type depend on 
 *    i.  the NLP formulation (sparse general or NLP with few dense constraints) 
 *   ii. the interface provided (general sparse (not yet supported), mixed sparse-dense, or dense
 * constraints).
 * Exact matching of MATRICES and hiopInterface is to be done by specializations of this class.
 */
class hiopNlpFormulation
{
public:
  hiopNlpFormulation(hiopInterfaceBase& interface);
  virtual ~hiopNlpFormulation();

  virtual bool finalizeInitialization();

  /**
   * Wrappers for the interface calls. 
   * Can be overridden for specialized formulations required by the algorithm.
   */
  virtual bool eval_f(double* x, bool new_x, double& f);
  virtual bool eval_grad_f(double* x, bool new_x, double* gradf);
  
  virtual bool eval_c(double* x, bool new_x, double* c);
  virtual bool eval_d(double* x, bool new_x, double* d);
  virtual bool eval_c_d(double* x, bool new_x, double* c, double* d);
  /* the implementation of the next two methods depends both on the interface and on the formulation */
  virtual bool eval_Jac_c(double* x, bool new_x, hiopMatrix& Jac_c)=0;
  virtual bool eval_Jac_d(double* x, bool new_x, hiopMatrix& Jac_d)=0;
  virtual bool eval_Jac_c_d(double* x, bool new_x, hiopMatrix& Jac_c, hiopMatrix& Jac_d);
protected:
  //calls specific hiopInterfaceXXX::eval_Jac_cons and deals with specializations of hiopMatrix arguments
  virtual bool eval_Jac_c_d_interface_impl(double* x, bool new_x, hiopMatrix& Jac_c, hiopMatrix& Jac_d) = 0;
public:
  virtual bool eval_Hess_Lagr(const double* x, bool new_x, 
			      const double& obj_factor,  
			      const double* lambda_eq, 
			      const double* lambda_ineq, 
			      bool new_lambdas, 
			      hiopMatrix& Hess_L)=0;
  /* starting point */
  virtual bool get_starting_point(hiopVector& x0,
				  bool& duals_avail,
				  hiopVector& zL0, hiopVector& zU0,
				  hiopVector& yc0, hiopVector& yd0);

  /** linear algebra factory */
  virtual hiopVector* alloc_primal_vec() const;
  virtual hiopVector* alloc_dual_eq_vec() const;
  virtual hiopVector* alloc_dual_ineq_vec() const;
  virtual hiopVector* alloc_dual_vec() const;
  /* the implementation of the next two methods depends both on the interface and on the formulation */
  virtual hiopMatrix* alloc_Jac_c() = 0;
  virtual hiopMatrix* alloc_Jac_d() = 0;
  virtual hiopMatrix* alloc_Jac_cons() = 0;
  virtual hiopMatrix* alloc_Hess_Lagr() = 0;

  virtual
  void user_callback_solution(hiopSolveStatus status,
			      const hiopVector& x,
			      const hiopVector& z_L, const hiopVector& z_U,
			      const hiopVector& c, const hiopVector& d,
			      const hiopVector& yc, const hiopVector& yd,
			      double obj_value);

  virtual
  bool user_callback_iterate(int iter,
			     double obj_value,
			     const hiopVector& x,
			     const hiopVector& z_L, const hiopVector& z_U,
			     const hiopVector& c, const hiopVector& d,
			     const hiopVector& yc, const hiopVector& yd,
			     double inf_pr, double inf_du, double mu,
			     double alpha_du, double alpha_pr, int ls_trials);
  
  /** const accessors */
  inline const hiopVectorPar& get_xl ()  const { return *xl;   }
  inline const hiopVectorPar& get_xu ()  const { return *xu;   }
  inline const hiopVectorPar& get_ixl()  const { return *ixl;  }
  inline const hiopVectorPar& get_ixu()  const { return *ixu;  }
  inline const hiopVectorPar& get_dl ()  const { return *dl;   }
  inline const hiopVectorPar& get_du ()  const { return *du;   }
  inline const hiopVectorPar& get_idl()  const { return *idl;  }
  inline const hiopVectorPar& get_idu()  const { return *idu;  }
  inline const hiopVectorPar& get_crhs() const { return *c_rhs;}

  /** const accessors */
  inline long long n() const      {return n_vars;}
  inline long long m() const      {return n_cons;}
  inline long long m_eq() const   {return n_cons_eq;}
  inline long long m_ineq() const {return n_cons_ineq;}
  inline long long n_low() const  {return n_bnds_low;}
  inline long long n_upp() const  {return n_bnds_upp;}
  inline long long m_ineq_low() const {return n_ineq_low;}
  inline long long m_ineq_upp() const {return n_ineq_upp;}
  inline long long n_complem()  const {return m_ineq_low()+m_ineq_upp()+n_low()+n_upp();}

  inline long long n_local() const{return xl->get_local_size();}
  inline long long n_low_local() const {return n_bnds_low_local;}
  inline long long n_upp_local() const {return n_bnds_upp_local;}

  /* methods for transforming the internal objects to corresponding user objects */
  inline double user_obj(double hiop_f) { return nlp_transformations.applyToObj(hiop_f); }
  inline void   user_x(hiopVectorPar& hiop_x, double* user_x) 
  { 
    double *hiop_xa = hiop_x.local_data();
    double *user_xa = nlp_transformations.applyTox(hiop_xa,/*new_x=*/true); 
    //memcpy(user_x, user_xa, hiop_x.get_local_size()*sizeof(double));
    memcpy(user_x, user_xa, nlp_transformations.n_post_local()*sizeof(double));
  }

  /* copies/unpacks duals of the bounds and of constraints from 'it' to the three arrays */
  void get_dual_solutions(const hiopIterate& it,
			  double* zl_a,
			  double* zu_a,
			  double* lambda_a);
  
  /* packs constraint rhs or constraint multipliers into one array based on the internal mappings 
   * 'cons_eq_mapping_'and 'cons_ineq_mapping_ */
  void copy_EqIneq_to_cons(const hiopVector& yc,
			   const hiopVector& yd,
			   int num_cons, //size of 'cons'
			   double* cons);
  
  /* outputing and debug-related functionality*/
  hiopLogger* log;
  hiopRunStats runStats;
  hiopOptions* options;
  //prints a summary of the problem
  virtual void print(FILE* f=NULL, const char* msg=NULL, int rank=-1) const;
#ifdef HIOP_USE_MPI
  inline MPI_Comm get_comm() const { return comm; }
  inline int      get_rank() const { return rank; }
  inline int      get_num_ranks() const { return num_ranks; }
#endif
protected:
#ifdef HIOP_USE_MPI
  MPI_Comm comm;
  int rank, num_ranks;
  bool mpi_init_called;
#endif

  /* problem data */
  //various dimensions
  long long n_vars, n_cons, n_cons_eq, n_cons_ineq;
  long long n_bnds_low, n_bnds_low_local, n_bnds_upp, n_bnds_upp_local, n_ineq_low, n_ineq_upp;
  long long n_bnds_lu, n_ineq_lu;
  hiopVectorPar *xl, *xu, *ixu, *ixl; //these will/can be global, memory distributed
  hiopInterfaceBase::NonlinearityType* vars_type; //C array containing the types for local vars

  hiopVectorPar *c_rhs; //local
  hiopInterfaceBase::NonlinearityType* cons_eq_type;

  hiopVectorPar *dl, *du,  *idl, *idu; //these will be local
  hiopInterfaceBase::NonlinearityType* cons_ineq_type;
  
  // keep track of the constraints indexes in the original, user's formulation
  long long *cons_eq_mapping_, *cons_ineq_mapping_; 

  //options for which this class was setup
  std::string strFixedVars; //"none", "fixed", "relax"
  double dFixedVarsTol;

  //internal NLP transformations (currently fixing/relaxing variables implemented)
  hiopNlpTransformations nlp_transformations;

#ifdef HIOP_USE_MPI
  //inter-process distribution of vectors
  long long* vec_distrib;
#endif

  /* User provided interface */
  hiopInterfaceBase& interface_base;

  /**
   * Flag to indicate whether to use evaluate all constraints once or separately for equalities
   * or inequalities. Possible values
   * -1 : not initialized/not decided
   *  0 : separately
   *  1 : at once
   */
  int cons_eval_type_;
  
  /** 
   * Internal buffer for constraints. Used only when constraints and Jacobian are evaluated at 
   * once (cons_eval_type_==1), otherwise NULL.
   */
  double* cons_body_;
  
  /** 
   * Internal buffer for the Jacobian. Used only when constraints and Jacobian are evaluated at 
   * once (cons_eval_type_==1), otherwise NULL.
   */
  hiopMatrix* cons_Jac_;

  /** 
   * Internal buffer for the multipliers of the constraints use to copy the multipliers of eq. and
   * ineq. into and to return it to the user via @user_callback_solution and @user_callback_iterate
   */
  double* cons_lambdas_;
private:
  hiopNlpFormulation(const hiopNlpFormulation& s) : interface_base(s.interface_base) {};
};

/* *************************************************************************
 * Class is for NLPs that has a small number of general/dense constraints *
 * Splits the constraints in ineq and eq.
 * *************************************************************************
 */
class hiopNlpDenseConstraints : public hiopNlpFormulation
{
public:
  hiopNlpDenseConstraints(hiopInterfaceDenseConstraints& interface);
  virtual ~hiopNlpDenseConstraints();

  virtual bool finalizeInitialization();

  virtual bool eval_Jac_c(double* x, bool new_x, hiopMatrix& Jac_c);
  virtual bool eval_Jac_d(double* x, bool new_x, hiopMatrix& Jac_d);
  /* specialized evals to avoid overhead of dynamic cast. Generic variants available above. */
  virtual bool eval_Jac_c(double* x, bool new_x, double** Jac_c);
  virtual bool eval_Jac_d(double* x, bool new_x, double** Jac_d);
protected:
  //calls specific hiopInterfaceXXX::eval_Jac_cons and deals with specializations of
  //hiopMatrix arguments
  virtual bool eval_Jac_c_d_interface_impl(double* x, bool new_x, hiopMatrix& Jac_c, hiopMatrix& Jac_d);
public:
  virtual bool eval_Hess_Lagr(const double* x,
			      bool new_x,
			      const double& obj_factor, 
			      const double* lambda_eq,
			      const double* lambda_ineq,
			      bool new_lambda, 
			      hiopMatrix& Hess_L)
  {
    //silently ignore the call since we're in the quasi-Newton case
    return true;
  }

  virtual hiopMatrixDense* alloc_Jac_c();
  virtual hiopMatrixDense* alloc_Jac_d();
  virtual hiopMatrixDense* alloc_Jac_cons();
  //returns hiopHessianLowRank which (fakely) inherits from hiopMatrix
  virtual hiopMatrix* alloc_Hess_Lagr();

  /* this is in general for a dense matrix with n_vars cols and a small number of 
   * 'nrows' rows. The second argument indicates how much total memory should the
   * matrix (pre)allocate.
   */
  virtual hiopMatrixDense* alloc_multivector_primal(int nrows, int max_rows=-1) const;

private:
  /* interface implemented and provided by the user */
  hiopInterfaceDenseConstraints& interface;
};



/* *************************************************************************
 * Class is for general NLPs that have mixed sparse-dense (MDS) derivatives
 * blocks. 
 * *************************************************************************
 */
class hiopNlpMDS : public hiopNlpFormulation
{
public:
  hiopNlpMDS(hiopInterfaceMDS& interface_)
    : hiopNlpFormulation(interface_), interface(interface_)
  {
    _buf_lambda = new hiopVectorPar(0);
  }
  virtual ~hiopNlpMDS() 
  {
    delete _buf_lambda;
  }

  virtual bool finalizeInitialization();

  virtual bool eval_Jac_c(double* x, bool new_x, hiopMatrix& Jac_c);
  virtual bool eval_Jac_d(double* x, bool new_x, hiopMatrix& Jac_d);


protected:
  //calls specific hiopInterfaceXXX::eval_Jac_cons and deals with specializations of hiopMatrix arguments
  virtual bool eval_Jac_c_d_interface_impl(double* x, bool new_x, hiopMatrix& Jac_c, hiopMatrix& Jac_d);
public:
  virtual bool eval_Hess_Lagr(const double* x,
			      bool new_x,
			      const double& obj_factor,
			      const double* lambda_eq,
			      const double* lambda_ineq,
			      bool new_lambdas,
			      hiopMatrix& Hess_L);
  
  virtual hiopMatrix* alloc_Jac_c() 
  {
    assert(n_vars == nx_sparse+nx_dense);
    return new hiopMatrixMDS(n_cons_eq, nx_sparse, nx_dense, nnz_sparse_Jaceq);
  }
  virtual hiopMatrix* alloc_Jac_d() 
  {
    assert(n_vars == nx_sparse+nx_dense);
    return new hiopMatrixMDS(n_cons_ineq, nx_sparse, nx_dense, nnz_sparse_Jacineq);
  }
  virtual hiopMatrix* alloc_Jac_cons()
  {
    assert(n_vars == nx_sparse+nx_dense);
    return new hiopMatrixMDS(n_cons, nx_sparse, nx_dense, nnz_sparse_Jaceq+nnz_sparse_Jacineq);
  }
  virtual hiopMatrix* alloc_Hess_Lagr()
  {
    assert(0==nnz_sparse_Hess_Lagr_SD);
    return new hiopMatrixSymBlockDiagMDS(nx_sparse, nx_dense, nnz_sparse_Hess_Lagr_SS);
  }
  virtual long long nx_sp() const { return nx_sparse; }
  virtual long long nx_de() const { return nx_dense; }
private:
  hiopInterfaceMDS& interface;
  int nx_sparse, nx_dense;
  int nnz_sparse_Jaceq, nnz_sparse_Jacineq;
  int nnz_sparse_Hess_Lagr_SS, nnz_sparse_Hess_Lagr_SD;

  hiopVectorPar* _buf_lambda;
};

}
#endif
