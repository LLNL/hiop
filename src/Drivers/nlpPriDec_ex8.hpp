#ifndef HIOP_EXAMPLE_EX8
#define HIOP_EXAMPLE_EX8

#include "hiopInterfacePrimalDecomp.hpp"

#include "hiopInterface.hpp"
#include "hiopNlpFormulation.hpp"
#include "hiopAlgFilterIPM.hpp"

#include <cassert>
#include <cstring> //for memcpy
#include <cstdio>
#include <cmath>
#include <chrono>

/** This class provide an example of what a user of hiop::hiopInterfacePriDecProblem 
 * should implement in order to provide both the base and recourse problem to 
 * hiop::hiopAlgPrimalDecomposition solver
 * 
 * Base case problem f
 * sum 0.5 {(x_i-1)*(x_i-1) : i=1,...,ns} 
 *           x_i >=0
 * Contingency problems r_i
 * r = 1/S * \sum{i=1^S} 0.5*|x+Se_i|^2
 * where {Se_i}_j = S  j=i
 *                = 0  otherwise,  i=1,2,....S
 * For i>ns, Se_i = 0
 *
 * This should produce the analytical solution of x* = 0
 */

using namespace hiop;
class Ex8 : public hiop::hiopInterfaceDenseConstraints
{
public:
  Ex8(int ns_);
  
  Ex8(int ns_, int S_);
  
  Ex8(int ns_, int S_, int nc_);

  Ex8(int ns_, int S_, int nc_,bool include_);
  
  Ex8(int ns_, 
      int S_, 
      bool include, 
      hiopInterfacePriDecProblem::RecourseApproxEvaluator* evaluator);

  virtual ~Ex8();
  
  bool get_prob_sizes(size_type& n, size_type& m);

  virtual bool get_vars_info(const size_type& n, double *xlow, double* xupp, NonlinearityType* type);

  virtual bool get_cons_info(const size_type& m, double* clow, double* cupp, NonlinearityType* type);
  
  virtual bool eval_f(const size_type& n, const double* x, bool new_x, double& obj_value);
  
  virtual bool eval_cons(const size_type& n,
                         const size_type& m, 
                         const size_type& num_cons,
                         const index_type* idx_cons,  
                         const double* x,
                         bool new_x,
                         double* cons);
  
  //sum 0.5 {(x_i-1)*(x_{i}-1) : i=1,...,ns} 
  virtual bool eval_grad_f(const size_type& n, const double* x, bool new_x, double* gradf);
  
  // Implementation of the primal starting point specification //
  virtual bool get_starting_point(const size_type& global_n, double* x0_);
  
  virtual bool get_starting_point(const size_type& n, const size_type& m,
                                  double* x0_,
                                  bool& duals_avail,
                                  double* z_bndL0, double* z_bndU0,
                                  double* lambda0);
  
  // pass the COMM_SELF communicator since this example is only intended to run inside 1 MPI process //
  virtual bool get_MPI_comm(MPI_Comm& comm_out);
  virtual bool eval_Jac_cons(const size_type& n, const size_type& m,
                             const size_type& num_cons, const index_type* idx_cons,  
                             const double* x, bool new_x, double* Jac); 
  
  virtual bool quad_is_defined();
  
  virtual bool set_quadratic_terms(const int& n, 
                                   hiopInterfacePriDecProblem::RecourseApproxEvaluator* evaluator);
  
  virtual bool set_include(bool include);
  
protected:
  int ns,S;
  int nc;
  bool include_r = false;
  hiopInterfacePriDecProblem::RecourseApproxEvaluator* evaluator_;
  
};


class PriDecMasterProblemEx8 : public hiopInterfacePriDecProblem
{
public:
  PriDecMasterProblemEx8(int n, int S)
    : n_(n),
      S_(S),
      obj_(-1e20),
      sol_(nullptr)
  {
    nc_ = n;
    my_nlp = new Ex8(n_,S_);   
  }
  PriDecMasterProblemEx8(int n, int S, int nc)
    : n_(n),
      S_(S),
      nc_(nc),
      obj_(-1e20),
      sol_(nullptr)
  {
    my_nlp = new Ex8(n,S,nc);   
  }
  virtual ~PriDecMasterProblemEx8()
  {
    delete[] sol_;
    delete my_nlp;
  }
  
  virtual hiopSolveStatus solve_master(hiopVector& x,
                                       const bool& include_r,
                                       const double& rval = 0,
                                       const double* grad= 0,
                                       const double* hess = 0,
                                       const char* master_options_file=nullptr);
  
  // The recourse solution is 0.5*(x+Se_i)(x+Se_i)
  virtual bool eval_f_rterm(size_t idx, const int& n,const  double* x, double& rval);
  
  virtual bool eval_grad_rterm(size_t idx, const int& n, double* x, hiopVector& grad);
  
  // Implement with alpha = 1 for now only
  // This function should only be used if quadratic regularization is included
  virtual bool set_recourse_approx_evaluator(const int n, 
                                             hiopInterfacePriDecProblem::RecourseApproxEvaluator* evaluator);
  /** 
   * Returns the number S of recourse terms
   */
  size_t get_num_rterms() const {return S_;}
  size_t get_num_vars() const {return n_;}
  void get_solution(double* x) const 
  {
    for(int i=0; i<n_; i++)
      x[i] = sol_[i];
  }
  double get_objective() {return obj_;}
private:
  size_t n_;
  size_t S_;
  size_t nc_;
  Ex8* my_nlp;
  double obj_;
  double* sol_; 
};

#endif
