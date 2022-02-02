#ifndef HIOP_EXAMPLE_EX9_SPARSE
#define HIOP_EXAMPLE_EX9_SPARSE

#include <hiopVectorInt.hpp>
//base interface (NLP specification for primal decomposable problems)
#include "hiopInterfacePrimalDecomp.hpp"

//basecase sparse NLP
#include "nlpPriDec_ex9_user_basecase.hpp"

//recourse sparse NLP
#include "nlpPriDec_ex9_user_recourse_sparse.hpp"
/**
 *
 * An example of the use of hiop::hiopInterfacePriDecProblem interface of HiOp.
 * This interface is used to specify and solve <i>primal decomposable problems</i>.
 * For a detailed mathematical form of such problems, please see
 * hiop::hiopInterfacePriDecProblem.
 *
 * The basecase NLP problem is given by Ex6 and the recourse problems are nonconvex
 * NLPs expressing the distance from the basecase variables `x` to a nonconvex nonlinear
 * closed set `S` of nonsmooth boundary. This set is essentially the intersection of the
 * exterior of a ball centered at (1,0,...,0) with the half-plane y_1>=1, where $y_1$
 * is the first dimension of the vector space containing `S`. In addition, `S` is "cut"
 * by half-planes; this is done to mimic a large number of constraints in the recourse
 * NLP and allow varying degrees of controlability of the dimension of the dual space,
 * which is used for performance evaluation purposes.
 *
 * Mathematically, the recourse function r_i(x) is defined based on the recourse problem
 *
 *     r_i(x;\xi^i) = 1/S *  min_y 0.5 || y - x ||^2 such that 
 * 
 *                   (1-y_1 + \xi^i_1)^2 + \sum_{k=2}^{n_S} (y_k+\xi^i_k)^2 
 * 
 *                                       + \sum_{k=n_S+1}^{n_y} y_k^2 >= 1 
 * 
 *                   y_k - y_{k-1} >=0, k=2, ..., n_y
 *
 *                   y_1 >=0
 * 
 * Eventually each of @f$ \xi^i_1, \xi^i_2, ..., \xi^i_{n_S} @f$  can be withdrawn from U[-0.25, 0.25]
 * and the size n_S of the sample satisfying 1<=n_S<=n_y. They are set to 1.0 for now.
 *
 * When $S$ samples (\xi^i_1, \xi^i_2, ..., \xi^i_{n_S}), i=1,\ldots,S, are used the
 * primal decomposable problem looks like 
 *
 *     min_x basecase(x) +  1/S \sum_{i=1}^S r_i(x;\xi_i)
 *
 * The above problem can be viewed as the sample average approximation of the two-stage
 * stochastic programming problem
 *
 *      min_x basecase(x) +  E_\xi[ r(x,\xi) | \xi ~ U[-0.25,0.25]]
 * 
 * where the random function r(x;\xi) is defined similarily to r_i(x;\xi) (excepting the
 * scaling by 1/S).
 *
 * centered, multiline:
 * @f[ 
 * \min_x \sum_{i=1}^n f(x_i)
 * @f] 
 * 
 */

using namespace hiop;
class PriDecMasterProblemEx9Sparse : public hiop::hiopInterfacePriDecProblem
{
public:
  PriDecMasterProblemEx9Sparse(size_t nx, size_t ny, size_t nS, size_t S);
  
  virtual ~PriDecMasterProblemEx9Sparse();

  hiop::hiopSolveStatus solve_master(hiopVector& x,
                                     const bool& include_r,
                                     const double& rval = 0, 
                                     const double* grad = 0,
                                     const double*hess = 0,
                                     const char* master_options_file = nullptr);

  virtual bool set_recourse_approx_evaluator(const int n, 
		                             hiopInterfacePriDecProblem::RecourseApproxEvaluator* evaluator);
  
  /**
   * solving the idxth recourse optimization subproblem
   * n is the number of coupled x, not the entire dimension of x
   * rval is the return value of the recourse solution function evaluation
   */
  bool eval_f_rterm(size_t idx, const int& n, const double* x, double& rval);
  
  /**
   * compute the gradient of the recourse solution function w.r.t x
   * n is the number of coupled x, not the entire dimension of x
   * grad is the output
   */
  bool eval_grad_rterm(size_t idx, const int& n, double* x, hiopVector& grad);
  
  inline size_t get_num_rterms() const;
  
  inline size_t get_num_vars() const;

  void get_solution(double* x) const;

  double get_objective();

private:
  /// dimension of primal variable `x`
  size_t nx_;
  /// dimension of the coupled variable, nc_<=nx_
  size_t nc_;
  ///dimension of recourse problem primal variable `y` for each contingency
  size_t ny_;
  /// dimension of uncertainty dimension entering the recourse problem
  size_t nS_;
  ///number of sample to use, effectively the number of recourse terms  
  size_t S_;

  double* y_;

  double* sol_;
  double obj_;
  // basecase problem
  PriDecBasecaseProblemEx9* basecase_;
};

#endif
