#ifndef HIOP_EXAMPLE_EX9
#define HIOP_EXAMPLE_EX9

//base interface (NLP specification for primal decomposable problems)
#include "hiopInterfacePrimalDecomp.hpp"

//basecase sparse NLP
#include "nlpPriDec_ex9_user_basecase.hpp"

//recourse MDS NLP
#include "nlpPriDec_ex9_user_recourse.hpp"

/**
 *
 * An example of the use of hiop::hiopInterfacePriDecProblem interface of HiOp.
 * This interface is used to specify and solve <i>primal decomposable problems</i>.
 * For a detailed mathematical form of such problems, please see
 *  hiop::hiopInterfacePriDecProblem.
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
 * Here  each of @f$ \xi^i_1, \xi^i_2, ..., \xi^i_{n_S} @f$  are withdrawn from U[-0.25, 0.25]
 * and the size n_S of the sample satisfying 1<=n_S<=n_y.
 *
 * When $S$ samples (\xi^i_1, \xi^i_2, ..., \xi^i_{n_S}), i=1,\ldots,S, are used the
 * primal decomposable problem looks like 
 *
 *     min_x basecase(x) +  \sum_{i=1}^S r_i(x;\xi_i)
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

class PriDecMasterProblemEx9 : public hiop::hiopInterfacePriDecProblem
{
public:
  PriDecMasterProblemEx9(size_t nx, size_t ny, size_t nS, size_t S);
  virtual ~PriDecMasterProblemEx9();

  hiop::hiopSolveStatus solve_master(double* x,
                                     const bool& include_r,
                                     const double& rval=0, 
                                     const double* grad=0,
                                     const double*hess =0);

  bool set_recourse_approx_evaluator(const int n, RecourseApproxEvaluator* evaluator)
  {
    return false;
  }
  
  bool eval_f_rterm(size_t idx, const int& n, double* x, double& rval);
  
  bool eval_grad_rterm(size_t idx, const int& n, double* x, double* grad);
  
  inline size_t get_num_rterms() const
  {
    return S_;
  }
  
  inline size_t get_num_vars() const
  {
    return nx_;
  }

  void get_solution(double* x) const
  {
    assert(false && "to be implemented");
  }

  double get_objective()
  {
    assert(false && "to be implemented");
    return 0.;
  }
private:
  /// dimension of primal variable `x`
  size_t nx_;
  ///dimension of recourse problem primal variable `y`
  size_t ny_;
  /// dimension of uncertainty dimension entering the recourse problem
  size_t nS_;
  ///number of sample to use, effectively the number of recourse terms  
  size_t S_;

  /// basecase
  PriDecBasecaseProblemEx9* basecase_;
};

#endif
