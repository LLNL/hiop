#ifndef HIOP_PRIDECOMP
#define HIOP_PRIDECOMP

#include "hiopInterfacePrimalDecomp.hpp"
//#include <cassert>
#include <cstdio>
#include <vector>
#include <chrono>
#include <thread>
#include <cmath>

#ifdef HIOP_USE_MPI
#include "mpi.h"
#else
#define MPI_COMM_WORLD 0
#define MPI_Comm int
#endif

namespace hiop
{

enum MPIout {
  outlevel0=0, //print nothing during from the MPI engine
  outlevel1=1, //print objective and start and end of a iteration
  outlevel2=2, //print the output x and gradient
  outlevel3=3, //print the send and receive messages 
  outlevel4=4  //print details about the algorithm
}; 

/* The main mpi engine for solving a class of problems with primal decomposition. 
 * The master problem is the user defined class that should be able to solve both
 * the base case and full problem depending whether a recourse approximation is 
 * included. 
 *
 */
class hiopAlgPrimalDecomposition
{
public:

  //constructor
  hiopAlgPrimalDecomposition(hiopInterfacePriDecProblem* prob_in,
                             MPI_Comm comm_world=MPI_COMM_WORLD);

  hiopAlgPrimalDecomposition(hiopInterfacePriDecProblem* prob_in,
		             const int nc, 
			     const std::vector<int>& xc_index,
                             MPI_Comm comm_world=MPI_COMM_WORLD);


  virtual ~hiopAlgPrimalDecomposition();

  //we should make the public methods to look like hiopAlgFilterIPMBase
  /* Main function to run the optimization in parallel */
  hiopSolveStatus run();
  /* Main function to run the optimization in serial */
  hiopSolveStatus run_single();

  double getObjective() const;
  
  void getSolution(double* x) const;
  
  void getDualSolutions(double* zl, double* zu, double* lambda);
  
  /* returns the status of the solver */
  inline hiopSolveStatus getSolveStatus() const;
  
  /* returns the number of iterations, meaning how many times the master was solved */
  int getNumIterations() const;

  bool stopping_criteria(const int it, const double convg);

  void set_verbosity(const int i);

  void set_initial_alpha_ratio(const double ratio);
    
  void set_max_iteration(const int max_it); 
  /* Contains information of a solution step including function value 
   * and gradient. Used for storing the solution for the previous iteration
   */
  struct prev_sol{
    prev_sol(const int n, const double f, const double* grad, const double* x)
    {
      n_ = n;
      f_ = f;
      grad_ = new double[n];
      memcpy(grad_, grad, n_*sizeof(double));
      x_ = new double[n];
      memcpy(x_, x, n_*sizeof(double));
    }
    void update(const double f, const double* grad, const double* x)
    {
      assert(grad!=NULL);
      memcpy(grad_, grad, n_*sizeof(double));
      memcpy(x_, x, n_*sizeof(double));
      f_ = f;
    }

    double get_f(){return f_;}
    double* get_grad(){return grad_;}
    double* get_x(){return x_;}

  private:
    int n_;
    double f_;
    double* grad_;
    double* x_;
  };

  /* Struct for the quadratic coefficient alpha in the recourse approximation
   * function. It contains quantities such as s_{k-1} = x_k-x_{k-1} that is 
   * otherwise not computed but useful for certian update rules for alpha,
   * as well as the convergence measure. The update function is called
   * every iteration to ensure the values are up to date.
   * The xk here should only be the coupled x.
   */
  struct HessianApprox{
    HessianApprox();
    HessianApprox(const int& n);
    
    /* ratio_ is used to compute alpha in alpha_f */
    HessianApprox(const int& n,const double ratio);

    ~HessianApprox();

    /* n_ is the dimension of x, hence the dimension of g_k, skm1, etc */
    void set_n(const int n);

    void set_xkm1(const double* xk);
    
    void set_gkm1(const double* grad);

    void initialize(const double f_val, const double* xk, const double* grad);
    
    /* updating variables for the current iteration */
    void update_hess_coeff(const double* xk, const double* gk, const double& f_val);
 
    /* updating ratio_ used to compute alpha i
     * Using trust-region notations,
     * rhok = (f_{k-1}-f_k)/(m(0)-m(p_k)), where m(p)=f_{k-1}+g_{k-1}^Tp+0.5 alpha_{k-1} pTp.
     * Therefore, m(0) = f_{k-1}. rhok is the ratio of real change in recourse function value
     * and the estimate change. Trust-region algorithms use a set heuristics to update alpha_k
     * based on rhok
     * rk: m(p_k)
     * The condition |x-x_{k-1}| = \Delatk is replaced by measuring the ratio of quadratic
     * objective and linear objective. 
     * User can provide a global maximum and minimum for alpha
     */
    void update_ratio();

    //a trust region way of updating alpha ratio
    //rkm1: true recourse value at {k-1}
    //rk: true recourse value at k
    void update_ratio_tr(const double rhok,const double rkm1, const double rk, const double alpha_g_ratio,
		         double& alpha_ratio);

    /* currently provides multiple ways to compute alpha, one is to the BB alpha
     * or the alpha computed through the BarzilaiBorwein gradient method, a quasi-Newton method.
     */
    double get_alpha_BB();

    /* Computing alpha through alpha = alpha_f*ratio_
     * alpha_f is computed through
     * min{f_k+g_k^T(x-x_k)+0.5 alpha_k|x-x_k|^2 >= beta_k f}
     * So alpha_f is based on the constraint on the minimum of recourse
     * approximition. This is to ensure good approximation.
     */ 
    double get_alpha_f(const double* gk);

    /* Function to check convergence based gradient 
     */
    double check_convergence_grad(const double* gk);
    double check_convergence_fcn( );


    // setting the output level for the Hessian approximation 
    void set_verbosity(const int i); 

    void set_alpha_ratio_min(const double alp_ratio_min); 
    
    void set_alpha_ratio_max(const double alp_ratio_max);

    void set_alpha_min(const double alp_min); 
    
    void set_alpha_max(const double alp_max);
  private:
    int n_;
    double alpha_=1.0;  
    double ratio_ = 1.0;
    double ratio_min = 0.5;  
    double ratio_max = 5.0;  
    double alpha_min = 1e-5;  
    double alpha_max = 1e10;  
    double fk; 
    double fkm1;
    double* xkm1;
    double* gkm1;
    double* skm1;
    double* ykm1;
    size_t ver_=1;
  };

private:
   
#ifdef HIOP_USE_MPI
  MPI_Request* request_;
  MPI_Status status_; 
  int  my_rank_,comm_size_;
  int my_rank_type_;
#endif

  MPI_Comm comm_world_;
  //master/solver(0), or worker(1:total rank)

  //maximum number of outer iterations, user specified
  int max_iter = 200;
  
  //pointer to the problem to be solved (passed as argument)
  hiopInterfacePriDecProblem* master_prob_;
  hiopSolveStatus solver_status_;
  
  //current primal iterate
  double* x_;

  //dimension of x_
  size_t n_;

  //dimension of coupled x_
  size_t nc_;

  //number of recourse terms
  size_t S_;

  //level of output through the MPI engine
  size_t ver_ = 1;

  //indices of the coupled x in the full x
  std::vector<int> xc_idx_;
  //tolerance of the convergence stopping criteria
  double tol_=1e-6;
  //initial alpha_ratio if used
  double alpha_ratio_=1.0;
};

}; //end of namespace

#endif
