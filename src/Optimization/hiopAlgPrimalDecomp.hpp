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
 * @file hiopAlgPrimalDecomp.hpp
 *
 * @author Jingyi "Frank" Wang <wang125@llnl.gov>, LLNL
 * @author Cosmin G. Petra <petra1@llnl.gov>, LLNL
 *
 */

#ifndef HIOP_PRIDECOMP
#define HIOP_PRIDECOMP

#include "hiopInterfacePrimalDecomp.hpp"
#include <cstdio>
#include <vector>
#include <chrono>
#include <thread>
#include <cmath>
#include "hiopMPI.hpp"

#include "hiopOptions.hpp"

namespace hiop
{

class hiopLogger;
  
// temporary output levels, aiming to integrate with hiop verbosity
enum MPIout {
  outlevel0=0, //print nothing during from the MPI engine
  outlevel1=1, //print standard output: objective, step size
  outlevel2=2, //print the details needed to debug the algorithm, including alpha info, 
               //also prints elapsed time and output x and gradient
  outlevel3=3, //print the send and receive messages 
  outlevel4=4  //print details about the algorithm
}; 

/* The main MPI solver for solving a class of problems with primal decomposition. 
 * The master problem is the user defined class that should be able to solve both
 * the base case and full problem depending whether a recourse approximation is 
 * included. 
 * Available options to be set in hiop_pridec.options file:
 * mem_space, alpha_max, alpha_min, tolerance, acceptable_tolerance, acceptable_iterations, 
 * max_iter, verbosity_level, print_options.
 */
class hiopAlgPrimalDecomposition
{
public:

  /**
   * Creates a primal decomposition algorithm for the primal decomposable problem
   * passed as an argument
   *
   * @param prob_in the primal decomposable problem
   * @param comm_world the communicator whose ranks should be used to schedule the tasks
   * (subproblems of the primal decomposable problem prob_in)
   */
  hiopAlgPrimalDecomposition(hiopInterfacePriDecProblem* prob_in,
                             MPI_Comm comm_world = MPI_COMM_WORLD);

  /**
   * Creates a primal decomposition algorithm for the primal decomposable problem
   * passed as an argument
   *
   * @param prob_in the primal decomposable problem
   * @param nc the number of coupling variables
   * @param xc_index array on the device  with the the indexes of the coupling variables 
   * in the full vector (primal) variables for the basecase/master problem within the primal 
   * decomposable problem prob_in.
   * @param comm_world the communicator whose ranks should be used to schedule the tasks
   * (subproblems of the primal decomposable problem prob_in)
   */
  hiopAlgPrimalDecomposition(hiopInterfacePriDecProblem* prob_in,
                             const int nc, 
                             const int* xc_index,
                             MPI_Comm comm_world = MPI_COMM_WORLD);

  virtual ~hiopAlgPrimalDecomposition();

  //we should make the public methods to look like hiopAlgFilterIPMBase
  /* Main function to run the optimization in parallel */
  hiopSolveStatus run();
  /* Main function to run the optimization in serial */
  hiopSolveStatus run_single();

  double getObjective() const;
  
  void getSolution(hiopVector& x) const;
  
  void getDualSolutions(double* zl, double* zu, double* lambda);
  
  /* returns the status of the solver */
  inline hiopSolveStatus getSolveStatus() const;
  
  /* returns the number of iterations, meaning how many times the master was solved */
  int getNumIterations() const;

  bool stopping_criteria(const int it, const double convg, const int accp_count);

  void set_verbosity(const int i);

  void set_initial_alpha_ratio(const double ratio);
    
  void set_alpha_min(const double alp_min); 
   
  void set_alpha_max(const double alp_max);
  
  void set_max_iteration(const int max_it); 
  
  void set_tolerance(const double tol);
  
  void set_acceptable_tolerance(const double tol);
  
  void set_acceptable_count(const int count);
 
  double step_size_inf(const int nc, const hiopVectorInt& idx, const hiopVector& x, const hiopVector& x0);
  
  /* Contains information of a previous solution step including function value 
   * and gradient. Used for storing the solution for the previous iteration
   * This struct is intended for internal use of hiopAlgPrimalDecomposition class only
   */
  struct Prevsol{
    Prevsol(const int n, const double f, const double* grad, const double* x)
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
   * This struct is intened for internal use of hiopAlgPrimalDecomposition class only
   */
  struct HessianApprox {
    HessianApprox(hiopInterfacePriDecProblem* priDecProb, 
                  hiopOptions* options_pridec,
                  MPI_Comm comm_world=MPI_COMM_WORLD);
    HessianApprox(const int& n, 
                  hiopInterfacePriDecProblem* priDecProb, 
                  hiopOptions* options_pridec,
                  MPI_Comm comm_world=MPI_COMM_WORLD);
    
    /* ratio is used to compute alpha in alpha_f */
    HessianApprox(const int& n,
                  const double ratio,
                  hiopInterfacePriDecProblem* priDecProb,
                  hiopOptions* options_pridec,
                  MPI_Comm comm_world=MPI_COMM_WORLD);

    ~HessianApprox();

    /* n_ is the dimension of x, hence the dimension of g_k, skm1, etc */
    void set_n(const int n);

    void set_xkm1(const hiopVector& xk);
    
    void set_gkm1(const hiopVector& grad);

    void initialize(const double f_val, const hiopVector& xk, const hiopVector& grad);
    
    /* updating variables for the current iteration */
    void update_hess_coeff(const hiopVector& xk, const hiopVector& gk, const double& f_val);
 
    /* updating ratio_ used to compute alpha i
     * Using trust-region notations,
     * rhok = (f_{k-1}-f_k)/(m(0)-m(p_k)), where m(p)=f_{k-1}+g_{k-1}^Tp+0.5 alpha_{k-1} pTp.
     * Therefore, m(0) = f_{k-1}. rhok is the ratio of real change in recourse function value
     * and the estimate change. Trust-region algorithms use a set heuristics to update alpha_k
     * based on rhok
     * rk: m(p_k)
     * The condition |x-x_{k-1}| = \Deltak is replaced by measuring the ratio of quadratic
     * objective and linear objective. 
     * User can provide a global maximum and minimum for alpha
     */
    void update_ratio();

    /** A trust-region way of updating alpha ratio
     *  rkm1: true recourse value at {k-1}
     *  rk: true recourse value at k
     */
    void update_ratio_tr(const double rhok, const double rkm1, const double rk, 
                         const double alpha_g_ratio, double& alpha_ratio);

    // updating ratio_ using both base case and recourse objective function
    void update_ratio(const double base_v, const double base_vm1);

    void update_ratio_tr(const double rhok, double& alpha_ratio);


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
    double get_alpha_f(const hiopVector& gk);

    /* Computing alpha through alpha = alpha_*ratio_
     * Not based on function information
     */ 
    double get_alpha_tr();
    
    /* Function to check convergence based gradient 
     */
    double check_convergence_grad(const hiopVector& gk);
    double check_convergence_fcn(const double base_v, const double base_vm1);
    double compute_base(const double val);

    // setting the output level for the Hessian approximation 
    void set_verbosity(const int i); 

    void set_alpha_ratio_min(const double alp_ratio_min); 
    
    void set_alpha_ratio_max(const double alp_ratio_max);

    void set_alpha_min(const double alp_min); 
   
    void set_alpha_max(const double alp_max);

  private:
    int n_;
    double alpha_ = 1e6;  
    double ratio_ = 1.0;
    double tr_ratio_ = 1.0;
    double ratio_min = 0.5;  
    double ratio_max = 5.0;  
    double alpha_min = 1e-5;  
    double alpha_max = 1e6;  

    double fk; 
    double fkm1;
    double fkm1_lin;
    hiopVector* xkm1;
    hiopVector* gkm1;
    hiopVector* skm1;
    hiopVector* ykm1;
    size_t ver_ = 2; //output level for HessianApprox class
    hiopInterfacePriDecProblem* priDecProb_;
    hiopOptions* options_;  // options is dependent on pridec options, no user set up yet
    hiopLogger* log_;
    MPI_Comm comm_world_;
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
  int max_iter_ = 200;
  int it_ = -1;

  //pointer to the problem to be solved (passed as argument)
  hiopInterfacePriDecProblem* master_prob_;
  hiopSolveStatus solver_status_;
  
  //current primal iterate
  hiopVector* x_;

  //dimension of x_
  size_t n_;

  //dimension of coupled x_
  size_t nc_;

  //number of recourse terms
  size_t S_;

  //level of output through the MPI engine
  size_t ver_ = 1;

  /// Indices of the coupled x in the full x of the basecase/master problem
  hiopVectorInt* xc_idx_;
  
  //tolerance of the convergence stopping criteria. User options from options file via hiop_pridec.options
  double tol_ = 1e-8;

  //acceptable tolerance is used to terminate hiop if NLP residuals are below the 
  //default value for 10 consecutive iterations
  double accp_tol_ = 1e-6;
  //consecutive iteration count where NLP residual is lower than acceptable tolerance
  int accp_count_ = 10;
  //initial alpha_ratio if used
  double alpha_ratio_ = 1.0;
  
  double alpha_min_ = 1e-5;  
  double alpha_max_ = 1e6;  
  
  //real decrease over expected decrease ratio
  double rhok_ = 0.;

protected:
  hiopOptions* options_;
  hiopLogger* log_;
  
};

} //end of namespace

#endif
