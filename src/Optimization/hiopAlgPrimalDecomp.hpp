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

#ifdef HIOP_USE_MPI
  /* This struct provides the info necessary for the recourse approximation function
   * buffer[n+1] contains both the function value and gradient w.r.t x
   * buffer[0] is the function value and buffer[1:n] the gradient
   * Contains send and receive functionalities for the values in buffer
   */
  struct ReqRecourseApprox
  {
    ReqRecourseApprox() : ReqRecourseApprox(1) {}
    ReqRecourseApprox(const int& n)
    {
      n_=n;
      buffer=new double[n_];
    }

    int test() 
    {
      int mpi_test_flag; MPI_Status mpi_status;
      int ierr = MPI_Test(&request_, &mpi_test_flag, &mpi_status);
      assert(MPI_SUCCESS == ierr);
      return mpi_test_flag;
    }
    void post_recv(int tag, int rank_from, MPI_Comm comm)
    {
      int ierr = MPI_Irecv(buffer, n_+1, MPI_DOUBLE, rank_from, tag, comm, &request_);
      assert(MPI_SUCCESS == ierr);
    }
    void post_send(int tag, int rank_to, MPI_Comm comm)
    {
      int ierr = MPI_Isend(buffer, n_+1, MPI_DOUBLE, rank_to, tag, comm, &request_);
      assert(MPI_SUCCESS == ierr);
    }
    double value(){return buffer[0];}
    void set_value(const double v){buffer[0]=v;}
    double grad(int i){return buffer[i+1];}
    void set_grad(const double* g)
    {
      for(int i=0;i<n_;i++)
      {
        buffer[i+1]=g[i];
      }
    }

    MPI_Request request_;
  private:
    int n_;
    double* buffer;
  };

  /* This struct is used to post receive and request for contingency
   * index that is to be solved by the solver ranks.
   */
  struct ReqContingencyIdx
  {
    ReqContingencyIdx() : ReqContingencyIdx(-1) {}
    ReqContingencyIdx(const int& idx_)
    {
      idx=idx_;
    }

    int test() {
      int mpi_test_flag; MPI_Status mpi_status;
      int ierr = MPI_Test(&request_, &mpi_test_flag, &mpi_status);
      assert(MPI_SUCCESS == ierr);
      return mpi_test_flag;
    }
    void post_recv(int tag, int rank_from, MPI_Comm comm)
    {
      int ierr = MPI_Irecv(&idx, 1, MPI_INT, rank_from, tag, comm, &request_);
      assert(MPI_SUCCESS == ierr);
    }
    void post_send(int tag, int rank_to, MPI_Comm comm)
    {
      int ierr = MPI_Isend(&idx, 1, MPI_INT, rank_to, tag, comm, &request_);
      assert(MPI_SUCCESS == ierr);
    }
    double value(){return idx;}
    void set_idx(const int& i){idx = i;}
    MPI_Request request_;
  private:
    int idx;
  };
#endif

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
                             MPI_Comm comm_world=MPI_COMM_WORLD)
    : master_prob_(prob_in), comm_world_(comm_world)
  {
    S_ = master_prob_->get_num_rterms();
    n_ = master_prob_->get_num_vars();

    //determine rank and rank type
    //only two rank types for now, master and evaluator/worker

    #ifdef HIOP_USE_MPI
      int ierr = MPI_Comm_rank(comm_world, &my_rank_); assert(ierr == MPI_SUCCESS);
      int ret = MPI_Comm_size(MPI_COMM_WORLD, &comm_size_); assert(ret==MPI_SUCCESS);
      if(my_rank_==0)
      { 
        my_rank_type_ = 0;
      }else{
        my_rank_type_ = 1;
      }
      request_ = new MPI_Request[4];   
    #endif

    x_ = new double[n_];
  }
  virtual ~hiopAlgPrimalDecomposition()
  {
    delete [] x_;
  }

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

  /* Contains information of a solution step including function value 
   * and gradient. Used for storing the solution for the previous iteration
   * */
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
   */
  struct HessianApprox{
    HessianApprox() :HessianApprox(-1) {}
    HessianApprox(const int& n)
    {
      n_=n;
      fkm1 = 1e20;
      fk = 1e20; 
      xkm1 = new double[n_]; // x at k-1 step, the current step is k
      skm1 = new double[n_]; // s_{k-1} = x_k - x_{k-1}
      ykm1 = new double[n_]; // y_{k-1} = g_k - g_{k-1}
      gkm1 = new double[n_]; // g_{k-1}
    }
    /* ratio_ is used to compute alpha in alpha_f */
    HessianApprox(const int& n,const double ratio):HessianApprox(n)
    {
      ratio_=ratio;
    }
    ~HessianApprox()
    {
      delete[] xkm1;
      delete[] skm1;
      delete[] ykm1;
      delete[] gkm1;    
    }
    /* n_ is the dimension of x, hence the dimension of g_k, skm1, etc */
    void set_n(const int n){n_=n;}

    void set_xkm1(const double* xk)
    {
      if(xkm1==NULL)
      {
        assert(n_!=-1);
        xkm1 = new double[n_];
      }else{
        memcpy(xkm1, xk, n_*sizeof(double));
      }
    }
    void set_gkm1(const double* grad)
    {
      if(gkm1==NULL)
      {
        assert(n_!=-1);
        gkm1 = new double[n_];
      }else{
        memcpy(gkm1, grad, n_*sizeof(double));
      }
    }

    void initialize(const double f_val, const double* xk, const double* grad)
    {
      fk = f_val;
      if(xkm1==NULL)
      {
        assert(n_!=-1);
        xkm1 = new double[n_];
      }else{
        memcpy(xkm1, xk, n_*sizeof(double));
      }
      if(gkm1==NULL)
      {
        assert(n_!=-1);
        gkm1 = new double[n_];
      }else{
        memcpy(gkm1, grad, n_*sizeof(double));
      }
      if(skm1==NULL)
      {
        assert(n_!=-1);
        skm1 = new double[n_];
      }
      if(ykm1==NULL)
      {
        assert(n_!=-1);
        ykm1 = new double[n_];
      }
    }
    
    /* updating variables for the current iteration */
    void update_hess_coeff(const double* xk, const double* gk, const double& f_val)
    {
      fkm1 = fk;
      fk = f_val;
      assert(skm1!=NULL && ykm1!=NULL);
      for(int i=0; i<n_; i++)
      {
        skm1[i] = xk[i]-xkm1[i];
        ykm1[i] = gk[i]-gkm1[i];
      }
      //alpha_min = std::max(temp3/2/f_val,2.5); 
      update_ratio();
      for(int i=0; i<n_; i++){
        xkm1[i] = xk[i];
        gkm1[i] = gk[i];
      }
    }
 
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
    void update_ratio()
    {
      double rk = fkm1;
      for(int i=0;i<n_;i++)
      {
        rk+= gkm1[i]*skm1[i]+0.5*alpha_*(skm1[i]*skm1[i]);
      }
      //printf("recourse estimate inside HessianApprox %18.12e\n",rk);
      double rho_k = (fkm1-fk)/(fkm1-rk);
      printf("previuos val  %18.12e, real val %18.12e, predicted val %18.12e, rho_k %18.12e\n",fkm1,fk,rk,rho_k);
      /* 
      double beta = 1.-1./ratio_;
      double diff = fabs(beta*fkm1-rkm1)/fkm1;
      printf("reaching beta limit  %18.12e",diff);
      */
      //a measure for when alpha should be decreasing (in addition to being good approximation)
      double quanorm = 0.; double gradnorm=0.;
      for(int i=0;i<n_;i++)
      {
	gradnorm += gkm1[i]*skm1[i];
        quanorm += skm1[i]*skm1[i];
      }
      quanorm = alpha_*quanorm;
      double alpha_g_ratio = quanorm/fabs(gradnorm);
      printf("alpha norm ratio  %18.12e",alpha_g_ratio);
      //using a trust region criteria for adjusting ratio
      update_ratio_tr(rho_k,fkm1, fk, alpha_g_ratio, ratio_);
    } 

    //a trust region way of updating alpha ratio
    //rkm1: true recourse value at {k-1}
    //rk: true recourse value at k
    void update_ratio_tr(const double rhok,const double rkm1, const double rk, const double alpha_g_ratio,
		         double& alpha_ratio)
    {
      if(rhok>0 && rhok < 1/4. && (rkm1-rk>0))
      {
	alpha_ratio = alpha_ratio/0.75;
        printf("increasing alpha ratio or increasing minimum for quadratic coefficient\n");
      }
      else if(rhok<0 && (rkm1-rk)<0){
        alpha_ratio = alpha_ratio/0.75;
	printf("increasing alpha ratio or increasing minimum for quadratic coefficient\n");
      }
      else{
	if (rhok > 0.75 && rhok<1.333 &&(rkm1-rk>0) && alpha_g_ratio>0.1){ 
	  alpha_ratio *= 0.75;
	  printf("decreasing alpha ratio or decreasing minimum for quadratic coefficient\n");
	}
        else if(rhok>1.333 && (rkm1-rk<0) ){
	  alpha_ratio = alpha_ratio/0.75;
	  printf("recourse increasing and increased more in real contingency, so increasing alpha\n");
	}
      }
      if ((rhok>0 &&rhok<1/8. && (rkm1-rk>0) ) || (rhok<0 && rkm1-rk<0 ) )
      {
        printf("This step is rejected.\n");
	//sol_base = solm1;
	//f = fm1;
        //gradf = gkm1;
      } 
    }

    /* currently provides multiple ways to compute alpha, one is to the BB alpha
     * or the alpha computed through the BarzilaiBorwein gradient method, a quasi-Newton method.
     */
    double get_alpha_BB()
    {
      double temp1 = 0.;
      double temp2 = 0.;
      for(int i=0; i<n_; i++)
      {
        temp1 += skm1[i]*skm1[i];
        temp2 += skm1[i]*ykm1[i];
      }
      alpha_ = temp2/temp1;
      alpha_ = std::max(alpha_min,alpha_);
      alpha_ = std::min(alpha_max,alpha_);
      //printf("alpha max %18.12e\n",alpha_max);
      return alpha_;
    }

    /* Computing alpha through alpha = alpha_f*ratio_
     * alpha_f is computed through
     * min{f_k+g_k^T(x-x_k)+0.5 alpha_k|x-x_k|^2 >= beta_k f}
     * So alpha_f is based on the constraint on the minimum of recourse
     * approximition. This is to ensure good approximation.
     */ 
    double get_alpha_f(const double* gk)
    {
      double temp3 = 0.;
      //call update first, gkm1 is already gk
      for(int i=0; i<n_; i++){temp3 += gk[i]*gk[i];}
      alpha_ = temp3/2.0/fk; 
      //printf("alpha check %18.12e\n",temp3/2.0);
      alpha_ *= ratio_;
      alpha_ = std::max(alpha_min,alpha_);
      alpha_ = std::min(alpha_max,alpha_);
      printf("alpha ratio %18.12e\n",ratio_);
      return alpha_;
    }

    /* Function to check convergence based gradient 
     */
    double check_convergence(const double* gk)
    {
      double temp1 = 0.;
      double temp2 = 0.;
      for(int i=0;i<n_;i++)
      {
        temp1 += std::pow(ykm1[i]-alpha_*skm1[i],2);
        temp2 += std::pow(gk[i],2);
      }
      double convg = std::sqrt(temp1)/std::sqrt(temp2);
      return convg;
    }
    private:
      int n_;
      double alpha_=1.0;
      double ratio_ = 1.0;
      double alpha_min = 1e-5;  double alpha_max = 1e10;  
      double fk; double fkm1;
      double* xkm1;
      double* gkm1;
      double* skm1;
      double* ykm1;
  };

private:
  //MPI 
#ifdef HIOP_USE_MPI
  MPI_Request* request_;
  MPI_Status status_; 
  int  my_rank_,comm_size_;
  int my_rank_type_;
#endif

  MPI_Comm comm_world_;
  //master/solver(0), or worker(1:total rank)

  //maximum number of outer iterations, user specified
  int max_iter = 100;
  
  //pointer to the problem to be solved (passed as argument)
  hiopInterfacePriDecProblem* master_prob_;
  hiopSolveStatus solver_status_;
  
  //current primal iterate
  double* x_;

  //dimension of x_
  size_t n_;

  //number of recourse terms
  size_t S_;

  //tolerance of the convergence stopping criteria
  double tol_=1e-6;

};

}; //end of namespace

#endif
