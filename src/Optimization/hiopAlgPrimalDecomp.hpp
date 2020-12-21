
#ifndef HIOP_PRIDECOMP
#define HIOP_PRIDECOMP

#include "hiopInterfacePrimalDecomp.hpp"

//#include <cassert>
#include <cstdio>
#include <vector>
#include <chrono>
#include <thread>
#include <cmath>

//#include <cstring> //for memcpy
namespace hiop
{


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
    int ierr = MPI_Comm_rank(comm_world, &my_rank_); assert(ierr == MPI_SUCCESS);
    int ret = MPI_Comm_size(MPI_COMM_WORLD, &comm_size_); assert(ret==MPI_SUCCESS);
    if(my_rank_==0)
    { 
      my_rank_type_ = 0;
    }else{
      my_rank_type_ = 1;
    }

    x_ = new double[n_];
    request_ = new MPI_Request[4];   
  }
  virtual ~hiopAlgPrimalDecomposition()
  {
    delete [] x_;
  }

  //we should make the public methods to look like hiopAlgFilterIPMBase
  hiopSolveStatus run();
  hiopSolveStatus run_single();

  double getObjective() const;
  
  void getSolution(double* x) const;
  
  void getDualSolutions(double* zl, double* zu, double* lambda);
  
  /* returns the status of the solver */
  inline hiopSolveStatus getSolveStatus() const;
  
  /* returns the number of iterations, meaning how many times the master was solved */
  int getNumIterations() const;


  struct HessianBBApprox{//BarzilaiBorwein
  
    HessianBBApprox() :HessianBBApprox(-1) {}
    HessianBBApprox(const int& n)
    {
      n_=n;
      xkm1 = new double[n_];
      skm1 = new double[n_];
      ykm1 = new double[n_];
      gkm1 = new double[n_];
    }

    void set_n(const int n)
    {
      n_=n;
    } 
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
    void initialize(const double* xk, const double* grad)
    {
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
    void update_hess_coeff(const double* xk, const double* gk, const double& f_val)
    {
      double temp1 = 0.;
      double temp2 = 0.;
      assert(skm1!=NULL && ykm1!=NULL);
      for(int i=0; i<n_; i++)
      {
        skm1[i] = xk[i]-xkm1[i];
        ykm1[i] = gk[i]-gkm1[i];
      }
      for(int i=0; i<n_; i++)
      {
        temp1 += skm1[i]*skm1[i];
        temp2 += skm1[i]*ykm1[i];
      }
      alpha_ = temp1/temp2;
      alpha_max = 0.;
    
      double temp3 = 0.;
      for(int i=0; i<n_; i++)
      {
        temp3 = gk[i]*gk[i];
      }
      alpha_max = 2*f_val/temp3; 
      for(int i=0; i<n_; i++){
        xkm1[i] = xk[i];
        gkm1[i] = gk[i];
      }
    //assert() what if alpha_max<alpha_min?
    }
    double get_alpha()
    {
      alpha_ = std::max(alpha_min,alpha_);
      alpha_ = std::min(alpha_max,alpha_);
      //printf("alpha max %18.12e\n",alpha_max);
      return alpha_;
    }
    double check_convergence(const double* gk)
    {
      double temp1 = 0.;
      double temp2 = 0.;
      for(int i=0;i<n_;i++)
      {
        temp1 = std::pow(ykm1[i]-1/alpha_*skm1[i],2);
        temp2 = std::pow(gk[i],2);
      }
      double convg = std::sqrt(temp1)/std::sqrt(temp2);
      return convg;
    }
  private:
    int n_;
    double alpha_=1.0;
    double alpha_min = 1e-10;
    double alpha_max = 1e20;  //= 2*f/dot(grad_f,grad_f)
    double* xkm1;
    double* gkm1;
    double* skm1;
    double* ykm1;
  };

private:
  //MPI stuff
  MPI_Comm comm_world_;
  MPI_Request* request_;
  MPI_Status status_; 
  int  my_rank_;
  int  comm_size_;

  //master/solver(0), or worker(1)
  int my_rank_type_;
  //maximum number of outer iterations
  int max_iter = 1;
  
  //pointer to the problem to be solved (passed as argument)
  hiopInterfacePriDecProblem* master_prob_;
  hiopSolveStatus solver_status_;
  
  //current primal iterate
  double* x_;

  //dimension of x_
  size_t n_;

  //number of recourse terms
  size_t S_;
};

}; //end of namespace

#endif
