//include header file
#include "hiopAlgPrimalDecomp.hpp"
#include "hiopInterfacePrimalDecomp.hpp"

#include <cassert>
#include <cstring>

using namespace std;

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
      buffer = LinearAlgebraFactory::createVector(n_+1);
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
      double* buffer_arr = buffer->local_data();
      int ierr = MPI_Irecv(buffer_arr, n_+1, MPI_DOUBLE, rank_from, tag, comm, &request_);
      assert(MPI_SUCCESS == ierr);
    }
    void post_send(int tag, int rank_to, MPI_Comm comm)
    {
      double* buffer_arr = buffer->local_data();
      int ierr = MPI_Isend(buffer_arr, n_+1, MPI_DOUBLE, rank_to, tag, comm, &request_);
      assert(MPI_SUCCESS == ierr);
    }
    double value(){return buffer->local_data()[0];}
    void set_value(const double v){buffer->local_data()[0]=v;}
    double grad(int i){return buffer->local_data()[i+1];}
    void set_grad(const double* g)
    {
      buffer->copyFromStarting(1,g,n_);
    }

    MPI_Request request_;
  private:
    int n_;
    hiopVector* buffer;
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



hiopAlgPrimalDecomposition::HessianApprox::
HessianApprox() : 
  HessianApprox(-1) 
{
}

hiopAlgPrimalDecomposition::HessianApprox::
HessianApprox(const int& n)
{
  n_=n;
  fkm1 = 1e20;
  fk = 1e20; 
  xkm1 = LinearAlgebraFactory::createVector(n_);// x at k-1 step, the current step is k
  skm1 = LinearAlgebraFactory::createVector(n_);// s_{k-1} = x_k - x_{k-1}
  ykm1 = LinearAlgebraFactory::createVector(n_);// y_{k-1} = g_k - g_{k-1}
  gkm1 = LinearAlgebraFactory::createVector(n_);// g_{k-1}
}

hiopAlgPrimalDecomposition::HessianApprox::
HessianApprox(const int& n,const double ratio):HessianApprox(n)
{
  ratio_=ratio;
}

hiopAlgPrimalDecomposition::HessianApprox::
~HessianApprox()
{
  //delete[] xkm1;
  delete xkm1;
  delete skm1;
  delete ykm1;
  delete gkm1;    
}

/* n_ is the dimension of x, hence the dimension of g_k, skm1, etc */
void hiopAlgPrimalDecomposition::HessianApprox::
set_n(const int n)
{
  n_=n;
}


void hiopAlgPrimalDecomposition::HessianApprox::
set_xkm1(const hiopVector& xk)
{
  if(xkm1==NULL) {
    assert(n_!=-1);
    xkm1 = LinearAlgebraFactory::createVector(n_);
  } else {
    xkm1->copyFromStarting(0, xk.local_data_const(), n_);
  }
}


void hiopAlgPrimalDecomposition::HessianApprox::
set_gkm1(const hiopVector& grad)
{
  if(gkm1==NULL) {
    assert(n_!=-1);
    gkm1 = LinearAlgebraFactory::createVector(n_);
  } else {
    gkm1->copyFromStarting(0, grad.local_data_const(), n_);
  }
}

void hiopAlgPrimalDecomposition::HessianApprox::
initialize(const double f_val, const hiopVector& xk, const hiopVector& grad)
{
  fk = f_val;
  if(xkm1==NULL) {
    assert(n_!=-1);
    xkm1 = LinearAlgebraFactory::createVector(n_);
    //xkm1 = new double[n_];
  } else {
    xkm1->copyFromStarting(0, xk.local_data_const(), n_);
    //memcpy(xkm1, xk, n_*sizeof(double));
  }
  if(gkm1==NULL) {
    assert(n_!=-1);
    gkm1 = LinearAlgebraFactory::createVector(n_);
  } else {
    gkm1->copyFromStarting(0, grad.local_data_const(), n_);
  }
  if(skm1==NULL) {
    assert(n_!=-1);
    skm1->copyFromStarting(0, xk.local_data_const(), n_);
  }
  if(ykm1==NULL) {
    assert(n_!=-1);
    ykm1->copyFromStarting(0, xk.local_data_const(), n_);
  }
}
    
/* updating variables for the current iteration */
void hiopAlgPrimalDecomposition::HessianApprox::
update_hess_coeff(const hiopVector& xk, 
                  const hiopVector& gk, 
                  const double& f_val)
{
  fkm1 = fk;
  fk = f_val;
  assert(skm1!=NULL && ykm1!=NULL);

  assert(xk.get_local_size()==skm1->get_local_size());
  skm1->copyFrom(xk);
  skm1->axpy(-1.0, *xkm1);

  ykm1->copyFrom(gk);
  ykm1->axpy(-1.0, *gkm1);
  
  //alpha_min = std::max(temp3/2/f_val,2.5); 
  update_ratio();
  assert(xkm1->get_local_size()==xk.get_local_size());
  xkm1->copyFrom(xk);
  gkm1->copyFrom(gk);
}
 
/** 
 * updating ratio_ used to compute alpha i
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
 
void hiopAlgPrimalDecomposition::HessianApprox::
update_ratio()
{
  double rk = fkm1;

  hiopVector* temp;
  temp = LinearAlgebraFactory::createVector(gkm1->get_local_size()); 
  temp->copyFrom(*gkm1);   
  rk += temp->dotProductWith(*skm1);
  rk += 0.5*alpha_*(skm1->twonorm())*(skm1->twonorm()); 

  //printf("recourse estimate inside HessianApprox %18.12e\n",rk);
  double rho_k = (fkm1-fk)/(fkm1-rk);
  if(ver_ >=outlevel2) {
    printf("previuos val  %18.12e, real val %18.12e, predicted val %18.12e, rho_k %18.12e\n",fkm1,fk,rk,rho_k);
  }
  //a measure for when alpha should be decreasing (in addition to being good approximation)
  double quanorm = 0.; double gradnorm=0.;
  quanorm += skm1->dotProductWith(*skm1);
  gradnorm += gkm1->dotProductWith(*skm1);

  quanorm = alpha_*quanorm;

  double alpha_g_ratio = quanorm/fabs(gradnorm);
  if(ver_ >=outlevel2) {

    printf("alpha norm ratio  %18.12e",alpha_g_ratio);
  }
  //using a trust region criteria for adjusting ratio
  update_ratio_tr(rho_k,fkm1, fk, alpha_g_ratio, ratio_);
} 

/* a trust region way of updating alpha ratio
 * rkm1: true recourse value at {k-1}
 * rk: true recourse value at k
 */
void hiopAlgPrimalDecomposition::HessianApprox::
update_ratio_tr(const double rhok,
                const double rkm1, 
                const double rk, 
                const double alpha_g_ratio,
                double& alpha_ratio)
{
  if(rhok>0 && rhok < 1/4. && (rkm1-rk>0)) {
    alpha_ratio = alpha_ratio/0.75;
    if(ver_ >=outlevel1) {
      printf("increasing alpha ratio or increasing minimum for quadratic coefficient\n");
    }
  } else if(rhok<0 && (rkm1-rk)<0) {
    alpha_ratio = alpha_ratio/0.75;
    if(ver_ >=outlevel1) {
      printf("increasing alpha ratio or increasing minimum for quadratic coefficient\n");
    }
  } else {
    if(rhok > 0.75 && rhok<1.333 &&(rkm1-rk>0) && alpha_g_ratio>0.1) { 
      alpha_ratio *= 0.75;
      if(ver_ >=outlevel2) {
        printf("decreasing alpha ratio or decreasing minimum for quadratic coefficient\n");
      }
    } else if(rhok>1.333 && (rkm1-rk<0)) {
      alpha_ratio = alpha_ratio/0.75;
      if(ver_ >=outlevel2) {
        printf("recourse increasing and increased more in real contingency, so increasing alpha\n");
      }
    }
  }
  if((rhok>0 &&rhok<1/8. && (rkm1-rk>0) ) || (rhok<0 && rkm1-rk<0 ) ) {
    if(ver_ >=outlevel3) {
      printf("This step is rejected.\n");
    }
    //sol_base = solm1;
    //f = fm1;
    //gradf = gkm1;
  }
  alpha_ratio = std::max(ratio_min,alpha_ratio); 
  alpha_ratio = std::min(ratio_max,alpha_ratio); 
}

/* currently provides multiple ways to compute alpha, one is to the BB alpha
 * or the alpha computed through the BarzilaiBorwein gradient method, a quasi-Newton method.
 */
double hiopAlgPrimalDecomposition::HessianApprox::
get_alpha_BB()
{
  double temp1 = 0.;
  double temp2 = 0.;
  
  temp1 = skm1->dotProductWith(*skm1);
  temp2 = skm1->dotProductWith(*ykm1);
  
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

double hiopAlgPrimalDecomposition::HessianApprox::get_alpha_f(const hiopVector& gk)
{
  double temp3 = 0.;

  //call update first, gkm1 is already gk
  temp3 = gk.twonorm()*gk.twonorm();

  alpha_ = temp3/2.0/fk; 
  //printf("alpha check %18.12e\n",temp3/2.0);
  alpha_ *= ratio_;
  alpha_ = std::max(alpha_min,alpha_);
  alpha_ = std::min(alpha_max,alpha_);
  if(ver_ >=outlevel3) {
    printf("alpha ratio %18.12e\n",ratio_);
  }
  return alpha_;
}

// stopping criteria based on gradient
double hiopAlgPrimalDecomposition::HessianApprox::check_convergence_grad(const hiopVector& gk)
{
  double temp1 = 0.;
  double temp2 = 0.;
  double temp3 = 0.;
  double temp4 = 0.;
  
  hiopVector* temp;
  temp = LinearAlgebraFactory::createVector(skm1->get_local_size()); 
  temp->copyFrom(*skm1);  
  temp->scale(-alpha_);
  temp4 = temp->twonorm()*temp->twonorm();
  
  temp3 = ykm1->twonorm()*ykm1->twonorm();
  temp->axpy(1.0,*ykm1);
  temp1 = temp->twonorm()*temp->twonorm();

  temp2 = gk.twonorm()*gk.twonorm();

  double convg = std::sqrt(temp1)/std::sqrt(temp2);
  if(ver_ >=outlevel2) {
    printf("temp1  %18.12e, temp2 %18.12e, temp3 %18.12e, temp4 %18.12e\n",temp1,temp2,temp3,temp4);
  }
  return convg;
}
// stopping criteria based on function value change
double hiopAlgPrimalDecomposition::HessianApprox::check_convergence_fcn()
{
  double predicted_decrease = 0.;

  assert(n_==gkm1->get_local_size());
  hiopVector* temp;
  temp = LinearAlgebraFactory::createVector(gkm1->get_local_size()); 
  temp->copyFrom(*gkm1);   
  predicted_decrease += temp->dotProductWith(*skm1);
  predicted_decrease += 0.5*alpha_*(skm1->twonorm())*(skm1->twonorm()); 

  if(ver_ >=outlevel2) {
    printf("predicted decrease  %18.12e",predicted_decrease);
  }
  predicted_decrease = fabs(predicted_decrease);
  return predicted_decrease;
}

// Compute the base case value at the kth step by subtracting
// recourse approximation value from the objective. rval is the real
// recourse function value at x_{k-1}, val is the master problem 
// objective which is the sum of the base case value and the recourse function value.
// This requires the previous steps to compute, hence in the HessianApprox class.
double hiopAlgPrimalDecomposition::HessianApprox::
compute_base(const double val, const double rval)
{
  double rec_appx = rval;
  
  hiopVector* temp;
  temp = LinearAlgebraFactory::createVector(gkm1->get_local_size()); 
  temp->copyFrom(*gkm1);   
  rec_appx += temp->dotProductWith(*skm1);
  rec_appx += 0.5*alpha_*(skm1->twonorm())*(skm1->twonorm()); 

  return val-rec_appx;
}

void hiopAlgPrimalDecomposition::HessianApprox::set_verbosity(const int i)
{
  assert(i<=3 && i>=0);
  ver_ = i;
}

void hiopAlgPrimalDecomposition::HessianApprox::
set_alpha_ratio_min(const double alp_ratio_min)
{
  ratio_min = alp_ratio_min;
}

void hiopAlgPrimalDecomposition::HessianApprox::
set_alpha_ratio_max(const double alp_ratio_max)
{
  ratio_max = alp_ratio_max;
}

void hiopAlgPrimalDecomposition::HessianApprox::
set_alpha_min(const double alp_min)
{
  alpha_min = alp_min;
}

void hiopAlgPrimalDecomposition::HessianApprox::
set_alpha_max(const double alp_max)
{
  alpha_max = alp_max;
}
hiopAlgPrimalDecomposition::
hiopAlgPrimalDecomposition(hiopInterfacePriDecProblem* prob_in,
                           MPI_Comm comm_world/*=MPI_COMM_WORLD*/) 
  : master_prob_(prob_in), 
    comm_world_(comm_world)
{
  S_ = master_prob_->get_num_rterms();
  n_ = master_prob_->get_num_vars();
  // if no coupling indices are specified, assume the entire x is coupled
  nc_ = n_;
  xc_idx_ = new int[nc_];
  
  for(int i=0; i<nc_; i++) {
    xc_idx_[i] = i;
  }
  //determine rank and rank type
  //only two rank types for now, master and evaluator/worker

  #ifdef HIOP_USE_MPI
    int ierr = MPI_Comm_rank(comm_world, &my_rank_); assert(ierr == MPI_SUCCESS);
    int ret = MPI_Comm_size(MPI_COMM_WORLD, &comm_size_); assert(ret==MPI_SUCCESS);
    if(my_rank_==0) { 
      my_rank_type_ = 0;
    } else {
      my_rank_type_ = 1;
    }
    request_ = new MPI_Request[4];   
  #endif
  //x_ = new double[n_];
  x_ = LinearAlgebraFactory::createVector(n_);
  
  //use "hiop_pridec.options" - if the file does not exist, built-in default options will be used
  options_ = new hiopOptionsPriDec(hiopOptions::default_filename_pridec_solver);
}

hiopAlgPrimalDecomposition::
hiopAlgPrimalDecomposition(hiopInterfacePriDecProblem* prob_in,
                           const int nc, 
                           const int* xc_index,
                           MPI_Comm comm_world/*=MPI_COMM_WORLD*/)
  : master_prob_(prob_in),
    nc_(nc), 
    comm_world_(comm_world)
{
  S_ = master_prob_->get_num_rterms();
  n_ = master_prob_->get_num_vars();

  xc_idx_ = new int[nc_];
  for(int i=0; i<nc; i++) {
    xc_idx_[i] = xc_index[i];
  }
  //determine rank and rank type
  //only two rank types for now, master and evaluator/worker

#ifdef HIOP_USE_MPI
  int ierr = MPI_Comm_rank(comm_world, &my_rank_); assert(ierr == MPI_SUCCESS);
  int ret = MPI_Comm_size(MPI_COMM_WORLD, &comm_size_); assert(ret==MPI_SUCCESS);
  if(my_rank_==0) { 
    my_rank_type_ = 0;
  } else {
    my_rank_type_ = 1;
  }
  request_ = new MPI_Request[4];   
#endif
  x_ = LinearAlgebraFactory::createVector(n_);

  //use "hiop_pridec.options" - if the file does not exist, built-in default options will be used
  options_ = new hiopOptionsPriDec(hiopOptions::default_filename_pridec_solver);
}

hiopAlgPrimalDecomposition::~hiopAlgPrimalDecomposition()
{
  delete x_;
  delete options_;
}


double hiopAlgPrimalDecomposition::getObjective() const
{
  return master_prob_->get_objective();
}

void hiopAlgPrimalDecomposition::getSolution(hiopVector& x) const
{
  double* x_vec = x.local_data();
  master_prob_->get_solution(x_vec);
}
  
void hiopAlgPrimalDecomposition::getDualSolutions(double* zl, double* zu, double* lambda)
{
  assert(false && "not implemented");
}

inline hiopSolveStatus hiopAlgPrimalDecomposition::getSolveStatus() const
{
  return solver_status_;
}

int hiopAlgPrimalDecomposition::getNumIterations() const
{
  assert(false && "not yet implemented");
  return 9;
}
  
bool hiopAlgPrimalDecomposition::stopping_criteria(const int it, const double convg)
{
  //gradient based stopping criteria
  if(convg<tol_){printf("reaching error tolerance, successfully found solution\n"); return true;}
  //stopping criteria based on the change in objective function
  if(it == max_iter-1) {
    printf("reached maximum iterations, optimization stops.\n");
    return true;
  }
  if(accp_count == 10) {
    printf("reached acceptable tolerance of %18.12e for 10 iterations, optimization stops.\n",
           accp_tol_);
    return true;
  }
  return false;
}
  
double hiopAlgPrimalDecomposition::
step_size_inf(const int nc, const hiopVector& x, const hiopVector& x0)
{
  double step = -1e20;
  hiopVector* temp;
  temp = LinearAlgebraFactory::createVector(x.get_local_size()); 
  temp->copyFrom(x);   
  temp->axpy(-1.0,x0); 
  step = temp->infnorm();

  return step;
}


void hiopAlgPrimalDecomposition::set_max_iteration(const int max_it)  
{
  max_iter = max_it;
}

void hiopAlgPrimalDecomposition::set_verbosity(const int i)
{
  assert(i<=3 && i>=0);
  ver_ = i;
}

void hiopAlgPrimalDecomposition::set_tolerance(const double tol)
{
  tol_ = tol;
}

void hiopAlgPrimalDecomposition::set_acceptable_tolerance(const double tol)
{
  accp_tol_ = tol;
}


void hiopAlgPrimalDecomposition::set_initial_alpha_ratio(const double alpha)
{
  assert(alpha>=0&&alpha<10.);
  alpha_ratio_ = alpha;
}

/* MPI engine for parallel solver
 */

#ifdef HIOP_USE_MPI
  hiopSolveStatus hiopAlgPrimalDecomposition::run()
  {

    if(comm_size_==1) {
      return run_single();//call the serial solver
    }
    if(my_rank_==0) {
      printf("total number of recourse problems  %lu\n", S_);
      printf("total ranks %d\n",comm_size_);
    }
    //initial point for now set to all zero
    x_->setToConstant(0.0);
      
    bool bret;
    int rank_master=0; //master rank is also the solver rank
    //Define the values and gradients as needed in the master rank
    double rval = 0.;
    //double grad_r[nc_];

    hiopVector* grad_r;
    grad_r = LinearAlgebraFactory::createVector(nc_); 
    grad_r->setToZero(); 
    double* grad_r_vec=grad_r->local_data();
  
    hiopVector* hess_appx;
    hess_appx = LinearAlgebraFactory::createVector(nc_); 
    double* hess_appx_vec=hess_appx->local_data_host();
   
    hiopVector* x0;
    x0 = LinearAlgebraFactory::createVector(nc_);
    x0->setToZero(); 
    double* x0_vec=x0->local_data_host();
    
    //local recourse terms for each evaluator, defined accross all processors
    double rec_val = 0.;
    hiopVector* grad_acc;
    grad_acc = LinearAlgebraFactory::createVector(nc_);
    grad_acc->setToZero(); 
    double* grad_acc_vec=grad_acc->local_data();

    //double grad_acc[nc_];
    //for(int i=0; i<nc_; i++) grad_acc[i] = 0.;

    //hess_appx_2 is declared by all ranks while only rank 0 uses it
    HessianApprox*  hess_appx_2 = new HessianApprox(nc_,alpha_ratio_);
    if(ver_ >= outlevel3) {
      hess_appx_2->set_verbosity(ver_);
    }

    double base_val = 0.; // base case objective value 
    double recourse_val = 0.;  // recourse objective value
    double dinf = 0.; // step size 

    double convg = 1e20;
    double convg_g = 1e20;
    double convg_f = 1e20;

    int end_signal = 0;
    double t1 = 0;
    double t2 = 0; 
    hiopInterfacePriDecProblem::RecourseApproxEvaluator* evaluator =
      new hiopInterfacePriDecProblem::RecourseApproxEvaluator(nc_, S_, xc_idx_);
    
    double* x_vec = x_->local_data();

    std::string options_file_master_prob;

    // Outer loop starts
    for(int it=0; it<max_iter;it++) {
      
      if(my_rank_==0) {
        t1 = MPI_Wtime(); 
      }
      // solve the base case
      if(my_rank_ == 0 && it==0) {//initial solve 
        // printf("my rank for solver  %d\n", my_rank_);
        // solve master problem base case on master and iteration 0

        options_file_master_prob = options_->GetString("options_file_master_prob");
        
        solver_status_ = master_prob_->solve_master(*x_, false, 0, 0, 0, options_file_master_prob.c_str());
        // to do, what if solve fails?
        if(solver_status_){     

        }
        if(ver_ >=outlevel2) {
	  x_->print();
          //for(int i=0;i<n_;i++) printf("x %d %18.12e ",i,x_[i]);
          //printf("\n ");
        }
	base_val = master_prob_->get_objective();
      }

      // send base case solutions to all ranks
      // todo error control

      int ierr = MPI_Bcast(x_vec, n_, MPI_DOUBLE, rank_master, comm_world_);
      assert(ierr == MPI_SUCCESS);

      //
      // set up recourse problem send/recv interface
      //
      std::vector<ReqRecourseApprox* > rec_prob;
      ReqRecourseApprox* p=NULL;
      for(int r=0; r<comm_size_;r++) {
        p = new ReqRecourseApprox(nc_);
        rec_prob.push_back(p);
      }
      
      ReqContingencyIdx* req_cont_idx = new ReqContingencyIdx(0);

      // master rank communication
      if(my_rank_ == 0) {
        // array for number of indices, currently the indices are in [0,S_] 
        // this is subjected to change	
        rval = 0.;
	grad_r->setToZero();
        
        int* cont_idx = new int[S_];
        for(int i=0;i<S_;i++) {
          cont_idx[i]=i;
        }
        // The number of contigencies should be larger than the number of processors
        // Otherwise not implemented yet
        assert(S_>=comm_size_-1);
        // idx is the next contingency to be sent out from the master
        int idx = 0;
        // Initilize the recourse communication by sending indices to the evaluator 
        // Using Blocking send here
        for(int r=1; r< comm_size_;r++) {
          int cur_idx = cont_idx[idx];
          int ierr = MPI_Send(&cur_idx, 1, MPI_INT, r, 1,comm_world_);
          assert(MPI_SUCCESS == ierr);  
          //printf("rank %d to get contingency index  %d\n", r,cur_idx);
          idx += 1;
        }
        int mpi_test_flag; // for testing if the send/recv is completed
        // Posting initial receive of recourse solutions from evaluators
        for(int r=1; r< comm_size_;r++) {
          //int cur_idx = cont_idx[idx];
          rec_prob[r]->post_recv(2,r,comm_world_);// 2 is the tag, r is the rank source 
          //printf("receive flag for contingency value %d)\n", mpi_test_flag);
        }
        // Both finish_flag and last_loop are used to deal with the final round remaining contingencies.
        // Some ranks are finished while others are not. The loop needs to continue to fetch the results. 
        //hiopVectorInt* finish_flag = LinearAlgebraFactory::createVectorInt(comm_size_);
	//finish_flag->setToZero();
	std::vector<int> finish_flag(comm_size_);
        for(int i=0;i<comm_size_;i++) {
          finish_flag[i]=0;
        }
        int last_loop = 0;
        //printf("total idx %d\n", S_);
        t2 = MPI_Wtime(); 
        if(ver_ >=outlevel2) {
          printf( "Elapsed time for iteration %d for misc is %f\n",it, t2 - t1 );  
        }
        while(idx<=S_ || last_loop) { 
          for(int r=1; r< comm_size_;r++) {
            int mpi_test_flag = rec_prob[r]->test();
            if(mpi_test_flag && (finish_flag[r]==0)) {// receive completed
              if(!last_loop && idx<S_) {
                if(ver_ >=outlevel2) {
                  printf("idx %d sent to rank %d\n", idx,r);
                }
              } else {
                if(ver_ >=outlevel2) {
                  printf("last loop for rank %d\n", r);
                }
              }
              // add to the master rank variables
              rval += rec_prob[r]->value();
              for(int i=0;i<nc_;i++) {
                grad_r_vec[i] += rec_prob[r]->grad(i);
              }
              if(last_loop) {
                finish_flag[r]=1;
              }
              // this is for dealing with the end of contingencies where some ranks have already finished
              if(idx<S_) {
                req_cont_idx->set_idx(cont_idx[idx]);
                req_cont_idx->post_send(1,r,comm_world_);
                rec_prob[r]->post_recv(2,r,comm_world_);// 2 is the tag, r is the rank source 
                //printf("recourse value: is %18.12e)\n", rec_prob[r]->value());
              } else {
                finish_flag[r] = 1;
                last_loop = 1; 
              }
              idx += 1; 
            } 
          }
          // Current way of ending the loop while accounting for all the last round of results
          if(last_loop) {
            last_loop=0;
            for(int r=1; r< comm_size_;r++) {
              if(finish_flag[r]==0){last_loop=1;}
            }
          }

        }
        // send end signal to all evaluators
        int cur_idx = -1;
        for(int r=1; r< comm_size_;r++) {
          req_cont_idx->set_idx(-1);
          req_cont_idx->post_send(1,r,comm_world_);
        }
        t2 = MPI_Wtime(); 
        if(ver_ >=outlevel2) {
          printf( "Elapsed time for iteration %d for contingency is %f\n",it, t2 - t1 );  
        }
      }

      //evaluators
      if(my_rank_ != 0) {
        /* old sychronous implmentation of contingencist
         * int cpr = S_/(comm_size_-1); //contingency per rank
         * int cr = S_%(comm_size_-1); //contingency remained
         * printf("my rank start evaluating work %d)\n",my_rank_);
         */
        std::vector<int> cont_idx(1); // currently sending/receiving one contingency index at a time
        int cont_i = 0;
        cont_idx[0] = 0;
        // int cur_idx = 0;
        // Receive the index of the contingency to evaluate
        int mpi_test_flag = 0;
        int ierr = MPI_Recv(&cont_i, 1, MPI_INT, rank_master, 1,comm_world_, &status_);
        assert(MPI_SUCCESS == ierr);  
        cont_idx[0] = cont_i;
        // printf("contingency index %d, rank %d)\n",cont_idx[0],my_rank_);
        // compute the recourse function values and gradients
        rec_val = 0.;

	grad_acc->setToZero();
        //for(int i=0; i<nc_; i++) {
        //  grad_acc[i] = 0.;
        //}
        double aux=0.;

        if(nc_<n_) {
          assert(xc_idx_[0]>=0);// if nc==0, why bother using this code?
	  x0->copyFrom(xc_idx_,*x_);
          //for(int i=0;i<nc_;i++) {
          //  x0_vec[i] = x_vec[xc_idx_[i]];
          //}
        } else {
          assert(nc_==n_);
          x0->copyFromStarting(0, *x_);
          //for(int i=0;i<nc_;i++) {
          //  x0_vec[i] = x_[i];
          //}
        }
        for(int ri=0; ri<cont_idx.size(); ri++) {
          aux = 0.;
          int idx_temp = cont_idx[ri];

          bret = master_prob_->eval_f_rterm(idx_temp, nc_, x0_vec, aux); // solving the recourse problem
          if(!bret) {
              //todo
          }
          rec_val += aux;
        }
        //printf("recourse value: is %18.12e)\n", rec_val);
	hiopVector* grad_aux;
        grad_aux = LinearAlgebraFactory::createVector(nc_);
        grad_aux->setToZero(); 

        for(int ri=0; ri<cont_idx.size(); ri++) {
          int idx_temp = cont_idx[ri];
          bret = master_prob_->eval_grad_rterm(idx_temp, nc_, x0_vec, *grad_aux);
          if(!bret) {
            //todo
          }
          grad_acc->axpy(1.0, *grad_aux);
          //for(int i=0; i<nc_; i++)
          //  grad_acc[i] += grad_aux[i];
        }
        rec_prob[my_rank_]->set_value(rec_val);

        rec_prob[my_rank_]->set_grad(grad_acc_vec);
        rec_prob[my_rank_]->post_send(2, rank_master, comm_world_);

        req_cont_idx->post_recv(1, rank_master, comm_world_);
        while(cont_idx[0]!=-1) {//loop until end signal received
          mpi_test_flag = req_cont_idx->test();
          /* contigency starts at 0 
           * sychronous implmentation of contingencist
          */
          if(mpi_test_flag) {
            //printf("contingency index %d, rank %d)\n",cont_idx[0],my_rank_);
            for(int ri=0; ri<cont_idx.size(); ri++) {
              cont_idx[ri] = req_cont_idx->value();
            }
            if(cont_idx[0]==-1) {
              break;
            }
            rec_val = 0.;
	    grad_acc->setToZero();
            //for(int i=0; i<nc_; i++) {
            //  grad_acc[i] = 0.;
            //}
            double aux=0.;
            //double x0[nc_]; 
            if(nc_<n_) {
              assert(xc_idx_[0]>=0);// if nc==0, why bother using this code?
	      x0->copyFrom(xc_idx_,*x_);
            } else {
              assert(nc_==n_);
              x0->copyFromStarting(0, *x_);
              //for(int i=0;i<nc_;i++) {
              // x0_vec[i] = x_[i];
              //}
            }
            for(int ri=0; ri<cont_idx.size(); ri++) {
              aux = 0.;
              int idx_temp = cont_idx[ri];
              
              bret = master_prob_->eval_f_rterm(idx_temp, nc_, x0_vec, aux); //need to add extra time here
              if(!bret) {
              //todo
              }
              rec_val += aux;
            }
            //printf("recourse value: is %18.12e)\n", rec_val);
	    hiopVector* grad_aux;
            grad_aux = LinearAlgebraFactory::createVector(nc_);
            grad_aux->setToZero(); 
            //double grad_aux[nc_];
            for(int ri=0; ri<cont_idx.size(); ri++) {
              int idx_temp = cont_idx[ri];
              bret = master_prob_->eval_grad_rterm(idx_temp, nc_, x0_vec, *grad_aux);
              if(!bret) {
                //todo
              }
              grad_acc->axpy(1.0, *grad_aux);
              //for(int i=0; i<nc_; i++) {
              //  grad_acc[i] += grad_aux[i];
              //}
            }

            rec_prob[my_rank_]->set_value(rec_val);

            rec_prob[my_rank_]->set_grad(grad_acc_vec);
            rec_prob[my_rank_]->post_send(2, rank_master, comm_world_);
            //do something with the func eval and gradient to determine the quadratic regularization  
            //printf("send recourse value flag for test %d \n", mpi_test_flag);
        
            //post recv for new index
            req_cont_idx->post_recv(1, rank_master, comm_world_);
            //ierr = MPI_Irecv(&cont_idx[0], 1, MPI_INT, rank_master, 1, comm_world_, &request_[0]); 	  
          }
        }
      }

      if(my_rank_==0) {
        int mpi_test_flag = 0;
        for(int r=1; r<comm_size_;r++) {
          MPI_Wait(&(rec_prob[r]->request_), &status_);
          MPI_Wait(&req_cont_idx->request_, &status_);
        }
        
	recourse_val = rval;

        if(ver_ >=outlevel2) {
          printf("real rval %18.12e\n",rval);
	}
        MPI_Status mpi_status; 

        for(int i=0; i<nc_; i++) {
          hess_appx_vec[i] = 1.0;
	}
    
        if(nc_<n_) {
          assert(xc_idx_[0]>=0);// if nc==0, why bother using this code?
	  x0->copyFrom(xc_idx_,*x_);
          //for(int i=0;i<nc_;i++) {
          //  x0_vec[i] = x_vec[xc_idx_[i]];
          //}
        } else {
          assert(nc_==n_);
          x0->copyFromStarting(0, *x_);
          //for(int i=0;i<nc_;i++) {
          //  x0_vec[i] = x_[i];
          //}
        }

        if(it==0) {
          hess_appx_2->initialize(rval, *x0, *grad_r);
          double alp_temp = hess_appx_2->get_alpha_f(*grad_r);
          if(ver_ >=outlevel2) {
            printf("alpd %18.12e\n",alp_temp);
          }
          for(int i=0; i<nc_; i++) {
            hess_appx_vec[i] = alp_temp;
	  }
        } else {
          hess_appx_2->update_hess_coeff(*x0, *grad_r, rval);

          //hess_appx_2->update_ratio();
          double alp_temp = hess_appx_2->get_alpha_f(*grad_r);
          //double alp_temp2 = hess_appx_2->get_alpha_BB();
          if(ver_ >=outlevel2) {
            printf("alpd %18.12e\n",alp_temp);
          }
          //printf("alpd BB %18.12e\n",alp_temp2);
          convg_g = hess_appx_2->check_convergence_grad(*grad_r);
          if(ver_ >=outlevel2) {
            printf("gradient convergence measure %18.12e\n",convg_g);
          }
          convg_f = hess_appx_2->check_convergence_fcn();
          if(ver_ >=outlevel2) {
            printf("function val convergence measure %18.12e\n",convg_f);
          }
          convg = std::min(convg_f,convg_g);
          for(int i=0; i<nc_; i++) {
            hess_appx_vec[i] = alp_temp;
	  }
	  base_val = hess_appx_2->compute_base(master_prob_->get_objective(),rval);
        }

        // wait for the sending/receiving to finish
        
        // for debugging purpose print out the recourse gradient
        if(ver_ >=outlevel2) {
          for(int i=0;i<nc_;i++) {
            printf("grad %d %18.12e ",i,grad_r_vec[i]);
          }
          printf("\n");
        }
       
        if(ver_ >=outlevel1 && it>0) {
          //printf("iteration         sub_obj              res               step_size           convg\n");
          printf("iteration          objective                 residual                   "   
	         "step_size                   convg\n");
          printf("%d            %18.12e          %18.12e             %18.12e          "
	         "%18.12e\n", it, base_val+recourse_val,convg_f,dinf, convg_g);
          fflush(stdout);
        }

        assert(evaluator->get_rgrad()!=NULL);// should be defined
        evaluator->set_rval(rval);
        evaluator->set_rgrad(nc_,*grad_r);
        evaluator->set_rhess(nc_,*hess_appx);
        evaluator->set_x0(nc_,*x0);

        bret = master_prob_->set_recourse_approx_evaluator(nc_, evaluator);
        if(!bret) {
          //todo
        }
        
        options_file_master_prob = options_->GetString("options_file_master_prob");
        
        //printf("solving full problem starts, iteration %d \n",it);
        solver_status_ = master_prob_->solve_master(*x_, true, 0, 0, 0, options_file_master_prob.c_str());

        if(ver_ >=outlevel2) {
          printf("solved full problem with objective %18.12e\n", master_prob_->get_objective());
          fflush(stdout);
        }

        if(ver_ >=outlevel2) {
	  x_->print();
          //for(int i=0;i<n_;i++) {
          //  printf("x%d %18.12e ",i,x_[i]);
          //}
          //printf(" \n");
        }
        t2 = MPI_Wtime(); 
        if(ver_ >=outlevel2) {
          printf( "Elapsed time for entire iteration %d is %f\n",it, t2 - t1 );  
        }
        // print out the iteration from the master rank
	dinf = step_size_inf(nc_, *x_, *x0);
	

      } else {
        // evaluator ranks do nothing     
      }
      if(convg <= accp_tol_) {
        accp_count += 1;
      } else {
        accp_count = 0;
      }

      if(stopping_criteria(it, convg)) {
        end_signal = 1; 
      }
      ierr = MPI_Bcast(&end_signal, 1, MPI_INT, rank_master, comm_world_);
      assert(ierr == MPI_SUCCESS);
      
     
      if(end_signal) {
        break;
      }

    }
    if(my_rank_==0) {
      return solver_status_;
    } else {
      return Solve_Success;    
    }
  }
#else
  hiopSolveStatus hiopAlgPrimalDecomposition::run()
  {
    return run_single();//call the serial solver
  }
#endif


/* Solve problem in serial with only one rank
 */
hiopSolveStatus hiopAlgPrimalDecomposition::run_single()
{
  printf("total number of recourse problems  %lu\n", S_);
  // initial point for now set to all zero
  x_->setToZero();
      
  bool bret;
  int rank_master=0; //master rank is also the solver rank
  //Define the values and gradients as needed in the master rank
  double rval = 0.;
  //double grad_r[nc_];
  hiopVector* grad_r;
  grad_r = LinearAlgebraFactory::createVector(nc_); 
  double* grad_r_vec=grad_r->local_data_host();
  
  hiopVector* hess_appx;
  hess_appx = LinearAlgebraFactory::createVector(nc_); 
  double* hess_appx_vec=hess_appx->local_data_host();
 
  hiopVector* x0;
  x0 = LinearAlgebraFactory::createVector(nc_); 
  double* x0_vec=x0->local_data_host();
 
  for(int i=0; i<nc_; i++) {
    grad_r_vec[i] = 0.;
  }

  //hess_appx_2 has to be declared by all ranks while only rank 0 uses it
  HessianApprox*  hess_appx_2 = new HessianApprox(nc_,alpha_ratio_);

  hiopInterfacePriDecProblem::RecourseApproxEvaluator* evaluator =
    new hiopInterfacePriDecProblem::RecourseApproxEvaluator(nc_, S_, xc_idx_);

  double base_val = 0.; // base case objective value 
  double recourse_val = 0.;  // recourse objective value
  double dinf = 0.; // step size 
  double convg = 1e20;
  double convg_f = 1e20;
  double convg_g = 1e20;
  double* x_vec = x_->local_data(); 

  std::string options_file_master_prob;

  // Outer loop starts
  for(int it=0; it<max_iter;it++) {
    //printf("iteration  %d\n", it);
    // solve the base case

    if(it==0) {
      options_file_master_prob = options_->GetString("options_file_master_prob");
      //solve master problem base case(solver rank supposed to do it)        
      solver_status_ = master_prob_->solve_master(*x_, false, 0, 0, 0, options_file_master_prob.c_str());
      // to do, what if solve fails?
      if(solver_status_) {     
      }
      base_val = master_prob_->get_objective();
    }

    // array for number of indices, this is subjected to change	
    rval = 0.;
    grad_r->setToZero( );

    int* cont_idx = new int[S_];
    for(int i=0;i<S_;i++) {
      cont_idx[i]=i;
    }
    // The number of contigencies should be larger than the number of processors, which is 1
    // idx is the next contingency to be sent out from the master
    int idx = 0;
    if(nc_<n_) {
      //printf("xc_idx %d ",xc_idx_[0]);
      assert(xc_idx_[0]>=0);// if nc==0, why bother using this code?
      x0->copyFrom(xc_idx_,*x_);
    } else {
      assert(nc_==n_);
      x0->copyFromStarting(0, *x_);
    }
    for(int i=0; i< S_;i++) {
      int idx_temp = cont_idx[i];
      double aux=0.;
      bret = master_prob_->eval_f_rterm(idx_temp, nc_, x0_vec, aux); //need to add extra time here
      if(!bret) {
            //todo
      }
      rval += aux;
      //assert("for debugging" && false); //for debugging purpose
    
      hiopVector* grad_aux;
      grad_aux = LinearAlgebraFactory::createVector(nc_);
      grad_aux->setToZero(); 
      //double grad_aux[nc_];
      bret = master_prob_->eval_grad_rterm(idx_temp, nc_, x0_vec, *grad_aux);
      if(!bret) {
          //todo
      }
      grad_r->axpy(1.0,*grad_aux);
      //for(int i=0; i<nc_; i++) {
      //  grad_r_vec[i] += grad_aux[i];
      //}
    }        
    if(ver_ >=outlevel2) {
      printf("real rval %18.12e\n",rval);
    }
    
    recourse_val = rval;

    for(int i=0; i<nc_; i++) {
      hess_appx_vec[i] = 1e6;
    }
 
    if(it==0) {
      hess_appx_2->initialize(rval, *x0, *grad_r);
      double alp_temp = hess_appx_2->get_alpha_f(*grad_r);
      if(ver_ >=outlevel2) {
        printf("alpd %18.12e\n",alp_temp);
      }
      for(int i=0; i<nc_; i++) {
        hess_appx_vec[i] = alp_temp;
      }
    } else {
      hess_appx_2->update_hess_coeff(*x0, *grad_r, rval);
      //hess_appx_2->update_ratio();
      double alp_temp = hess_appx_2->get_alpha_f(*grad_r);
      if(ver_ >=outlevel2) {
        printf("alpd %18.12e\n",alp_temp);
      }
      convg_g = hess_appx_2->check_convergence_grad(*grad_r);

      if(ver_ >=outlevel2) {
        printf("convergence measure %18.12e\n",convg_g);
      }
      convg_f = hess_appx_2->check_convergence_fcn();
      if(ver_ >=outlevel2) {
        printf("function val convergence measure %18.12e\n",convg_f);
      }
      convg = std::min(convg_f,convg_g);
      for(int i=0; i<nc_; i++) {
        hess_appx_vec[i] = alp_temp;
      }

      base_val = hess_appx_2->compute_base(master_prob_->get_objective(),rval);
    }

    // for debugging purpose print out the recourse gradient
    if(ver_ >=outlevel2) {
      grad_r->print();
      //for(int i=0;i<nc_;i++) {
      //  printf("grad %d  %18.12e ",i,grad_r_vec[i]);
      //}
      //printf(" \n");
    }
    // nc_ is the demesnion of coupled x

    if(ver_ >=outlevel1 && it>0) {
      //printf("iteration         sub_obj              res               step_size          convg\n");
      printf("iteration           objective                   residual                   "   
             "step_size                   convg\n");
      printf("%d              %18.12e            %18.12e           %18.12e         " 
             "%18.12e\n", it, base_val+recourse_val, convg_f, dinf, convg_g);
      fflush(stdout);
    }

    assert(evaluator->get_rgrad()!=NULL);// should be defined
    evaluator->set_rval(rval);
    evaluator->set_rgrad(nc_,*grad_r);
    evaluator->set_rhess(nc_,*hess_appx);
    evaluator->set_x0(nc_,*x0);


    bret = master_prob_->set_recourse_approx_evaluator(nc_, evaluator);
    if(!bret) {
      //todo
    }
    options_file_master_prob = options_->GetString("options_file_master_prob");
    //printf("solving full problem starts, iteration %d \n",it);
    solver_status_ = master_prob_->solve_master(*x_, true, 0, 0, 0, options_file_master_prob.c_str());
    
    dinf = step_size_inf(nc_, *x_, *x0); 

    // print solution x at the end of a full solve
    if(ver_ >=outlevel2) {
      x_->print();
    }
    //assert("for debugging" && false); //for debugging purpose
    if(convg <= accp_tol_) {
      accp_count += 1;
    } else {
      accp_count = 0;
    }
    //printf("count  %d \n", accp_count);
    if(stopping_criteria(it, convg)){break;}
  }
    return Solve_Success;    
}

}//end namespace
