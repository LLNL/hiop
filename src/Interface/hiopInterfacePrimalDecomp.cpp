#include "hiopInterfacePrimalDecomp.hpp"


using namespace hiop;
hiopInterfacePriDecProblem::RecourseApproxEvaluator::
RecourseApproxEvaluator(int nc) : RecourseApproxEvaluator(nc, nc)  //nc_ <= nx, nd=S
{
}

hiopInterfacePriDecProblem::RecourseApproxEvaluator::
RecourseApproxEvaluator(int nc, int S) 
  : nc_(nc),S_(S),rval_(0.)//nc = nx, nd=S
{
  assert(S>=nc);
  xc_idx_.resize(nc_);
  for(int i=0;i<nc_;i++) xc_idx_[i] = i;
  rgrad_ = new double[nc];
  rhess_ = new double[nc];
  x0_ = new double[nc];
}

hiopInterfacePriDecProblem::RecourseApproxEvaluator::
RecourseApproxEvaluator(const int nc, 
                        const int S, 
                        const std::vector<int>& list)
  : nc_(nc),S_(S),rval_(0.),rgrad_(NULL), rhess_(NULL),x0_(NULL)//nc = nx, nd=S
{
  rgrad_ = new double[nc];
  rhess_ = new double[nc];
  x0_ = new double[nc];
  assert(list.size()==nc_);
  xc_idx_ = list;
}

hiopInterfacePriDecProblem::RecourseApproxEvaluator::
RecourseApproxEvaluator(const int nc, 
                        const int S, 
                        const double& rval, 
                        const double* rgrad, 
                        const double* rhess, 
                        const double* x0)
  : nc_(nc), S_(S)
{
  //assert(S>=nc);
  rval_ = rval;
  rgrad_ = new double[nc];
  rhess_ = new double[nc];
  x0_ = new double[nc];
    
  xc_idx_.resize(nc_);
  for(int i=0;i<nc_;i++) xc_idx_[i] = i;

  memcpy(rgrad_,rgrad, nc*sizeof(double));
  memcpy(rhess_,rhess , nc*sizeof(double));
  memcpy(x0_,x0, nc*sizeof(double));
}

hiopInterfacePriDecProblem::RecourseApproxEvaluator::
RecourseApproxEvaluator(const int nc, 
                        const int S, 
                        const std::vector<int>& list,
                        const double& rval, 
                        const double* rgrad, 
                        const double* rhess, 
                        const double* x0)
  : nc_(nc), S_(S)
{
  //assert(S>=nc);
  rval_ = rval;
  rgrad_ = new double[nc];
  rhess_ = new double[nc];
  x0_ = new double[nc];
  assert(list.size()==nc_);
  xc_idx_ = list;

  memcpy(rgrad_,rgrad, nc*sizeof(double));
  memcpy(rhess_,rhess , nc*sizeof(double));
  memcpy(x0_,x0, nc*sizeof(double));
}

/**
 * x is the full optimization variable for the base problem
 * x0 only stores the coupled x
 * Therefore need to pick out the coupled ones
 * n is the total dimension of x and not really used in the function
 */
bool hiopInterfacePriDecProblem::RecourseApproxEvaluator::
eval_f(const long long& n, const double* x, bool new_x, double& obj_value)
{
  //assert(nc>=4);
  //sum 0.5 {(x_i-1)*(x_{i}-1) : i=1,...,nc} 
  assert(rgrad_!=NULL);
  obj_value += rval_;

  for(int i=0; i<nc_; i++) {
    obj_value += rgrad_[i]*(x[xc_idx_[i]]-x0_[i]);
  }
  for(int i=0; i<nc_; i++) {
    obj_value += 0.5*rhess_[i]*(x[xc_idx_[i]]-x0_[i])*(x[xc_idx_[i]]-x0_[i]);
  }
  return true;
}
 
// grad is assumed to be of the length n, of the entire x
bool hiopInterfacePriDecProblem::RecourseApproxEvaluator::
eval_grad(const long long& n, const double* x, bool new_x, double* grad)
{
  assert(rgrad_!=NULL);
  for(int i=0; i<nc_; i++) {
    grad[xc_idx_[i]] += rgrad_[i]+rhess_[i]*(x[xc_idx_[i]]-x0_[i]);
  }
  return true;
}

/**
 * Hessian evaluation is different since it's hard to decipher the 
 * sepcific Lagrangian arrangement at this level
 * So hess currently is a vector of nc_ length 
 * Careful when implementing in the full problem  
 */
bool hiopInterfacePriDecProblem::RecourseApproxEvaluator::
eval_hess(const long long& n, const double* x, bool new_x, double* hess)
{
  assert(rgrad_!=NULL);
  for(int i=0; i<nc_; i++) { 
    hess[i] += rhess_[i];
  }
  return true;
}

// pass the COMM_SELF communicator since this example is only intended to run inside 1 MPI process //

bool hiopInterfacePriDecProblem::RecourseApproxEvaluator::
get_MPI_comm(MPI_Comm& comm_out) { comm_out=MPI_COMM_SELF; return true;}
  

void hiopInterfacePriDecProblem::RecourseApproxEvaluator::
set_rval(const double rval){rval_ = rval;}
  
void hiopInterfacePriDecProblem::RecourseApproxEvaluator::
set_rgrad(const int n,const double* rgrad)
{
  assert(n == nc_);
  if(rgrad_==NULL) {
    rgrad_ = new double[nc_];
  }
  memcpy(rgrad_,rgrad , nc_*sizeof(double));
}

// setting the hess vector of size nc_, diagnol of the Hessian matrix on
// relevant x
void hiopInterfacePriDecProblem::RecourseApproxEvaluator::
set_rhess(const int n,const double* rhess)
{
  assert(n == nc_);
  if(rgrad_==NULL) {
    rhess_ = new double[nc_];
  }
  memcpy(rhess_,rhess , nc_*sizeof(double));
}


void hiopInterfacePriDecProblem::RecourseApproxEvaluator::
set_x0(const int n,const double* x0)
{
  assert(n == nc_);
  if(rgrad_==NULL) {
    rhess_ = new double[nc_];
  }
  memcpy(x0_,x0 , nc_*sizeof(double));
}

void hiopInterfacePriDecProblem::RecourseApproxEvaluator::
set_xc_idx(const std::vector<int>& idx)
{
  assert(nc_=idx.size());
  xc_idx_ = idx;
}

int hiopInterfacePriDecProblem::
RecourseApproxEvaluator::get_S() const 
{
  return S_;
} 

double hiopInterfacePriDecProblem::
RecourseApproxEvaluator::get_rval() const 
{
  return rval_;
}

double* hiopInterfacePriDecProblem::
RecourseApproxEvaluator::get_rgrad() const 
{
  return rgrad_;
}

double* hiopInterfacePriDecProblem::
RecourseApproxEvaluator::get_rhess() const 
{
  return rhess_;
}

double* hiopInterfacePriDecProblem::
RecourseApproxEvaluator::get_x0() const 
{
  return x0_;
}

std::vector<int> hiopInterfacePriDecProblem::
RecourseApproxEvaluator::get_xc_idx() const 
{
  return xc_idx_;
}


