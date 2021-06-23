#include "hiopInterfacePrimalDecomp.hpp"


using namespace hiop;
hiopInterfacePriDecProblem::RecourseApproxEvaluator::
RecourseApproxEvaluator(int nc) : RecourseApproxEvaluator(nc, nc)  //nc_ <= nx, nd=S
{
}

hiopInterfacePriDecProblem::RecourseApproxEvaluator::
~RecourseApproxEvaluator( ) 
{
  delete rgrad_;
  delete rhess_;
  delete x0_;  
}

hiopInterfacePriDecProblem::RecourseApproxEvaluator::
RecourseApproxEvaluator(int nc, int S) 
  : nc_(nc),S_(S),rval_(0.)//nc = nx, nd=S
{
  assert(S>=nc);
  xc_idx_ = new int[nc_];
  for(int i=0;i<nc_;i++) xc_idx_[i] = i;
  rgrad_ = LinearAlgebraFactory::createVector(nc);
  rhess_ = LinearAlgebraFactory::createVector(nc);
  x0_ = LinearAlgebraFactory::createVector(nc);
}

hiopInterfacePriDecProblem::RecourseApproxEvaluator::
RecourseApproxEvaluator(const int nc, 
                        const int S, 
                        const int* list)
  : nc_(nc),S_(S),rval_(0.),rgrad_(NULL), rhess_(NULL),x0_(NULL)//nc = nx, nd=S
{
  rgrad_ = LinearAlgebraFactory::createVector(nc);
  rhess_ = LinearAlgebraFactory::createVector(nc);
  x0_ = LinearAlgebraFactory::createVector(nc);
  //assert(list.size()==nc_);
  xc_idx_ = new int[nc_];
  
  for(int i=0;i<nc_;i++) xc_idx_[i] = list[i];
}

hiopInterfacePriDecProblem::RecourseApproxEvaluator::
RecourseApproxEvaluator(const int nc, 
                        const int S, 
                        const double& rval, 
                        const hiopVector& rgrad, 
                        const hiopVector& rhess, 
                        const hiopVector& x0)
  : nc_(nc), S_(S)
{
  //assert(S>=nc);
  rval_ = rval;
  rgrad_ = LinearAlgebraFactory::createVector(nc);
  //rgrad_ = new double[nc];
  rhess_ = LinearAlgebraFactory::createVector(nc);
  x0_ = LinearAlgebraFactory::createVector(nc);
  xc_idx_ = new int[nc_];
  for(int i=0;i<nc_;i++) xc_idx_[i] = i;

  rgrad_->copyFromStarting(0, rgrad.local_data_const(), nc);
  rhess_->copyFromStarting(0, rhess.local_data_const(), nc);
  x0_->copyFromStarting(0, x0.local_data_const(), nc);
}

hiopInterfacePriDecProblem::RecourseApproxEvaluator::
RecourseApproxEvaluator(const int nc, 
                        const int S, 
                        const int* list,
                        const double& rval, 
                        const hiopVector& rgrad, 
                        const hiopVector& rhess, 
                        const hiopVector& x0)
  : nc_(nc), S_(S)
{
  //assert(S>=nc);
  rval_ = rval;
  rgrad_ = LinearAlgebraFactory::createVector(nc);
  rhess_ = LinearAlgebraFactory::createVector(nc);
  x0_ = LinearAlgebraFactory::createVector(nc);
  xc_idx_ = new int[nc_];
  for(int i=0;i<nc_;i++) xc_idx_[i] = list[i];

  rgrad_->copyFromStarting(0, rgrad.local_data_const(), nc);
  //memcpy(rgrad_,rgrad, nc*sizeof(double));
  rhess_->copyFromStarting(0, rhess.local_data_const(), nc);
  x0_->copyFromStarting(0, x0.local_data_const(), nc);
}

/**
 * x is the full optimization variable for the base problem
 * x0 only stores the coupled x
 * Therefore need to pick out the coupled ones
 * n is the total dimension of x and not really used in the function
 */
bool hiopInterfacePriDecProblem::RecourseApproxEvaluator::
eval_f(const size_type& n, const double* x, bool new_x, double& obj_value)
{
  //assert(nc>=4);
  //sum 0.5 {(x_i-1)*(x_{i}-1) : i=1,...,nc} 
  assert(rgrad_!=NULL);
  obj_value += rval_;

  hiopVector* temp;
  temp = LinearAlgebraFactory::createVector(nc_); 
  temp->copyFrom(xc_idx_,x);   
  temp->axpy(-1.0,*x0_);
  obj_value += temp->dotProductWith(*rgrad_);

  temp->componentMult(*temp);
  obj_value += 0.5*temp->dotProductWith(*rhess_);
  
  return true;
}
 
// grad is assumed to be of the length n, of the entire x
bool hiopInterfacePriDecProblem::RecourseApproxEvaluator::
eval_grad(const size_type& n, const double* x, bool new_x, double* grad)
{
  assert(rgrad_!=NULL);
  double* rgrad_arr = rgrad_->local_data();
  double* rhess_arr = rhess_->local_data();
  double* x0_arr = x0_->local_data();

  hiopVector* temp;
  temp = LinearAlgebraFactory::createVector(nc_); 
  temp->copyFrom(xc_idx_,x);
  temp->axpy(-1.0,*x0_);
  temp->componentMult(*rhess_);
  temp->axpy(1.0,*rgrad_);

  for(int i=0; i<nc_; i++) {
    grad[xc_idx_[i]] += temp->local_data()[i];
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
eval_hess(const size_type& n, const hiopVector& x, bool new_x, hiopVector& hess)
{
  assert(rgrad_!=NULL);
  assert(rhess_->get_local_size()==hess.get_local_size());
  hess.axpy(1.0,*rhess_); 
  return true;
}

// pass the COMM_SELF communicator since this example is only intended to run inside 1 MPI process //
bool hiopInterfacePriDecProblem::RecourseApproxEvaluator::
get_MPI_comm(MPI_Comm& comm_out) { comm_out=MPI_COMM_SELF; return true;}
  

void hiopInterfacePriDecProblem::RecourseApproxEvaluator::
set_rval(const double rval){rval_ = rval;}
  
void hiopInterfacePriDecProblem::RecourseApproxEvaluator::
set_rgrad(const int n, const hiopVector& rgrad)
{
  assert(n == nc_);
  if(rgrad_==NULL) {
    rgrad_ = LinearAlgebraFactory::createVector(nc_);
  }
  rgrad_->copyFromStarting(0, rgrad.local_data_const(), nc_);
}

// setting the hess vector of size nc_, diagnol of the Hessian matrix on
// relevant x
void hiopInterfacePriDecProblem::RecourseApproxEvaluator::
set_rhess(const int n, const hiopVector& rhess)
{
  assert(n == nc_);
  if(rgrad_==NULL) {
    rhess_ = LinearAlgebraFactory::createVector(nc_);
  }
  rhess_->copyFromStarting(0, rhess.local_data_const(), nc_);
}


void hiopInterfacePriDecProblem::RecourseApproxEvaluator::
set_x0(const int n, const hiopVector& x0)
{
  assert(n == nc_);
  if(rgrad_==NULL) {
    x0_ = LinearAlgebraFactory::createVector(nc_);
  }
  x0_->copyFromStarting(0, x0.local_data_const(), nc_);
}

void hiopInterfacePriDecProblem::RecourseApproxEvaluator::
set_xc_idx(const int* idx)
{
  if(xc_idx_==NULL){ 
    xc_idx_ = new int[nc_];
  }
  for(int i=0;i<nc_;i++) xc_idx_[i] = idx[i];
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

hiopVector* hiopInterfacePriDecProblem::
RecourseApproxEvaluator::get_rgrad() const 
{
  return rgrad_;
}

hiopVector* hiopInterfacePriDecProblem::
RecourseApproxEvaluator::get_rhess() const 
{
  return rhess_;
}

hiopVector* hiopInterfacePriDecProblem::
RecourseApproxEvaluator::get_x0() const 
{
  return x0_;
}

int* hiopInterfacePriDecProblem::
RecourseApproxEvaluator::get_xc_idx() const 
{
  return xc_idx_;
}


