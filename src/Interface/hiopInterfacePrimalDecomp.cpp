#include "hiopInterfacePrimalDecomp.hpp"


using namespace hiop;
hiopInterfacePriDecProblem::RecourseApproxEvaluator::
RecourseApproxEvaluator(int nc, const std::string& mem_space)
  : RecourseApproxEvaluator(nc, nc, mem_space)  //nc_ <= nx, nd=S
{
}

hiopInterfacePriDecProblem::RecourseApproxEvaluator::
~RecourseApproxEvaluator() 
{
  delete rgrad_;
  delete rhess_;
  delete x0_;  
}

hiopInterfacePriDecProblem::RecourseApproxEvaluator::
RecourseApproxEvaluator(int nc, int S, const std::string& mem_space) 
  : nc_(nc), S_(S), rval_(0.), //nc = nx, nd=S
    mem_space_(mem_space)
{
  assert(S>=nc);
  xc_idx_ = LinearAlgebraFactory::create_vector_int(mem_space_, nc);
  xc_idx_->linspace(0,1);
  
  rgrad_ = LinearAlgebraFactory::create_vector(mem_space_, nc);
  rhess_ = rgrad_->alloc_clone();
  x0_ = rgrad_->alloc_clone();
}

hiopInterfacePriDecProblem::RecourseApproxEvaluator::
RecourseApproxEvaluator(const int nc, 
                        const int S, 
                        const int* list,
                        const std::string& mem_space)
  : nc_(nc), S_(S), rval_(0.), rgrad_(NULL), rhess_(NULL), x0_(NULL),
    mem_space_(mem_space)
{
  rgrad_ = LinearAlgebraFactory::create_vector(mem_space_, nc);
  rhess_ = rgrad_->alloc_clone();
  x0_ = rgrad_->alloc_clone();

  xc_idx_ = LinearAlgebraFactory::create_vector_int(mem_space_, nc);
  xc_idx_->copy_from(list);
}

hiopInterfacePriDecProblem::RecourseApproxEvaluator::
RecourseApproxEvaluator(const int nc, 
                        const int S, 
                        const double& rval, 
                        const hiopVector& rgrad, 
                        const hiopVector& rhess, 
                        const hiopVector& x0,
                        const std::string& mem_space)
  : nc_(nc), S_(S),
    mem_space_(mem_space)
{
  //assert(S>=nc);
  rval_ = rval;
  rgrad_ = LinearAlgebraFactory::create_vector(mem_space_, nc);
  rhess_ = rgrad_->alloc_clone();

  x0_ = rgrad_->alloc_clone();

  xc_idx_ = LinearAlgebraFactory::create_vector_int(mem_space_, nc);
  xc_idx_->linspace(0,1);

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
                        const hiopVector& x0,
                        const std::string& mem_space)
  : nc_(nc),
    S_(S),
    mem_space_(mem_space)
{
  //assert(S>=nc);
  rval_ = rval;
  rgrad_ = LinearAlgebraFactory::create_vector(mem_space_, nc);
  rhess_ = rgrad_->alloc_clone();
  x0_ = rgrad_->alloc_clone();

  xc_idx_ = LinearAlgebraFactory::create_vector_int(mem_space_, nc);
  xc_idx_->copy_from(list);

  rgrad_->copyFromStarting(0, rgrad.local_data_const(), nc);
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
  assert(rgrad_!=NULL);
  obj_value += rval_;

  //TODO: have a buffer_x member in the class, of type hiopVector;
  //      allocate once and reuse to avoid repeated allocations
  hiopVector* temp = rgrad_->alloc_clone(); 
  temp->copy_from_indexes(x, *xc_idx_);   
  temp->axpy(-1.0, *x0_);
  obj_value += temp->dotProductWith(*rgrad_);

  temp->componentMult(*temp);
  obj_value += 0.5*temp->dotProductWith(*rhess_);
  delete temp;
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
  
  //TODO: have a buffer_x member in the class, of type hiopVector;
  //      allocate once and reuse to avoid repeated allocations
  hiopVector* temp = rgrad_->alloc_clone(); 
  temp->copy_from_indexes(x, *xc_idx_);
  temp->axpy(-1.0, *x0_);
  temp->componentMult(*rhess_);
  temp->axpy(1.0, *rgrad_);

  //TODO: this is cpu code
  //for(int i=0; i<nc_; i++) {
  //  grad[xc_idx_[i]] += temp->local_data()[i];
  //}

  hiopVector* grad_vec = LinearAlgebraFactory::create_vector(mem_space_, 0);
  grad_vec->attach_to(grad, n);
  
  grad_vec->axpy(1.0, *temp, *xc_idx_);
  delete grad_vec;
  
  delete temp;
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
bool hiopInterfacePriDecProblem::RecourseApproxEvaluator::get_MPI_comm(MPI_Comm& comm_out)
{
  comm_out=MPI_COMM_SELF;
  return true;
}
  

void hiopInterfacePriDecProblem::RecourseApproxEvaluator::set_rval(const double rval)
{
  rval_ = rval;
}
  
void hiopInterfacePriDecProblem::RecourseApproxEvaluator::set_rgrad(const int n, const hiopVector& rgrad)
{
  assert(n == nc_);
  assert(rgrad.get_size()>=nc_);
  if(rgrad_==NULL) {
    rgrad_ = LinearAlgebraFactory::create_vector(mem_space_, nc_);
  }
  rgrad_->copyFromStarting(0, rgrad.local_data_const(), nc_);
}

// setting the hess vector of size nc_, diagnol of the Hessian matrix on
// relevant x
void hiopInterfacePriDecProblem::RecourseApproxEvaluator::set_rhess(const int n, const hiopVector& rhess)
{
  assert(n == nc_);
  if(rhess_==NULL) {
    rhess_ = LinearAlgebraFactory::create_vector(mem_space_, nc_);
  }
  rhess_->copyFromStarting(0, rhess.local_data_const(), nc_);
}

void hiopInterfacePriDecProblem::RecourseApproxEvaluator::set_x0(const int n, const hiopVector& x0)
{
  assert(n == nc_);
  if(rgrad_==NULL) {
    x0_ = LinearAlgebraFactory::create_vector(mem_space_, nc_);
  }
  x0_->copyFromStarting(0, x0.local_data_const(), nc_);
}

int hiopInterfacePriDecProblem::RecourseApproxEvaluator::get_S() const 
{
  return S_;
} 

double hiopInterfacePriDecProblem::RecourseApproxEvaluator::get_rval() const 
{
  return rval_;
}

hiopVector* hiopInterfacePriDecProblem::RecourseApproxEvaluator::get_rgrad() const 
{
  return rgrad_;
}

hiopVector* hiopInterfacePriDecProblem::RecourseApproxEvaluator::get_rhess() const 
{
  return rhess_;
}

hiopVector* hiopInterfacePriDecProblem::RecourseApproxEvaluator::get_x0() const 
{
  return x0_;
}


