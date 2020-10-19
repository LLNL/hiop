#include "nlpMDSForm_raja_ex4.hpp"

#include <umpire/Allocator.hpp>
#include <umpire/ResourceManager.hpp>
#include <RAJA/RAJA.hpp>
#include <hiopMatrixDenseRowMajor.hpp>
#include <hiopMatrixRajaDense.hpp>

#ifdef HIOP_USE_GPU
  #include "cuda.h"
  #define RAJA_CUDA_BLOCK_SIZE 128
  using ex4_raja_exec   = RAJA::cuda_exec<RAJA_CUDA_BLOCK_SIZE>;
  using ex4_raja_reduce = RAJA::cuda_reduce;
  #define RAJA_LAMBDA [=] __device__
#else
  using ex4_raja_exec   = RAJA::omp_parallel_for_exec;
  using ex4_raja_reduce = RAJA::omp_reduce;
  #define RAJA_LAMBDA [=]
#endif


using namespace hiop;

Ex4::Ex4(int ns_in, int nd_in, std::string mem_space)
  : mem_space_(mem_space), 
    ns_(ns_in),
    sol_x_(NULL),
    sol_zl_(NULL),
    sol_zu_(NULL),
    sol_lambda_(NULL)
{
  // Make sure mem_space_ is uppercase
  transform(mem_space_.begin(), mem_space_.end(), mem_space_.begin(), ::toupper);

  auto& resmgr = umpire::ResourceManager::getInstance();
  umpire::Allocator allocator 
    = resmgr.getAllocator(mem_space_ == "DEFAULT" ? "HOST" : mem_space_);

  if(ns_<0)
  {
    ns_ = 0;
  }
  else
  {
    if(4*(ns_/4) != ns_)
    {
      ns_ = 4*((4+ns_)/4);
      std::cout << "[warning] number " << ns_in 
                << " of sparse vars is not a multiple ->was altered to " << ns_
                << "\n";
    }
  }

  if(nd_in<0) 
    nd_=0;
  else
    nd_ = nd_in;

  // Allocate data buffer and matrices Q_ and Md_
  if(mem_space_ == "DEFAULT")
  {
    Q_  = new hiop::hiopMatrixDenseRowMajor(nd_, nd_);
    Md_ = new hiop::hiopMatrixDenseRowMajor(ns_, nd_);
    buf_y_ = new double[nd_];
  }
  else
  {
    Q_  = new hiop::hiopMatrixRajaDense(nd_, nd_, mem_space_);
    Md_ = new hiop::hiopMatrixRajaDense(ns_, nd_, mem_space_);
    buf_y_ = static_cast<double*>(allocator.allocate(nd_ * sizeof(double)));
  }

  haveIneq_ = true;

  initialize();
}

Ex4::~Ex4()
{
  auto& resmgr = umpire::ResourceManager::getInstance();
  umpire::Allocator allocator;
  if(mem_space_ == "DEFAULT")
    allocator = resmgr.getAllocator("HOST");
  else
    allocator = resmgr.getAllocator(mem_space_);

  allocator.deallocate(sol_lambda_);
  allocator.deallocate(sol_zu_);
  allocator.deallocate(sol_zl_);
  allocator.deallocate(sol_x_);

  if(mem_space_ == "DEFAULT")
    delete [] buf_y_;
  else
    allocator.deallocate(buf_y_);

  delete Md_;
  delete Q_;
}

void Ex4::initialize()
{
  Q_->setToConstant(1e-8);
  Q_->addDiagonal(2.);

  double* data = Q_->local_buffer();
  RAJA::View<double, RAJA::Layout<2>> Qview(data, nd_, nd_);
  RAJA::forall<ex4_raja_exec>(RAJA::RangeSegment(1, nd_-1),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      Qview(i,   i+1) = 1.0;
      Qview(i+1, i)   = 1.0;
    });

  Md_->setToConstant(-1.0);
}

bool Ex4::get_prob_sizes(long long& n, long long& m)
{ 
  n = 2*ns_ + nd_;
  m = ns_ + 3*( haveIneq_ ? 1 : 0 ); 
  return true; 
}

/**
 * @todo will param _type_ live on host or device?
 * @todo register pointers with umpire in case they need to be copied
 * from device to host.
 */
bool Ex4::get_vars_info(const long long& n, double *xlow, double* xupp, NonlinearityType* type)
{
  //assert(n>=4 && "number of variables should be greater than 4 for this example");
  assert(n == 2*ns_+nd_);
  int ns = ns_;

  //x
  RAJA::forall<ex4_raja_exec>(RAJA::RangeSegment(0, ns_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      xlow[i] = -1e+20;
    });

  //s
  RAJA::forall<ex4_raja_exec>(RAJA::RangeSegment(ns_, 2*ns_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      xlow[i] = 0.;
    });

  //y 
  // xlow[2*ns_] = -4.;
  // for(int i=2*ns_+1; i<n; ++i) xlow[i] = -1e+20;
  RAJA::forall<ex4_raja_exec>(RAJA::RangeSegment(2*ns_, n),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      /// @todo move this assignment outside the kernel somehow
      /// limiting factor is that this will eventually have to run on
      /// a GPU device, and cannot be assigned to directly. This is a
      /// workaround for now.
      if (i == 2*ns)
      {
        xlow[i] = -4.;
      }
      else
      {
        xlow[i] = -1e+20;
      }
    });

  //x
  RAJA::forall<ex4_raja_exec>(RAJA::RangeSegment(0, ns_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      xupp[i] = 3.;
    });

  //s
  RAJA::forall<ex4_raja_exec>(RAJA::RangeSegment(ns_, 2*ns_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      xupp[i] = +1e20;
    });

  //y
  // xupp[2*ns_] = 4.;
  // for(int i=2*ns_+1; i<n; ++i) xupp[i] = +1e+20;
  RAJA::forall<ex4_raja_exec>(RAJA::RangeSegment(2*ns_, n),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      /// @todo Same situation as above case. Figure out how to
      /// remove conditional.
      if (i == 2*ns)
      {
        xupp[i] = 4.;
      }
      else
      {
        xupp[i] = 1e+20;
      }
    });

  /// Using OpenMP execution policy for nonlinearity type setting
  RAJA::forall<RAJA::omp_parallel_for_exec>(RAJA::RangeSegment(0, n),
    [=](RAJA::Index_type i)
    {
      type[i] = hiopNonlinear;
    });
  return true;
}

/**
 * @todo fill out param descriptions below to determine whether or not
 * they will reside on device and will have to be accessed/assigned to 
 * in a RAJA kernel
 *
 * @param[out] m    - number of constraints
 * @param[out] clow - inequality constraint lower bound
 * @param[out] cupp - inequality constraint upper bound
 * @param[out] type - constraint type
 */
bool Ex4::get_cons_info(const long long& m, double* clow, double* cupp, NonlinearityType* type)
{
  assert(m == ns_ + 3*haveIneq_);
  bool haveIneq = haveIneq_; ///< Cannot capture member variable in RAJA lambda
  //x+s - Md_ y = 0, i=1,...,ns_
  RAJA::forall<ex4_raja_exec>(RAJA::RangeSegment(0, ns_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      clow[i] = cupp[i] = 0.;
      if(i == 0 && haveIneq)
      {
        // [-2  ]    [ x_1 + e^T s]   [e^T]      [ 2 ]
        clow[m-3] = -2;                cupp[m-3] = 2.;
        // [-inf] <= [ x_2        ] + [e^T] y <= [ 2 ]
        clow[m-2] = -1e+20;            cupp[m-2] = 2.;
        // [-2  ]    [ x_3        ]   [e^T]      [inf]
        clow[m-1] = -2;                cupp[m-1] = 1e+20;
      }
    });

  /// Using OpenMP execution policy for nonlinearity type setting
  RAJA::forall<RAJA::omp_parallel_for_exec>(RAJA::RangeSegment(0, m),
    [=](RAJA::Index_type i)
    {
      type[i] = hiopNonlinear;
    });

  return true;
}

bool Ex4::get_sparse_dense_blocks_info(int& nx_sparse, int& nx_dense,
    int& nnz_sparse_Jace, int& nnz_sparse_Jaci,
    int& nnz_sparse_Hess_Lagr_SS, int& nnz_sparse_Hess_Lagr_SD)
{
  nx_sparse = 2*ns_;
  nx_dense = nd_;
  nnz_sparse_Jace = 2*ns_;
  nnz_sparse_Jaci = (ns_==0 || !haveIneq_) ? 0 : 3+ns_;
  nnz_sparse_Hess_Lagr_SS = 2*ns_;
  nnz_sparse_Hess_Lagr_SD = 0.;
  return true;
}

bool Ex4::eval_f(const long long& n, const double* x, bool new_x, double& obj_value)
{
  //assert(ns_>=4);
  assert(Q_->n()==nd_); assert(Q_->m()==nd_);
  obj_value=0.;//x[0]*(x[0]-1.);

  {
    RAJA::ReduceSum<ex4_raja_reduce, double> aux(0);
    RAJA::forall<ex4_raja_exec>(RAJA::RangeSegment(0, ns_),
      RAJA_LAMBDA(RAJA::Index_type i)
      {
        aux += x[i] * (x[i]-1.);
      });
    obj_value += aux.get();
    obj_value *= 0.5;
  }

  {
    const double* y = x+2*ns_;
    Q_->timesVec(0.0, buf_y_, 1., y);
    double* _buf_y_vec = this->buf_y_;
    RAJA::ReduceSum<ex4_raja_reduce, double> aux(0);
    RAJA::forall<ex4_raja_exec>(RAJA::RangeSegment(0, nd_),
      RAJA_LAMBDA(RAJA::Index_type i)
      {
        aux += _buf_y_vec[i] * y[i];
      });
    obj_value += 0.5 * aux.get();
  }

  {
    const double* s=x+ns_;
    RAJA::ReduceSum<ex4_raja_reduce, double> aux(0);
    RAJA::forall<ex4_raja_exec>(RAJA::RangeSegment(0, ns_),
      RAJA_LAMBDA(RAJA::Index_type i)
      {
        aux += s[i]*s[i];
      });
    obj_value += 0.5 * aux.get();
  }

  return true;
}

/**
 * @todo figure out which of these pointers (if any) will need to be
 * copied over to device when this is fully running on device.
 * @todo find descriptoins of parameters (perhaps from ipopt interface?).
 *
 * @param[in] idx_cons ?
 * @param[in] x ?
 * @param[in] cons ?
 */
bool Ex4::eval_cons(const long long& n, const long long& m, 
    const long long& num_cons, const long long* idx_cons_in,
    const double* x, bool new_x, double* cons)
{
  const double* s = x+ns_;
  const double* y = x+2*ns_;

  assert(num_cons==ns_ || num_cons==3*haveIneq_);

  // Temporary solution: Move idx_cons array to GPU, works with UM and PINNED only
  long long* idx_cons = nullptr;
  auto& resmgr = umpire::ResourceManager::getInstance();
  if(mem_space_ == "DEFAULT")
  {
    idx_cons = new long long[num_cons];
  }
  else
  {
    umpire::Allocator allocator = resmgr.getAllocator(mem_space_);
    idx_cons = static_cast<long long*>(allocator.allocate(num_cons * sizeof(long long)));
  }

  for(int i=0; i<num_cons; ++i)
  {
    idx_cons[i] = idx_cons_in[i];
  }

  int ns = ns_; ///< Cannot capture member inside RAJA lambda
  int nd = nd_; ///< Cannot capture member inside RAJA lambda
  
  // equality constraints
  if(num_cons == ns_ && ns_ > 0)
  {
    RAJA::forall<ex4_raja_exec>(RAJA::RangeSegment(0, num_cons),
      RAJA_LAMBDA(RAJA::Index_type irow)
      {
        const int con_idx = (int) idx_cons[irow];
        if(con_idx < ns)
        {
          //equalities: x+s - Md_ y = 0
          cons[con_idx] = x[con_idx] + s[con_idx];
        }
      });
    Md_->timesVec(1.0, cons, 1.0, y);
  }

  /// @todo This is parallelizing only 3 iterations
  // inequality constraints
  if(num_cons == 3 && haveIneq_)
  {
    RAJA::forall<ex4_raja_exec>(RAJA::RangeSegment(0, num_cons),
      RAJA_LAMBDA(RAJA::Index_type irow)
      {
        const int con_idx = (int) idx_cons[irow];
        assert(con_idx < ns+3);
        const int conineq_idx=con_idx - ns;
        if(conineq_idx==0)
        {
          cons[conineq_idx] = x[0];
          for(int i=0; i<ns; i++) cons[conineq_idx] += s[i];
          for(int i=0; i<nd; i++) cons[conineq_idx] += y[i];
        }
        else if(conineq_idx==1)
        {
          cons[conineq_idx] = x[1];
          for(int i=0; i<nd; i++) cons[conineq_idx] += y[i];
        }
        else if(conineq_idx==2)
        {
          cons[conineq_idx] = x[2];
          for(int i=0; i<nd; i++) cons[conineq_idx] += y[i];
        }
        else
        {
          assert(false);
        }
      });
  }

  // Temporary solution: Move idx_cons array to GPU, works with UM and PINNED only
  if(mem_space_ == "DEFAULT")
  {
    delete [] idx_cons;
  }
  else
  {
    umpire::Allocator allocator = resmgr.getAllocator(mem_space_);
    allocator.deallocate(idx_cons);
  }
  return true;
}

//sum 0.5 {x_i*(x_{i}-1) : i=1,...,ns_} + 0.5 y'*Qd*y + 0.5 s^T s
bool Ex4::eval_grad_f(const long long& n, const double* x, bool new_x, double* gradf)
{
  //! assert(ns_>=4); assert(Q_->n()==ns_/4); assert(Q_->m()==ns_/4);
  //x_i - 0.5 
  RAJA::forall<ex4_raja_exec>(RAJA::RangeSegment(0, ns_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      gradf[i] = x[i] - 0.5;
    });

  //Qd*y
  const double* y = x+2*ns_;
  double* gradf_y = gradf+2*ns_;
  Q_->timesVec(0.0, gradf_y, 1., y);

  //s
  const double* s=x+ns_;
  double* gradf_s = gradf+ns_;
  RAJA::forall<ex4_raja_exec>(RAJA::RangeSegment(0, ns_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      gradf_s[i] = s[i];
    });

  return true;
}

/**
 * @brief Evaluate Jacobian for equality and inequality constraints
 * 
 * @param[in] n - number of variables
 * @param[in] m - number of equality and inequality constraints
 * @param[in] num_cons - number of constraints to update
 * @param[in] idx_cons - indices of constraints to update
 * @param[in] x - solution vector (optimization variables)
 * @param[in] new_x - if variable is updated (?) 
 * @param[in] nsparse - number of sparse variables 
 * @param[in] ndense  - number of dense variables 
 * @param[in] nnzJacS - number of nonzeros in sparse Jacobian block
 * @param[out] - sparse matrix row indices
 * @param[out] - sparse matrix column indices
 * @param[out] - sparse matrix values
 * @param[out] - array to dense matrix row pointers
 * 
 * This method runs on GPU.
 * 
 */
bool Ex4::eval_Jac_cons(const long long& n, const long long& m, 
    const long long& num_cons, const long long* idx_cons_in,
    const double* x, bool new_x,
    const long long& nsparse, const long long& ndense, 
    const int& nnzJacS, int* iJacS, int* jJacS, double* MJacS, 
    double** JacD)
{
  assert(num_cons==ns_ || num_cons==3*haveIneq_);

  // Temporary solution: Move idx_cons array to GPU, works with UM and PINNED only
  long long* idx_cons = nullptr;
  auto& resmgr = umpire::ResourceManager::getInstance();
  if(mem_space_ == "DEFAULT")
  {
    idx_cons = new long long[num_cons];
  }
  else
  {
    umpire::Allocator allocator = resmgr.getAllocator(mem_space_);
    idx_cons = static_cast<long long*>(allocator.allocate(num_cons * sizeof(long long)));
  }

  for(int i=0; i<num_cons; ++i)
  {
    idx_cons[i] = idx_cons_in[i];
  }

  int ns = ns_; ///< Cannot capture member inside RAJA lambda
  
  if(iJacS!=NULL && jJacS!=NULL)
  {
    // Compute equality constraints Jacobian
    if(num_cons==ns_ && ns_>0)
    {
      assert(2*ns_==nnzJacS);
      RAJA::forall<ex4_raja_exec>(RAJA::RangeSegment(0, num_cons),
        RAJA_LAMBDA(RAJA::Index_type itrow)
        {
          const int con_idx = (int) idx_cons[itrow];
          //sparse Jacobian eq w.r.t. x and s
          //x
          iJacS[2*itrow] = con_idx;
          jJacS[2*itrow] = con_idx;
          if(MJacS != nullptr)
            MJacS[2*itrow] = 1.0;

          //s
          iJacS[2*itrow+1] = con_idx;
          jJacS[2*itrow+1] = con_idx + ns;
          if(MJacS != nullptr)
            MJacS[2*itrow+1] = 1.0;
        });
    }

    // Compute inequality constraints Jacobian
    if(num_cons==3 && haveIneq_ && ns_>0) 
    {
      assert(ns_+3==nnzJacS);
      // Loop over all matrix nonzeros
      RAJA::forall<ex4_raja_exec>(RAJA::RangeSegment(0, ns_+3),
        RAJA_LAMBDA(RAJA::Index_type tid)
        {
          if(tid==0)
          {
            iJacS[tid] = 0;
            jJacS[tid] = 0;
            if(MJacS != nullptr)
              MJacS[tid] = 1.0;
            assert(idx_cons[0] == ns);
          }
          else if(tid > ns)
          {
            iJacS[tid] = tid - ns;
            jJacS[tid] = tid - ns;
            if(MJacS != nullptr)
              MJacS[tid] = 1.0;
            assert(idx_cons[1] == ns + 1 && idx_cons[2] == ns + 2);
          }
          else
          {
            iJacS[tid] = 0;
            jJacS[tid] = ns + tid - 1;
            if(MJacS != nullptr)
              MJacS[tid] = 1.0;
          }
        });

    } // if(num_cons==3 && haveIneq_)
  } // if(iJacS!=NULL && jJacS!=NULL)

  //dense Jacobian w.r.t y
  if(JacD!=NULL) 
  {
    if(num_cons == ns_ && ns_ > static_cast<int>(idx_cons[0]))
    {
      //assert(num_cons==ns_);
      double* data = JacD[0];
      auto& rm = umpire::ResourceManager::getInstance();
      if(mem_space_ == "DEFAULT")
        memcpy(data, Md_->local_buffer(), ns_*nd_*sizeof(double));
      else
        rm.copy(data, Md_->local_buffer(), ns_*nd_*sizeof(double));
    }

    if(num_cons==3 && haveIneq_ && ns_>0)
    {
      for(int itrow=0; itrow<num_cons; itrow++)
      {
        const int con_idx = static_cast<int>(idx_cons[itrow]);
        //do an in place fill-in for the ineq Jacobian corresponding to e^T
        assert(con_idx-ns_==0 || con_idx-ns_==1 || con_idx-ns_==2);
        assert(num_cons==3);
        double* J = JacD[con_idx-ns_];
        RAJA::forall<ex4_raja_exec>(RAJA::RangeSegment(0, nd_),
          RAJA_LAMBDA(RAJA::Index_type i)
          {
            J[i] = 1.0;
          });
      }
    }
  } // if(JacD != nullptr)

  // Temporary solution: Move idx_cons array to GPU, works with UM and PINNED only
  if(mem_space_ == "DEFAULT")
  {
    delete [] idx_cons;
  }
  else
  {
    umpire::Allocator allocator = resmgr.getAllocator(mem_space_);
    allocator.deallocate(idx_cons);
  }
  return true;
}

/// Hessian evaluation
bool Ex4::eval_Hess_Lagr(const long long& n, const long long& m, 
    const double* x, bool new_x, const double& obj_factor,
    const double* lambda, bool new_lambda,
    const long long& nsparse, const long long& ndense, 
    const int& nnzHSS, int* iHSS, int* jHSS, double* MHSS, 
    double** HDD,
    int& nnzHSD, int* iHSD, int* jHSD, double* MHSD)
{
  //Note: lambda is not used since all the constraints are linear and, therefore, do 
  //not contribute to the Hessian of the Lagrangian

  assert(nnzHSS==2*ns_);
  assert(nnzHSD==0);
  assert(iHSD==NULL); assert(jHSD==NULL); assert(MHSD==NULL);

  if(iHSS!=NULL && jHSS!=NULL)
  {
    RAJA::forall<ex4_raja_exec>(RAJA::RangeSegment(0, 2*ns_),
      RAJA_LAMBDA(RAJA::Index_type i)
      {
        iHSS[i] = jHSS[i] = i;
      });
  }

  if(MHSS!=NULL)
  {
    RAJA::forall<ex4_raja_exec>(RAJA::RangeSegment(0, 2*ns_),
      RAJA_LAMBDA(RAJA::Index_type i)
      {
        MHSS[i] = obj_factor;
      });
  }

  if(HDD!=NULL)
  {
    const int nx_dense_squared = nd_*nd_;
    const double* Qv = Q_->local_buffer();
    RAJA::forall<ex4_raja_exec>(RAJA::RangeSegment(0, nx_dense_squared),
      RAJA_LAMBDA(RAJA::Index_type i)
      {
        HDD[0][i] = obj_factor*Qv[i];
      });
  }
  return true;
}

/* Implementation of the primal starting point specification */
bool Ex4::get_starting_point(const long long& global_n, double* x0)
{
  assert(global_n==2*ns_+nd_); 
  RAJA::forall<ex4_raja_exec>(RAJA::RangeSegment(0, global_n),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      x0[i] = 1.;
    });
  return true;
}

bool Ex4::get_starting_point(const long long& n, const long long& m,
    double* x0,
    bool& duals_avail,
    double* z_bndL0, double* z_bndU0,
    double* lambda0)
{
  if(sol_x_ && sol_zl_ && sol_zu_ && sol_lambda_)
  {
    duals_avail = true;

    if(mem_space_ == "DEFAULT")
    {
      memcpy(x0, sol_x_, n);
      memcpy(z_bndL0, sol_zl_, n);
      memcpy(z_bndU0, sol_zu_, n);

      memcpy(lambda0, sol_lambda_, m);
    }
    else
    {
      auto& resmgr = umpire::ResourceManager::getInstance();
      resmgr.copy(x0, sol_x_, n);
      resmgr.copy(z_bndL0, sol_zl_, n);
      resmgr.copy(z_bndU0, sol_zu_, n);

      resmgr.copy(lambda0, sol_lambda_, m);
    }
  }
  else
  {
    duals_avail = false;
    return false;
  }
  return true;
}

/* The public methods below are not part of hiopInterface. They are a proxy
 * for user's (front end) code to set solutions from a previous solve. 
 *
 * Same behaviour can be achieved internally (in this class ) if desired by 
 * overriding @solution_callback and @get_starting_point
 */
void Ex4::set_solution_primal(const double* x_vec)
{
  int n=2*ns_+nd_;
  /// @note: The constnesss is cast away only for memcpys - still don't touch!
  auto* x = const_cast<double*>(x_vec);

  auto& resmgr = umpire::ResourceManager::getInstance();
  umpire::Allocator allocator = resmgr.getAllocator(mem_space_);

  if(NULL == sol_x_)
  {
    sol_x_ = static_cast<double*>(allocator.allocate(n * sizeof(double)));
  }
  resmgr.copy(sol_x_, x);
}

void Ex4::set_solution_duals(const double* zl_vec, const double* zu_vec, const double* lambda_vec)
{
  int m=ns_+3*haveIneq_;
  int n=2*ns_+nd_;

  /// @note: The constnesss is cast away only for memcpys - still don't touch!
  auto* zl = const_cast<double*>(zl_vec);
  auto* zu = const_cast<double*>(zu_vec);
  auto* lambda = const_cast<double*>(lambda_vec);

  auto& resmgr = umpire::ResourceManager::getInstance();
  umpire::Allocator allocator = resmgr.getAllocator(mem_space_);

  if(NULL == sol_zl_)
  {
    sol_zl_ = static_cast<double*>(allocator.allocate(n * sizeof(double)));
  }
  resmgr.copy(sol_zl_, zl);

  if(NULL == sol_zu_)
  {
    sol_zu_ = static_cast<double*>(allocator.allocate(n * sizeof(double)));
  }
  resmgr.copy(sol_zu_, zu);

  if(NULL == sol_lambda_)
  {
    sol_lambda_ = static_cast<double*>(allocator.allocate(m * sizeof(double)));
  }
  resmgr.copy(sol_lambda_, lambda);
}

/** all constraints evaluated in here */
bool Ex4OneCallCons::eval_cons(const long long& n, const long long& m, 
    const double* x, bool new_x, double* cons)
{
  assert(3*haveIneq_+ns_ == m);
  assert(2*ns_ + nd_     == n);
  const double* s = x + ns_;
  const double* y = x + 2*ns_;

  int ns = ns_; ///< Cannot capture member inside RAJA lambda
  int nd = nd_; ///< Cannot capture member inside RAJA lambda
  bool haveIneq = haveIneq_;
  
  /// @todo determine whether this outter loop should be raja lambda, or
  /// if the inner loops should each be kernels, or if a more complex
  /// kernel execution policy should be used.
  RAJA::forall<ex4_raja_exec>(RAJA::RangeSegment(0, m),
    RAJA_LAMBDA(RAJA::Index_type con_idx)
    {
      if(con_idx<ns)
      {
        //equalities
        cons[con_idx] = x[con_idx] + s[con_idx];
      }
      else if(haveIneq)
      {
        //inequalties
        assert(con_idx<ns+3);
        if(con_idx==ns)
        {
          cons[con_idx] = x[0];
          for(int i=0; i<ns; i++)
            cons[con_idx] += s[i];
          for(int i=0; i<nd; i++)
            cons[con_idx] += y[i];
        }
        else if(con_idx==ns+1)
        {
          cons[con_idx] = x[1];
          for(int i=0; i<nd; i++)
            cons[con_idx] += y[i];
        }
        else if(con_idx==ns+2)
        {
          cons[con_idx] = x[2];
          for(int i=0; i<nd; i++)
            cons[con_idx] += y[i];
        }
        else
        {
          assert(false);
        }
      }
    });

  //we know that equalities are the first ns_ constraints so this should work
  Md_->timesVec(1.0, cons, 1.0, y);
  return true;
}

/**
 * @brief Evaluate Jacobian for equality and inequality constraints
 * 
 * @param[in] n - number of variables
 * @param[in] m - number of equality and inequality constraints
 * @param[in] x - solution vector (optimization variables)
 * @param[in] new_x - if variable is updated (?) 
 * @param[in] nsparse - number of sparse variables 
 * @param[in] ndense  - number of dense variables 
 * @param[in] nnzJacS - number of nonzeros in sparse Jacobian block
 * @param[out] - sparse matrix row indices
 * @param[out] - sparse matrix column indices
 * @param[out] - sparse matrix values
 * @param[out] - array to dense matrix row pointers
 * 
 * This method runs on GPU.
 * 
 */
bool Ex4OneCallCons::eval_Jac_cons(const long long& n, const long long& m, 
    const double* x, bool new_x,
    const long long& nsparse, const long long& ndense, 
    const int& nnzJacS, int* iJacS, int* jJacS, double* MJacS, 
    double** JacD)
{
  assert(m==ns_+3*haveIneq_);

  int ns = ns_; ///< Cannot capture member inside RAJA lambda
  
  if(iJacS!=NULL && jJacS!=NULL)
  {
    // Compute equality constraints Jacobian
    RAJA::forall<ex4_raja_exec>(RAJA::RangeSegment(0, ns_),
      RAJA_LAMBDA(RAJA::Index_type itrow)
      {
        //sparse Jacobian eq w.r.t. x and s
        //x
        iJacS[2*itrow] = itrow;
        jJacS[2*itrow] = itrow;
        if(MJacS != nullptr)
          MJacS[2*itrow] = 1.0;

        //s
        iJacS[2*itrow+1] = itrow;
        jJacS[2*itrow+1] = itrow+ns;
        if(MJacS != nullptr)
          MJacS[2*itrow+1] = 1.0;
      });

    // Compute inequality constraints Jacobian
    if(haveIneq_ && ns_>0) 
    {
      // Loop over all matrix nonzeros
      RAJA::forall<ex4_raja_exec>(RAJA::RangeSegment(0, ns_+3),
        RAJA_LAMBDA(RAJA::Index_type tid)
        {
          const int offset = 2*ns;
          if(tid==0)
          {
            iJacS[tid+offset] = ns;
            jJacS[tid+offset] = 0;
            if(MJacS != nullptr)
              MJacS[tid+offset] = 1.0;
          }
          else if(tid>ns)
          {
            iJacS[tid+offset] = tid;
            jJacS[tid+offset] = tid-ns;
            if(MJacS != nullptr)
              MJacS[tid+offset] = 1.0;
          }
          else
          {
            iJacS[tid+offset] = ns;
            jJacS[tid+offset] = ns+tid-1;
            if(MJacS != nullptr)
              MJacS[tid+offset] = 1.0;
          }
        });
    } // if(haveIneq_)
  } // if(iJacS!=NULL && jJacS!=NULL)

  //dense Jacobian w.r.t y
  if(JacD!=NULL)
  {
    //just copy the dense Jacobian corresponding to equalities
    auto& rm = umpire::ResourceManager::getInstance();
    if(mem_space_ == "DEFAULT")
      memcpy(JacD[0], Md_->local_buffer(), ns_*nd_*sizeof(double));
    else
      rm.copy(JacD[0], Md_->local_buffer(), ns_*nd_*sizeof(double));

    if(haveIneq_)
    {
      assert(ns_+3 == m);
      //do an in place fill-in for the ineq Jacobian corresponding to e^T
      double* J = JacD[ns_];
      RAJA::forall<ex4_raja_exec>(RAJA::RangeSegment(0, 3*nd_),
        RAJA_LAMBDA(RAJA::Index_type i)
        {
          J[i] = 1.0;
        });
    }
  }
  return true;
}
