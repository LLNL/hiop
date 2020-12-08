#ifndef EX9_RECOURSE
#define EX9_RECOURSE


/** This class provide an example of what a user of hiop::hiopInterfacePriDecProblem 
 * should implement in order to provide the recourse problem to 
 * hiop::hiopAlgPrimalDecomposition solver
 * 
 * For a given vector x\in R^n and \xi \in R^{n_S}, this example class implements
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
 */

class PriDecRecourseProblemEx9 : public hiop::hiopInterfaceMDS
{
public:
  PriDecRecourseProblemEx9(int n, int ny)
  {
  }

  virtual ~PriDecRecourseProblemEx9();

  /// Set the basecase solution `x`
  void set_x(const double* x);

  /// Set the "sample" vector \xi
  void set_center(const double *xi);

  //
  // this are pure virtual in hiop::hiopInterfaceMDS and need to be implemented
  //
  bool get_prob_sizes(long long& n, long long& m)
  { 
    assert(false && "not implemented");
    return true; 
  }

  bool get_vars_info(const long long& n, double *xlow, double* xupp, NonlinearityType* type)
  {
    assert(false && "not implemented");
    return true;
  }

  bool get_cons_info(const long long& m, double* clow, double* cupp, NonlinearityType* type)
  {
    assert(false && "not implemented");
    return true;
  }

  bool get_sparse_dense_blocks_info(int& nx_sparse, int& nx_dense,
				    int& nnz_sparse_Jace, int& nnz_sparse_Jaci,
				    int& nnz_sparse_Hess_Lagr_SS, int& nnz_sparse_Hess_Lagr_SD)
  {
    assert(false && "not implemented");
    return true;
  }

  bool eval_f(const long long& n, const double* x, bool new_x, double& obj_value)
  {
    assert(false && "not implemented");
    return true;
  }

  virtual bool eval_cons(const long long& n, const long long& m, 
			 const long long& num_cons, const long long* idx_cons,  
			 const double* x, bool new_x, double* cons)
  {
    assert(false && "not implemented");
    return true;
  }
  
  //sum 0.5 {x_i*(x_{i}-1) : i=1,...,ns} + 0.5 y'*Qd*y + 0.5 s^T s
  bool eval_grad_f(const long long& n, const double* x, bool new_x, double* gradf)
  {
    assert(false && "not implemented");
    return true;
  }
 
  virtual bool
  eval_Jac_cons(const long long& n, const long long& m, 
		const long long& num_cons, const long long* idx_cons,
		const double* x, bool new_x,
		const long long& nsparse, const long long& ndense, 
		const int& nnzJacS, int* iJacS, int* jJacS, double* MJacS, 
		double* JacD)
  {
    //indexes for sparse part
    if(iJacS!=NULL && jJacS!=NULL) {
      
      assert(false && "not implemented");
    } 
    //values for sparse Jacobian if requested by the solver
    if(MJacS!=NULL) {
      assert(false && "not implemented");
    }
    
    //dense Jacobian w.r.t y
    if(JacD!=NULL) {
      assert(false && "not implemented");
    }

    return true;
  }
 
  bool eval_Hess_Lagr(const long long& n, const long long& m, 
                      const double* x, bool new_x, const double& obj_factor,
                      const double* lambda, bool new_lambda,
                      const long long& nsparse, const long long& ndense, 
                      const int& nnzHSS, int* iHSS, int* jHSS, double* MHSS, 
                      double* HDD,
                      int& nnzHSD, int* iHSD, int* jHSD, double* MHSD)
  {
    if(iHSS!=NULL && jHSS!=NULL) {
      assert(false && "not implemented");
    }

    if(MHSS!=NULL) {
      assert(false && "not implemented");
    }

    if(HDD!=NULL) {
      assert(false && "not implemented");
    }
    return true;
  }

  /* Implementation of the primal starting point specification */
  bool get_starting_point(const long long& global_n, double* x0)
  {    
    //for(int i=0; i<global_n; i++) x0[i]=1.;
    assert(false && "not implemented");
    return true;
  }
  bool get_starting_point(const long long& n, const long long& m,
				  double* x0,
				  bool& duals_avail,
				  double* z_bndL0, double* z_bndU0,
				  double* lambda0)
  {
    assert(false && "not implemented");
    return true;
  }
  
  /**
   * Returns COMM_SELF communicator since this example is only intended to run 
   * on one MPI process 
   */
  bool get_MPI_comm(MPI_Comm& comm_out) {
    comm_out=MPI_COMM_SELF;
    return true;
  }

protected:
  // bla bla  
};

#endif
