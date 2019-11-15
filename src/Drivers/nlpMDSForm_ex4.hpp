#ifndef HIOP_EXAMPLE_EX4
#define HIOP_EXAMPLE_EX4

#include "hiopInterface.hpp"

//this include is not needed in general; we use hiopMatrixDense in this particular example
#include "hiopMatrix.hpp" 

#ifdef HIOP_USE_MPI
#include "mpi.h"
#else
#define MPI_COMM_WORLD 0
#define MPI_Comm int
#endif

#include <cassert>
#include <cstring> //for memcpy
#include <cstdio>
#include <cmath>

/* Problem test for the linear algebra of Mixed Dense-Sparse NLPs
 *  min   sum 0.5 {x_i*(x_{i}-1) : i=1,...,ns} + 0.5 y'*Qd*y + 0.5 s^T s
 *  s.t.  x+s - Md y = 0, i=1,...,ns
 *        [-2  ]    [ x_1 + e^T s]   [e^T]      [ 2 ]
 *        [-inf] <= [ x_2        ] + [e^T] y <= [ 2 ]
 *        [-2  ]    [ x_3        ]   [e^T]      [inf]
 *        x <= 3
 *        s>=0
 *        -4 <=y_1 <=4, the rest of y are free
 *        
 * The vector 'y' is of dimension nd = ns/4
 * Dense matrices Qd and Md are such that
 * Qd  = two on the diagonal, one on the first offdiagonals, zero elsewhere
 * Md  = one everywhere
 * e   = vector of all ones
 *
 * Coding of the problem in MDS HiOp input: order of variables need to be [x,s,y] 
 * since [x,s] are the so-called sparse variables and y are the dense variables
 */
class Ex4 : public hiop::hiopInterfaceMDS
{
public: 
  Ex4(int ns_)
    : ns(ns_)
  {
    if(ns<=0) {
      ns = 4;
    } else {
      if(4*(ns/4) != ns) {
	ns = 4*((4+ns)/4);
	printf("[warning] number (%d) of sparse vars is not a multiple ->was altered to %d\n", 
	       ns_, ns); 
      }
    }
    
    const int n = ns/4;
    Q  = new hiop::hiopMatrixDense(n,n);
    Q->setToZero();
    Q->addDiagonal(2.);
    double** Qa = Q->get_M();
    for(int i=1; i<n-1; i++) {
      Qa[i][i+1] = 1.;
      Qa[i+1][i] = 1.;
    }

    Md = new hiop::hiopMatrixDense(ns,ns/4);
    Md->setToConstant(1.0);

    _buf_y = new double[ns/4];
  }

  virtual ~Ex4()
  {
    delete[] _buf_y;
    delete Md;
    delete Q;
  }
  
  bool get_prob_sizes(long long& n, long long& m)
  { 
    n=2*ns+ns/4;
    m=ns+3; 
    return true; 
  }

  bool get_vars_info(const long long& n, double *xlow, double* xupp, NonlinearityType* type)
  {
    assert(n>=4 && "number of variables should be greater than 4 for this example");
    assert(n==2*ns+ns/4);

    //x
    for(int i=0; i<ns; ++i) xlow[i] = -1e+20;
    //s
    for(int i=ns; i<2*ns; ++i) xlow[i] = 0.;
    //y 
    xlow[2*ns] = -4.;
    for(int i=2*ns+1; i<n; ++i) xlow[i] = -1e+20;
    
    //x
    for(int i=0; i<ns; ++i) xupp[i] = 3.;
    //s
    for(int i=ns; i<2*ns; ++i) xupp[i] = +1e+20;
    //y
    xupp[2*ns] = 4.;
    for(int i=2*ns+1; i<n; ++i) xupp[i] = +1e+20;

    for(int i=0; i<n; ++i) type[i]=hiopNonlinear;
    return true;
  }

  bool get_cons_info(const long long& m, double* clow, double* cupp, NonlinearityType* type)
  {
    assert(m==ns+3);
    int i;
    //x+s - Md y = 0, i=1,...,ns
    for(i=0; i<ns; i++) clow[i] = cupp[i] = 0.;

    // [-2  ]    [ x_1 + e^T s]   [e^T]      [ 2 ]
    clow[i] = -2; cupp[i++] = 2.;
    // [-inf] <= [ x_2        ] + [e^T] y <= [ 2 ]
    clow[i] = -1e+20; cupp[i++] = 2.;
    // [-2  ]    [ x_3        ]   [e^T]      [inf]
    clow[i] = -2; cupp[i++] = 1e+20;
    assert(i==m);

    for(i=0; i<m; ++i) type[i]=hiopNonlinear;
    return true;
  }

  bool get_sparse_dense_blocks_info(int& nx_sparse, int& nx_dense,
				    int& nnz_sparse_Jace, int& nnz_sparse_Jaci,
				    int& nnz_sparse_Hess_Lagr_SS, int& nnz_sparse_Hess_Lagr_SD)
  {
    nx_sparse = 2*ns;
    nx_dense = ns/4;
    nnz_sparse_Jace = 2*ns;
    nnz_sparse_Jaci = 1+ns+1+1;
    nnz_sparse_Hess_Lagr_SS = 2*ns;
    nnz_sparse_Hess_Lagr_SD = 0.;
    return true;
  }

  bool eval_f(const long long& n, const double* x, bool new_x, double& obj_value)
  {
    assert(ns>=4); assert(Q->n()==ns/4); assert(Q->m()==ns/4);
    obj_value=x[0]*(x[0]-1.);
    //sum 0.5 {x_i*(x_{i}-1) : i=1,...,ns} + 0.5 y'*Qd*y + 0.5 s^T s
    for(int i=1; i<ns; i++) obj_value += x[i]*(x[i]-1.);
    obj_value *= 0.5;

    double term2=0.;
    const double* y = x+2*ns;
    Q->timesVec(0.0, _buf_y, 1., y);
    for(int i=0; i<ns/4; i++) term2 += _buf_y[i] * y[i];
    obj_value += 0.5*term2;
    
    const double* s=x+ns;
    double term3=s[0]*s[0];
    for(int i=1; i<ns; i++) term3 += s[i]*s[i];
    obj_value += 0.5*term3;

    return true;
  }

  bool eval_cons(const long long& n, const long long& m, 
			 const long long& num_cons, const long long* idx_cons,  
			 const double* x, bool new_x, double* cons)
  {
    const double* s = x+ns;
    const double* y = x+2*ns;

    assert(num_cons==ns || num_cons==3);

    bool isEq=false;
    for(int irow=0; irow<num_cons; irow++) {
      const int con_idx = idx_cons[irow];
      if(con_idx<ns) {
	//equalities: x+s - Md y = 0
	cons[con_idx] = x[con_idx] + s[con_idx];
	isEq=true;
      } else {
	assert(con_idx<ns+3);
	//inequality
	if(con_idx-ns==0) {
	  cons[con_idx] = x[0];
	  for(int i=0; i<ns; i++)   cons[con_idx] += s[i];
	  for(int i=0; i<ns/4; i++) cons[con_idx] += y[i];

	} else if(con_idx-ns==1) {
	  cons[con_idx] = x[1];
	  for(int i=0; i<ns/4; i++) cons[con_idx] += y[i];
	} else if(con_idx-ns==2) {
	  cons[con_idx] = x[2];
	  for(int i=0; i<ns/4; i++) cons[con_idx] += y[i];
	} else { assert(false); }
      }  
    }
    if(isEq) {
      Md->timesVec(1.0, cons, -1., y);
    }
    return true;
  }
  
  //sum 0.5 {x_i*(x_{i}-1) : i=1,...,ns} + 0.5 y'*Qd*y + 0.5 s^T s
  bool eval_grad_f(const long long& n, const double* x, bool new_x, double* gradf)
  {
    assert(ns>=4); assert(Q->n()==ns/4); assert(Q->m()==ns/4);
    //x_i - 0.5 
    for(int i=0; i<ns; i++) 
      gradf[i] = x[i]-0.5;


    //Qd*y
    const double* y = x+2*ns;
    double* gradf_y = gradf+2*ns;
    Q->timesVec(0.0, gradf_y, 1., y);

    //s
    const double* s=x+ns;
    double* gradf_s = gradf+ns;
    for(int i=0; i<ns; i++) gradf_s[i] = s[i];

    return true;
  }
 
  bool eval_Jac_cons(const long long& n, const long long& m, 
		     const long long& num_cons, const long long* idx_cons,
		     const double* x, bool new_x,
		     const long long& nsparse, const long long& ndense, 
		     const int& nnzJacS, int* iJacS, int* jJacS, double* MJacS, 
		     double** JacD)
  {
    const double* s = x+ns;
    const double* y = x+2*ns;

    assert(num_cons==ns || num_cons==3);

    if(iJacS!=NULL && jJacS!=NULL) {
      int nnzit=0;
      for(int itrow=0; itrow<num_cons; itrow++) {
	const int con_idx = idx_cons[itrow];
	if(con_idx<ns) {
	  //sparse Jacobian eq w.r.t. x and s
	  //x
	  iJacS[nnzit] = con_idx;
	  jJacS[nnzit] = con_idx;
	  nnzit++;

	  //s
	  iJacS[nnzit] = con_idx;
	  jJacS[nnzit] = con_idx+ns;
	  nnzit++;

	} else {
	  //sparse Jacobian ineq w.r.t x and s
	  if(con_idx-ns==0) {
	    //w.r.t x_1
	    iJacS[nnzit] = 0;
	    jJacS[nnzit] = 0;
	    nnzit++;
	    //w.r.t s
	    for(int i=0; i<ns; i++) {
	      iJacS[nnzit] = 0;
	      jJacS[nnzit] = ns+i;
	      nnzit++;
	    }
	  } else {
	    assert(con_idx-ns==1 || con_idx-ns==2);
	    //w.r.t x_2 or x_3
	    iJacS[nnzit] = con_idx-ns;
	    jJacS[nnzit] = con_idx-ns;
	    nnzit++;
	  }
	}
      }
      assert(nnzit==nnzJacS);
    } 
    //values for sparse Jacobian if requested by the solver
    if(MJacS!=NULL) {
     int nnzit=0;
     for(int itrow=0; itrow<num_cons; itrow++) {
       const int con_idx = idx_cons[itrow];
       if(con_idx<ns) {
	 //sparse Jacobian EQ w.r.t. x and s
	 //x
	 MJacS[nnzit] = 1.;
	 nnzit++;
	 
	 //s
	 MJacS[nnzit] = 1.;
	 nnzit++;
	 
       } else {
	 //sparse Jacobian INEQ w.r.t x and s
	 if(con_idx-ns==0) {
	   //w.r.t x_1
	   MJacS[nnzit] = 1.;
	   nnzit++;
	   //w.r.t s
	   for(int i=0; i<ns; i++) {
	     MJacS[nnzit] = 1.;
	     nnzit++;
	   }
	 } else {
	   assert(con_idx-ns==1 || con_idx-ns==2);
	   //w.r.t x_2 or x_3
	   MJacS[nnzit] = 1.;
	   nnzit++;
	 }
       }
     }
     assert(nnzit==nnzJacS);
    }
    
    //dense Jacobian w.r.t y
    if(JacD!=NULL) {
      bool isEq=false;
      for(int itrow=0; itrow<num_cons; itrow++) {
	const int con_idx = idx_cons[itrow];
	if(con_idx<ns) {
	  isEq=true;
	  assert(num_cons==ns);
	  continue;
	} else {
	  //do an in place fill-in for the ineq Jacobian corresponding to e^T
	  assert(con_idx-ns==0 || con_idx-ns==1 || con_idx-ns==2);
	  assert(num_cons==3);
	  for(int i=0; i<ns/4; i++) {
	    JacD[con_idx-ns][i] = 1.;
	  }
	}
      }
      if(isEq) {
	memcpy(JacD[0], Md->local_buffer(), ns*(ns/4)*sizeof(double));
      }
    }

    return true;
  }
 
  bool eval_Hess_Lagr(const long long& n, const long long& m, 
			      const double* x, bool new_x, const double& obj_factor,
			      const double* lambda, bool new_lambda,
			      const long long& nsparse, const long long& ndense, 
			      const int& nnzHSS, int* iHSS, int* jHSS, double* MHSS, 
			      double** HDD,
			      int& nnzHSD, int* iHSD, int* jHSD, double* MHSD)
  {
    //Note: lambda is not used since all the constraints are linear and, therefore, do 
    //not contribute to the Hessian of the Lagrangian

    assert(nnzHSS==2*ns);
    assert(nnzHSD==0);
    assert(iHSD==NULL); assert(jHSD==NULL); assert(MHSD==NULL);

    if(iHSS!=NULL && jHSS!=NULL) {
      for(int i=0; i<ns; i++) {
	iHSS[i] = jHSS[i] = i;
      }
      for(int i=0; i<ns; i++) {
	const int is = i+ns;
	iHSS[is] = i;
	jHSS[is] = is;
      }
    }

    if(MHSS!=NULL) {
      for(int i=0; i<2*ns; i++) MHSS[i] = 1.;
    }

    if(HDD!=NULL) {
      memcpy(HDD[0], Q->local_buffer(), ns*ns/16*sizeof(double));
    }
    return true;
  }
  
  bool get_starting_point(const long long& global_n, double* x0)
  {
    
    assert(global_n==2*ns+ns/4); 
    for(int i=0; i<global_n; i++) x0[i]=0.;
    return true;
  }

private:
  int ns;
  hiop::hiopMatrixDense *Q, *Md;
  double* _buf_y;
};
#endif
