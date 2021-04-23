#include <cmath>
#include <cstring> //for memcpy
#include <cstdio>
#include "hiopInterface.hpp"
#include <cassert>
#include <chrono>

#ifndef HIOP_EXAMPLE_EX9
#define  HIOP_EXAMPLE_EX9

#ifdef HIOP_USE_MPI
#include "mpi.h"
#else
#define MPI_Comm int
#define MPI_COMM_WORLD 0
#endif


/* This is the full problem defined directly in hiopSparse to test the result of hiopAlgPriDecomp
 * Base case from Ex6. 
 *  min   sum 1/4* { (x_{i}-1)^4 : i=1,...,n} + sum_{i=1^S} 1/S *  min_y 0.5 || y - x ||^2
 *  s.t.
 *            4*x_1 + 2*x_2                     == 10
 *        5<= 2*x_1         + x_3
 *        1<= 2*x_1                 + 0.5*x_4   <= 2*n, for i=4,...,n
 *        x_1 free
 *        0.0 <= x_2
 *        1.5 <= x_3 <= 10
 *        x_i >=0.5, i=4,...,n

 * For each i=1...S, an independent y^i, n_y=n_x
 *        (1-y^i_1 + \xi^i_1)^2 + \sum_{k=2}^{n_S} (y^i_k+\xi^i_k)^2 
 *          + \sum_{k=n_S+1}^{n_y} y_k^{i,2} >= 1   //last one in the constraint implementation 
 * 
 *        y^i_k - y^i_{k-1} >=0, k=2, ..., n_y
 *
 *        y^i_1 >=0
 */


using namespace hiop;
class Ex9 : public hiop::hiopInterfaceSparse
{
public:
  Ex9(int nx,int S)
  : nx_(nx),n_vars(nx+S*nx), n_cons{2+S*nx},S_(S) //total number of variables should be nx+S*nx
  {
    assert(nx>=3);
    if(nx>3)
      n_cons += nx-3;
    nS_ = int(nx_/2);
    xi_ = new double[S_*nS_];
    x0_ = new double[nx_];
    for(int i=0;i<nS_*S_;i++) xi_[i] = 1.0;
    for(int i=0;i<nx_;i++) x0_[i] = 1.0;
    //for(int i=0;i<S_;i++){
    //  xi_[i*nS_+nS_-1] = -1.0;
    //}
  }

  Ex9(int nx,int S,int nS)
  : nx_(nx),n_vars(nx+S*nx), n_cons{2+S*nx},S_(S),nS_(nS) //total number of variables should be nx+S*nx
  {
    assert(nx>=3);
    if(nx>3)
      n_cons += nx-3;
    xi_ = new double[nS_*S_];
    x0_ = new double[nx_];
    for(int i=0;i<nS_*S_;i++) { 
      xi_[i] = 1.0;
    }
    //for(int i=0;i<S_;i++) {
    //  xi_[i*nS_+nS_-1] = 1.0;
    //}
    for(int i=0;i<nx_;i++) x0_[i] = 1.0;
  }
  bool get_prob_sizes(long long& n, long long& m)
  { n=n_vars; m=n_cons; return true; }


  bool get_vars_info(const long long& n, double *xlow, double* xupp, NonlinearityType* type)
  {
    assert(n==n_vars);
    for(long long i=0; i<nx_; i++) {
      if(i==0) { xlow[i]=-1e20; xupp[i]=1e20; type[i]=hiopNonlinear; continue; }
      if(i==1) { xlow[i]= 0.0;  xupp[i]=1e20; type[i]=hiopNonlinear; continue; }
      if(i==2) { xlow[i]= 1.5;  xupp[i]=10.0; type[i]=hiopNonlinear; continue; }
      //this is for x_4, x_5, ... , x_n (i>=3), which are bounded only from below
      xlow[i]= 0.5; xupp[i]=1e20; type[i]=hiopNonlinear;
    }
    for(long long i=0; i<S_; i++) {
      for(long long j=0; j<nx_; j++) {
        if(j==0){xlow[nx_+i*nx_] = 0.; xupp[nx_+i*nx_] = 1e20;type[i]=hiopNonlinear;continue;}
        xlow[nx_+i*nx_+j] = -1e+20; xupp[nx_+i*nx_+j] = +1e+20; type[i]=hiopNonlinear; 
      }
    }
    return true;
  }

  ~Ex9()
  { 
    delete[] xi_;
    delete[] x0_;
  }

  bool get_cons_info(const long long& m, double* clow, double* cupp, NonlinearityType* type)
  {
    assert(m==n_cons);
    long long conidx{0};
    clow[conidx]= 10.0;    cupp[conidx]= 10.0;      type[conidx++]=hiopInterfaceBase::hiopLinear;
    clow[conidx]= 5.0;     cupp[conidx]= 1e20;      type[conidx++]=hiopInterfaceBase::hiopLinear;
    for(long long i=3; i<nx_; i++) {
      clow[conidx] = 1.0;   cupp[conidx]= 2*nx_;  type[conidx++]=hiopInterfaceBase::hiopLinear;
    }
    if(nx_>3){assert(conidx==2+nx_-3);}//nx_-1
    for(long long i=0;i<S_;i++) { 
      for(long long j=0;j<nx_-1;j++) {
        clow[conidx+nx_*i+j] = 0.;
        cupp[conidx+nx_*i+j] = 1e20;
      }
      clow[conidx+nx_*i+nx_-1] = 1.;
      cupp[conidx+nx_*i+nx_-1] = 1e20;
    }
    return true;
  }
 
  bool get_sparse_blocks_info(int& n,
                              int& nnz_sparse_Jaceq, 
                              int& nnz_sparse_Jacineq,
                              int& nnz_sparse_Hess_Lagr)
  {
    n = n_vars;
    nnz_sparse_Jaceq = 2;
    nnz_sparse_Jacineq = 2+2*(nx_-3)+S_*(nx_+ 2*(nx_-1)) ;
    //printf("nnz_sparse_Jacineq %d\n",nnz_sparse_Jacineq);
    //nnz_sparse_Hess_Lagr = nx_+S_*nx_*2+S_*nx_; //this variable should always be <= n_vars?
    nnz_sparse_Hess_Lagr = nx_+S_*nx_+S_*nx_; //this variable should always be <= n_vars?
    return true;
  }

  bool eval_f(const long long& n, const double* x, bool new_x, double& obj_value)
  {
    assert(n==n_vars);
    obj_value=0.;
    for(int i=0;i<nx_;i++) {
      //obj_value += 0.25*std::pow(x[i]-1.,4);
      obj_value += 0.25*std::pow(x[i]-1.,4);
    }
    for(int i=0;i<S_;i++) {
      for(int j=0;j<nx_;j++) {
        obj_value += (x[nx_*(i+1)+j]-x[j])*(x[nx_*(i+1)+j]-x[j])*0.5/double(S_);
      }
    }
    return true;
  }

  bool eval_grad_f(const long long& n, const double* x, bool new_x, double* gradf)
  {
    assert(n==n_vars);
    for(int i=0;i<n_vars;i++) gradf[i] = 0.;
    //for(int i=0;i<nx_;i++) gradf[i] = std::pow(x[i]-1.,3);
    for(int i=0;i<nx_;i++) gradf[i] = 1.0*std::pow(x[i]-1.,3);

    for(int i=0;i<S_;i++) {
      for(int j=0;j<nx_;j++) {
        gradf[nx_*(i+1)+j] += (x[nx_*(i+1)+j]-x[j])/double(S_);
        gradf[j] += (x[j]-x[nx_*(i+1)+j])/double(S_);
      }
    }
    //std::cout<<"gradf15 "<<gradf[15]<<std::endl;
    return true;
  }

  /* Four constraints no matter how large n is */
  // This constraint attempts to use existing functions
  bool eval_cons(const long long& n, 
                 const long long& m,
                 const double* x, 
                 bool new_x, 
                 double* cons)
  {

    assert(n==n_vars); assert(m==n_cons);
    //for the base problem n_cons==2+n-3;

    //local contributions to the constraints in cons are reset
    for(auto j=0;j<m; j++) cons[j]=0.;

    long long conidx{0};
    //compute the constraint one by one.
    // --- constraint 1 body --->  4*x_1 + 2*x_2 == 10
    cons[conidx++] += 4*x[0] + 2*x[1];

    // --- constraint 2 body ---> 2*x_1 + x_3
    cons[conidx++] += 2*x[0] + 1*x[2];

    // --- constraint 3 body --->   2*x_1 + 0.5*x_i, for i>=4
    for(auto i=3; i<nx_; i++) {
      cons[conidx++] += 2*x[0] + 0.5*x[i];
    }

    for(int i=0;i<S_;i++) {
      for(int j=0;j<nx_-1;j++) {
        cons[conidx+i*nx_+j] = x[nx_+i*nx_+j+1]-x[nx_+i*nx_+j];
      }
      cons[conidx+i*nx_+nx_-1] = (1-x[nx_+i*nx_]+xi_[i*nS_])*(1-x[nx_+i*nx_]+xi_[i*nS_]);
      for(int j=1;j<nS_;j++) {
        cons[conidx+i*nx_+nx_-1] += (x[nx_+i*nx_+j] + xi_[i*nS_+j])*(x[nx_+i*nx_+j] + xi_[i*nS_+j]);
      }
      for(int j=nS_;j<nx_;j++) {
        cons[conidx+i*nx_+nx_-1] += x[nx_+i*nx_+j]*x[nx_+i*nx_+j];
      }
    }
    
    return true;
  }
  
  bool eval_cons(const long long& n, 
                 const long long& m,
                 const long long& num_cons, 
                 const long long* idx_cons,
                 const double* x, 
                 bool new_x, 
                 double* cons)
  {
    return false;
  }
  bool eval_Jac_cons(const long long& n, 
                     const long long& m,
                     const long long& num_cons, 
                     const long long* idx_cons,
                     const double* x, 
                     bool new_x,
                     const int& nnzJacS, 
                     int* iJacS, 
                     int* jJacS, 
                     double* MJacS)
  {
    return false;
  }

  bool eval_Jac_cons(const long long& n, 
                     const long long& m,
                     const double* x, 
                     bool new_x,
                     const int& nnzJacS, 
                     int* iJacS, 
                     int* jJacS, 
                     double* MJacS)
  {
    assert(n==n_vars); assert(m==n_cons);
    assert(n>=3);
    //2*(n-1) for basecase

    assert(nnzJacS == 2*(nx_-1)+S_*(2*(nx_-1)+nx_));

    int nnzit{0};
    long long conidx{0};

    if(iJacS!=NULL && jJacS!=NULL) {
      // --- constraint 1 body --->  4*x_1 + 2*x_2 == 10
      iJacS[nnzit] = conidx;   jJacS[nnzit++] = 0;
      iJacS[nnzit] = conidx;   jJacS[nnzit++] = 1;
      conidx++;

      // --- constraint 2 body ---> 2*x_1 + x_3
      iJacS[nnzit] = conidx;   jJacS[nnzit++] = 0;
      iJacS[nnzit] = conidx;   jJacS[nnzit++] = 2;
      conidx++;

      // --- constraint 3 body --->   2*x_1 + 0.5*x_i, for i>=4
      for(auto i=3; i<nx_; i++){
          iJacS[nnzit] = conidx;   jJacS[nnzit++] = 0;
          iJacS[nnzit] = conidx;   jJacS[nnzit++] = i;
          conidx++;
      }
      for(auto i=0;i<S_;i++) {
        for(auto j=0;j<nx_-1;j++) {
          iJacS[nnzit] = conidx+i*nx_+j; jJacS[nnzit] = nx_+i*nx_+j; 
          nnzit++; 
          iJacS[nnzit] = conidx+i*nx_+j; jJacS[nnzit] = nx_+i*nx_+j+1; 
          nnzit++;
        }
        for(auto j=0;j<nx_;j++) {
          iJacS[nnzit] = conidx+i*nx_+nx_-1; jJacS[nnzit] = nx_+i*nx_+j;
          nnzit++;
        }
      } 
      assert(nnzit == nnzJacS);
    }

    //values for sparse Jacobian if requested by the solver
    nnzit = 0;
    if(MJacS!=NULL) {
      // --- constraint 1 body --->  4*x_1 + 2*x_2 == 10
      MJacS[nnzit++] = 4;
      MJacS[nnzit++] = 2;

      // --- constraint 2 body ---> 2*x_1 + x_3
      MJacS[nnzit++] = 2;
      MJacS[nnzit++] = 1;

      // --- constraint 3 body --->   2*x_1 + 0.5*x_4
      for(auto i=3; i<nx_; i++){
        MJacS[nnzit++] = 2;
        MJacS[nnzit++] = 0.5;
      }
      for(auto i=0;i<S_;i++) {
        for(auto j=0;j<nx_-1;j++) {
          MJacS[nnzit] = -1.; 
          nnzit++; 
          MJacS[nnzit] = 1.; 
          nnzit++;
        }
        MJacS[nnzit] = -2*(1-x[nx_+i*nx_]+xi_[i*nS_]);
        nnzit++;
        for(auto j=1;j<nS_;j++) {
          MJacS[nnzit] = 2*(x[nx_+i*nx_+j]+xi_[i*nS_+j]);
          nnzit++;
        }
        for(auto j=nS_;j<nx_;j++) {
          MJacS[nnzit] = 2*x[nx_+i*nx_+j];
          nnzit++;
        }
      }
      assert(nnzit == nnzJacS);
    }
    return true;
  }

  /*bool eval_Hess_Lagr(const long long& n, const long long& m,
                           const double* x, bool new_x, const double& obj_factor,
                           const double* lambda, bool new_lambda,
                           const int& nnzHSS, int* iHSS, int* jHSS, double* MHSS)
  {return false;}*/
  bool eval_Hess_Lagr(const long long& n, 
                      const long long& m,
                      const double* x, 
                      bool new_x, 
                      const double& obj_factor,
                      const double* lambda, 
                      bool new_lambda,
                      const int& nnzHSS, 
                      int* iHSS, 
                      int* jHSS, 
                      double* MHSS)
  {
    //assert(nnzHSS == nx_+S_*nx_*2+S_*nx_);
    assert(nnzHSS == nx_+S_*nx_+S_*nx_);
    int nnzit = 0;
    if(iHSS!=NULL && jHSS!=NULL) {
      for(int i=0; i<nx_; i++) { 	
        iHSS[nnzit] = jHSS[nnzit] = i;
        nnzit++;
        //for(int j=0;j<S_;j++){
        //  iHSS[nnzit] = i; jHSS[nnzit] = nx_+j*nx_+i;
        //  nnzit++;
        //}    
      }
     // r_i(x;\xi^i) = 1/S *  min_y 0.5 || y - x ||^2 such that 
      for(int i=0;i<S_;i++) {
        for(int j=0;j<nx_;j++) {
          iHSS[nnzit] = nx_+ i*nx_+j; jHSS[nnzit] = j; nnzit++;
          iHSS[nnzit] = nx_+ i*nx_+j; jHSS[nnzit] = nx_+i*nx_+j; nnzit++;
        }
      }      
      assert(nnzHSS == nnzit);
    }
    nnzit = 0;
    
    if(MHSS!=NULL) {
      //std::cout<<"MHSS "<<-obj_factor/double(S_)<<std::endl;
      for(int i=0; i<nx_; i++) { 		
        MHSS[nnzit] = obj_factor * 3*pow(x[i]-1., 2);
        for(int j=0;j<S_;j++){
          MHSS[nnzit] += obj_factor/double(S_);
        }
        nnzit++;
        /*for(int j=0;j<S_;j++){
          MHSS[nnzit] = -obj_factor/double(S_);
          nnzit++;
        }*/
      }
      // r_i(x;\xi^i) = 1/S *  min_y 0.5 || y - x ||^2 such that 
      for(int i=0;i<S_;i++) {
        for(int j=0;j<nx_;j++) {
          MHSS[nnzit] = -obj_factor/double(S_);
          nnzit++;
          MHSS[nnzit] = obj_factor/double(S_)+lambda[(nx_-1)+nx_*i+nx_-1]*2.;
          nnzit++;
        }
      }
      assert(nnzHSS == nnzit);
      //for(int i=0;i<nnzHSS;i++) {
      //  std::cout<<"MHSS "<<i<<" "<<MHSS[i];
      //}
      //std::cout<<std::endl;
    }

    return true;
  }


  //bool get_MPI_comm(MPI_Comm& comm_out) { comm_out=MPI_COMM_SELF; return true;};

  bool get_starting_point(const long long& n, double* x0)
  {
    assert(n==n_vars);
    for(auto i=0; i<n; i++) {
      x0[i]=1.0;
    }
    for(auto i=0;i<nx_;i++) {
      x0[i]=x0_[i]; 
    }
    return true;
  }

  void set_starting_point(const double* x0) 
  {
    for(auto i=0;i<nx_;i++) {
      x0_[i]=x0[i]; 
    }
  }

private:
  int n_vars, n_cons;
  int nx_, S_,nS_;
  double* xi_;// of size S_*nS_
  double* x0_;// of size S_*nS_
};

#endif
