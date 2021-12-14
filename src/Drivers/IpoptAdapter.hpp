/* Adapter that converts  HiOP interface to IPOPT TNLP interface.
 * Can be easily used with existing IPOPT drivers (to solve problems in HiOP's input), for example
 *    Ex2 hiop_interface(7);
 *    // create a new instance of your nlp by using the adapter offered by hiOP.
 *    SmartPtr<TNLP> mynlp = new hiop2IpoptTNLP(&hiop_interface);
 *    // from now on everything is the usual IPOPT bussiness
 * 
 * An example IPOPT driver that solve HiOP NLPs using IPOPT is available upon request.
 *
 */

/*
 * Author: Cosmin G. Petra, LLNL, 2016.
 * License: internal use only
 */

#ifndef HIOP_IPOPT_ADAPTER
#define HIOP_IPOPT_ADAPTER

#include "IpTNLP.hpp"
#include "hiopInterface.hpp"

#include <cassert>
#include <cstring>
using namespace Ipopt;
using namespace hiop;

/* "Converts" HiOP interface to Ipopt TNLP interface */

class hiop2IpoptTNLP : public TNLP
{
public:
  hiop2IpoptTNLP(hiopInterfaceBase* hiopNLP_) 
    : hiopNLP(dynamic_cast<hiopInterfaceDenseConstraints*>(hiopNLP_)) {};
  virtual ~hiop2IpoptTNLP() {};

  /* Overloads from TNLP */
  /** Method to return some info about the nlp */
  virtual bool get_nlp_info(Index& n, Index& m, Index& nnz_jac_g,
                            Index& nnz_h_lag, IndexStyleEnum& index_style) 
  {
    long long nvars, ncons;
    if(false==hiopNLP->get_prob_sizes(nvars, ncons))
      return false;
    n = (int)nvars; m=(int)ncons;
    nnz_jac_g = n*m;
    nnz_h_lag=2*n-1;
    index_style = TNLP::C_STYLE;
    return true;
  }

  /** Method to return the bounds for my problem */
  virtual bool get_bounds_info(Index n, Number* x_l, Number* x_u,
                               Index m, Number* g_l, Number* g_u) 
  {
    bool bSuccess=true;
    long long nll=n, mll=m;
    hiopInterfaceBase::NonlinearityType* types=new hiopInterfaceBase::NonlinearityType[n];
    bSuccess = hiopNLP->get_vars_info(nll,x_l,x_u, types);
    delete[] types;
    
    if(bSuccess) {
      types=new hiopInterfaceBase::NonlinearityType[m];
      bSuccess = hiopNLP->get_cons_info(mll, g_l, g_u, types);
      delete[] types;
    }
    return bSuccess;
  }


  /** Method to return the starting point for the algorithm */
  virtual bool get_starting_point(Index n, bool init_x, Number* x,
                                  bool init_z, Number* z_L, Number* z_U,
                                  Index m, bool init_lambda,
                                  Number* lambda) 
  {
    assert(false==init_z);
    assert(false==init_lambda);
    long long nll=n;
    return hiopNLP->get_starting_point(nll,x);
  }


  /** Method to return the objective value */
  virtual bool eval_f(Index n, const Number* x, bool new_x, Number& obj_value) 
  {
    long long nll=n;
    return hiopNLP->eval_f(nll,x,new_x,obj_value);
  }


  /** Method to return the gradient of the objective */
  virtual bool eval_grad_f(Index n, const Number* x, bool new_x, Number* grad_f) 
  {
    long long nll=n;
    return hiopNLP->eval_grad_f(nll,x,new_x,grad_f);
  }


  /** Method to return the constraint residuals */
  virtual bool eval_g(Index n, const Number* x, bool new_x, Index m, Number* g) 
  {
    long long nll=n, mll=m;
    long long* idx_cons=new long long[m];
    for(int i=0; i<m; i++) idx_cons[i]=i;
    bool bret = hiopNLP->eval_cons(nll,mll,mll,idx_cons,x,new_x,g);
    delete[] idx_cons;
    return bret;
  }


  /** Method to return:
   *   1) The structure of the jacobian (if "values" is NULL)
   *   2) The values of the jacobian (if "values" is not NULL)
   */
  virtual bool eval_jac_g(Index n, const Number* x, bool new_x,
                          Index m, Index nele_jac, Index* iRow, Index *jCol,
                          Number* values) 
  {
    bool bret=true; long long nll=n, mll=m, onell=1;
    double* constraint=new double[n]; 
    long long nz=0;
    for(long long i=0; i<m && bret; i++) {

      if(values) {
	bret=hiopNLP->eval_Jac_cons(nll, mll, onell, &i, x, new_x, &constraint);
	if(!bret) break;
	
	memcpy(values+i*n, constraint, n*sizeof(double));

      } else { //this is only for iRow and jCol

	for(long long j=0; j<n; j++) {
	  iRow[nz]=(int) i; 
	  jCol[nz]=(int) j;
	  nz++;
	}
      }

    }
    delete[] constraint;
    return bret;
  }


  /** Method to return:
   *   1) The structure of the hessian of the lagrangian (if "values" is NULL)
   *   2) The values of the hessian of the lagrangian (if "values" is not NULL)
   */
  virtual bool eval_h(Index n, const Number* x, bool new_x,
                      Number obj_factor, Index m, const Number* lambda,
                      bool new_lambda, Index nele_hess, Index* iRow,
                      Index* jCol, Number* values)
  {
    bool bret = hiopNLP->eval_Hess_Lagr(n, 2, x, new_x, obj_factor, lambda, new_lambda, nele_hess, iRow, jCol, values);
    assert(bret);
    return bret;
  }

  /* This method is called when the algorithm is complete so the TNLP can store/write the solution */
  virtual void finalize_solution(SolverReturn status,
                                 Index n, const Number* x, const Number* z_L, const Number* z_U,
                                 Index m, const Number* g, const Number* lambda,
                                 Number obj_value,
				 const IpoptData* ip_data,
				 IpoptCalculatedQuantities* ip_cq) { };
  
private:
  hiopInterfaceDenseConstraints* hiopNLP;

  /* Methods to block default compiler methods.
   * The compiler automatically generates the following three methods.
   *  Since the default compiler implementation is generally not what
   *  you want (for all but the most simple classes), we usually 
   *  put the declarations of these methods in the private section
   *  and never implement them. This prevents the compiler from
   *  implementing an incorrect "default" behavior without us
   *  knowing. (See Scott Meyers book, "Effective C++")
   *  
   */
  hiop2IpoptTNLP() {};
  hiop2IpoptTNLP(const hiop2IpoptTNLP&) {};
  hiop2IpoptTNLP& operator=(const hiop2IpoptTNLP&);
  //@}

};

#endif
