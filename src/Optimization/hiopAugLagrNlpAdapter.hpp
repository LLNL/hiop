#ifndef HIOP_AUGLAGRNLP_ADAPTER_HPP
#define HIOP_AUGLAGRNLP_ADAPTER_HPP

#include "hiopInterface.hpp"
#include "hiopVector.hpp"

#include "hiopRunStats.hpp"
#include "hiopLogger.hpp"
#include "hiopOptions.hpp"

//TODO
#define NLP_CLASS_IN Ipopt::TNLP
//use typedef or using rather than define
//typedef Ipopt::TNLP NLP_CLASS_IN;
//using NLP_CLASS_IN = IPOPT::TNLP;

//TODO
#include "IpIpoptCalculatedQuantities.hpp"
#include "IpIpoptApplication.hpp"
#include "IpTNLPAdapter.hpp"
#include "IpOrigIpoptNLP.hpp"
#include "IpTNLP.hpp"
#include "IpIpoptData.hpp"

//TODO
using namespace Ipopt;

namespace hiop
{

class hiopAugLagrNlpAdapter : public hiop::hiopInterfaceDenseConstraints
{
public:
    hiopAugLagrNlpAdapter(NLP_CLASS_IN* nlp_in_);
    virtual ~hiopAugLagrNlpAdapter();

    /** problem dimensions: n number of variables, m number of constraints */
    bool get_prob_sizes(long long& n, long long& m)
        { n=n_vars+n_slacks; m=0; return true; }

    /** bounds on the variables 
     *  (xlow<=-1e20 means no lower bound, xupp>=1e20 means no upper bound) */
    bool get_vars_info(const long long& n, double *xlow, double* xupp, NonlinearityType* type);

    /** bounds on the constraints, we have no constraints */ 
    bool get_cons_info(const long long& m, double* clow, double* cupp, NonlinearityType* type)
        { assert(m==0); return true; }

    /** Objective function evaluation, this is the augmented lagrangian function
     * La(x,lambda,rho) = f(x) + lam^t p(x,s) + rho ||p(x,s)||^2
     */
    bool eval_f(const long long& n, const double* x_in, bool new_x, double& obj_value);

    /** Gradient of the augmented Lagrangian function 
     *  d_La/d_x = df_x + J^T lam + 2rho J^T p(x,s)
     *  d_La/d_s =  0   + (-I) lam[cons_ineq_mapping] + (-I)2rho*p[cons_ineq_mapping]
     * */
    bool eval_grad_f(const long long& n, const double* x_in, bool new_x, double* gradf);

    /** Evaluation of the constraints, we have no constraints */
    bool eval_cons(const long long& n, 
            const long long& m,  
            const long long& num_cons, const long long* idx_cons,
            const double* x_in, bool new_x, double* cons)
        { assert(m==0); assert(num_cons==0); return true; }

    /** Evaluation of the Jacobian, we have no constraints */
    bool eval_Jac_cons(const long long& n, const long long& m, 
            const long long& num_cons, const long long* idx_cons,
            const double* x_in, bool new_x, double** Jac) 
        { assert(m==0); assert(num_cons==0); return true; }

    /**
     * The get method returns the value of the starting point x0
     * which was set from outside by the Solver and stored in #x0_AugLagr.
     * Motivation: every major iteration we want to reuse the previous
     * solution x_k, not start from the user point every time!!!
     */
    bool get_starting_point(const long long& global_n, double* x0);

    /**
     * The set method stores the provided starting point into the private
     * member #x0_AugLagr
     * Motivation: every major iteration we want to reuse the previous
     * solution x_k, not start from the user point every time!!!
     */
    bool set_starting_point(const long long& global_n, const double* x0_in);

    /**
     * The method returns true (and populates x0 with user provided TNLP starting point)
     * or returns false, in which case hiOP will set x0 to all zero.
     */
    bool get_user_starting_point(const long long& global_n, double* x0);

    /**
     * The set method stores the provided penalty into the private
     * member #rho
     */
    inline void set_rho(const double& rho_in)
        { rho = rho_in; }
   
    /**
     * Returns size of the penalty function stored in member var #m_cons
     */
    bool get_penalty_size(long long& m) { m = m_cons; return true; }

    /**
     * The set method stores the provided multipliers into the private
     * member #lambda which is used to evaluate the Augmented Lagrangian fcn.
     */
    void set_lambda(const hiopVectorPar* lambda_in);
    
    /**
     * Evaluates the penalty function residuals and gradient of the Lagrangian
     *
     * penalty  := [ce(x) - c_rhs; ci(x) - s] = 0
     * gradLagr := d_L/d_x = df_x + J^T lam   = 0
     *
     */
    bool eval_residuals(const long long& n, const double* x_in,
                        bool new_x, double *penalty, double* gradLagr);

protected:
    bool initialize();
    bool eval_penalty(const double *x_in, bool new_x, double *penalty_data);
    bool eval_grad_Lagr(const long long& n, const double* x_in, bool new_x, double* gradLagr);

protected:
    //general nlp to be "adapted" to Augmented Lagrangian form Ipopt::TNLP
    //needed by HiOp's AugLagr solver
    NLP_CLASS_IN* nlp_in;

    //TODO adapt also from hiop::hiopInterfaceDenseConstraints
    //hiop::hiopInterfaceDenseConstraints* nlp;
    
    //specific variables of the augmented lagrangian formulation
    double rho; ///< penalty parameter for the quadratic penalty term ||c(x)||^2
    hiopVectorPar *lambda; ///< Lagrange multipliers
    hiopVectorPar *x0_AugLagr; //< stored starting point (including slack)

    //various sizes and properties of the original NLP problem
    long long n_vars; ///< number of primal variables x (original NLP problem)
    long long n_slacks; ///< number of slack variables (equal to #ineq constr in original NLP)
    long long m_cons; ///< number of constraints (original NLP problem)
    long long m_cons_eq; ///< number of equality constraints (original NLP problem)
    long long m_cons_ineq; ///< number of inequality constraints (original NLP problem)
    long long nnz_jac; //< number of nonzeros in jacobian of constraints (original NLP problem)
    hiopVectorPar *xl, *xu; ///< variables bounds (original NLP problem)
    hiopVectorPar *sl, *su; ///< slack variables bounds (equal to ineq. cons. bounds in original NLP problem)

    //auxiliary arrays for handling the original NLP constraints
    long long *cons_eq_mapping, *cons_ineq_mapping; 
    hiopVectorPar *c_rhs; ///< rhs for the equality constraints

    //working memory for internal evaluations of the AL functions
    //motivation to have in on class level is to avoid alloc/dealloc
    //during each call of the evaluation routines
    hiopVector *penalty_fcn; ///< original constraints transformed  AL penalty function p(x)=0
    hiopSparseMatrix *penalty_fcn_jacobian; ///< Jacobian of the penalty w.r.t the primal variables x (excluding slacks), which is equivalent to the Jacobian of the original NLP constraints

    /* outputing and debug-related functionality*/
    //TODO: do we need these? see NlpFormulation.hpp wrapper
    hiopLogger* log;
    hiopRunStats runStats; //times functionEvaluations
    hiopOptions* options;
};

}

#endif
