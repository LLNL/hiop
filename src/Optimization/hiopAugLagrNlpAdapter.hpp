#ifndef HIOP_AUGLAGRNLP_ADAPTER_HPP
#define HIOP_AUGLAGRNLP_ADAPTER_HPP

#include "hiopInterface.hpp"
#include "IpTNLP.hpp"
#include "IpIpoptData.hpp"

//TODO
#define NLP_CLASS_IN Ipopt::TNLP
//use typedef or using rather than define
//typedef Ipopt::TNLP NLP_CLASS_IN;
//using NLP_CLASS_IN = IPOPT::TNLP;

#include "hiopRunStats.hpp"
#include "hiopLogger.hpp"
#include "hiopOptions.hpp"

using namespace Ipopt;

namespace hiop
{
class hiopAugLagrHessian;
class hiopMatrixSparse;
class hiopVectorPar;

class hiopAugLagrNlpAdapter : public hiop::hiopInterfaceDenseConstraints, public Ipopt::TNLP
{
public:
    hiopAugLagrNlpAdapter(NLP_CLASS_IN* nlp_in_);
    virtual ~hiopAugLagrNlpAdapter();

    /***********************************************************************
     * HiOP Interface (overloaded from hip::hiopInterfaceDenseConstraints) *
     ***********************************************************************/                        
    /** problem dimensions: n number of variables, m number of constraints */
    virtual bool get_prob_sizes(
       long long& n,
       long long& m) override;

    /** bounds on the variables (xlow<=-1e20, xupp>=1e20 means upper bound) */
    virtual bool get_vars_info(
       const long long& n,
       double *xlow,
       double* xupp,
       NonlinearityType* type) override;

    /** bounds on the constraints, we have no constraints */ 
    virtual bool get_cons_info(
       const long long& m,
       double* clow,
       double* cupp,
       NonlinearityType* type) override
       { assert(m==0); return true; }

    /** Objective function evaluation, this is the augmented lagrangian function
     * La(x,lambda,rho) = f(x) - lam^t p(x,s) + rho ||p(x,s)||^2
     */
    virtual bool eval_f(
       const long long& n,
       const double* x_in,
       bool new_x,
       double& obj_value) override;

    /** Gradient of the augmented Lagrangian function 
     *  d_La/d_x = df_x - J^T lam + 2rho J^T p(x,s)
     *  d_La/d_s =  0   - (-I) lam[cons_ineq_mapping] + (-I)2rho*p[cons_ineq_mapping]
     *  where p(x,s) is a penalty fcn and rho is the penalty param and
     *  J is the Jacobian of the original NLP constraints.
     *  .
     * */
    virtual bool eval_grad_f(
       const long long& n,
       const double* x_in,
       bool new_x,
       double* gradf) override;

    /** Evaluation of the constraints, we have no constraints (transformed into penalty fcn) */
    virtual bool eval_cons(const long long& n, 
       const long long& m,  
       const long long& num_cons,
       const long long* idx_cons,
       const double* x_in,
       bool new_x,
       double* cons) override
       { assert(m==0); assert(num_cons==0); return true; }

    /** Evaluation of the Jacobian, we have no constraints */
    virtual bool eval_Jac_cons(const long long& n, const long long& m, 
       const long long& num_cons,
       const long long* idx_cons,
       const double* x_in,
       bool new_x,
       double** Jac) override
       { assert(m==0); assert(num_cons==0); return true; }

    /**
     * The get method returns the value of the starting point x0
     * which was set from outside by the Solver and stored in #startingPoint.
     * Motivation: every major iteration we want to reuse the previous
     * solution x_k, not start from the user point every time!!!
     */
    virtual bool get_starting_point(
       const long long& global_n,
       double* x0) override;
    
    /***********************************************************************
     *            IPOPT interface (overloaded from Ipopt::TNLP)            *
     ***********************************************************************/                        
    virtual bool get_nlp_info(
       Index&          n,
       Index&          m,
       Index&          nnz_jac_g,
       Index&          nnz_h_lag,
       IndexStyleEnum& index_style
       ) override;

    virtual bool get_bounds_info(
       Index   n,
       Number* x_l,
       Number* x_u,
       Index   m,
       Number* g_l,
       Number* g_u
       ) override;

    virtual bool get_starting_point(
       Index   n,
       bool    init_x,
       Number* x,
       bool    init_z,
       Number* z_L,
       Number* z_U,
       Index   m,
       bool    init_lambda,
       Number* lambda
       ) override;

    virtual bool eval_f(
       Index         n,
       const Number* x,
       bool          new_x,
       Number&       obj_value
       ) override;

    virtual bool eval_grad_f(
       Index         n,
       const Number* x,
       bool          new_x,
       Number*       grad_f
       ) override;

    virtual bool eval_g(
       Index         n,
       const Number* x,
       bool          new_x,
       Index         m,
       Number*       g
       ) override;

    virtual bool eval_jac_g(
       Index         n,
       const Number* x,
       bool          new_x,
       Index         m,
       Index         nele_jac,
       Index*        iRow,
       Index*        jCol,
       Number*       values
       ) override;

    virtual bool eval_h(
       Index         n,
       const Number* x,
       bool          new_x,
       Number        obj_factor,
       Index         m,
       const Number* lambda_ipopt,
       bool          new_lambda,
       Index         nele_hess,
       Index*        iRow,
       Index*        jCol,
       Number*       values
       ) override;

    virtual void finalize_solution(
       SolverReturn               status,
       Index                      n,
       const Number*              x,
       const Number*              z_L,
       const Number*              z_U,
       Index                      m,
       const Number*              g,
       const Number*              lambda,
       Number                     obj_value,
       const IpoptData*           ip_data,
       IpoptCalculatedQuantities* ip_cq
       ) override;

    /***********************************************************************
     *     Other routines providing access to the internal data            *
     ***********************************************************************/                        
public:
    /**
     * The set method stores the provided starting point by the solver
     * into the private member #startingPoint
     * Motivation: every major iteration we want to reuse the previous
     * solution x_k, not start from the user point every time!!!
     */
    bool set_starting_point(const long long& global_n, const double* x0_in);

    /**
     * The method returns true (and populates x0 with user provided TNLP starting point)
     * or returns false, in which case hiOP will set x0 to all zero.
     */
    bool get_user_starting_point(const long long& global_n, double* x0, bool init_lambda, double *lambda);
   
    /**
     * Returns size of the penalty function stored in member var #m_cons
     */
    bool get_penalty_size(long long& m) { m = m_cons; return true; }

    /**
     * The set method stores the provided penalty into the private
     * member #rho
     */
    inline void set_rho(const double& rho_in) { rho = rho_in; }

    /**
     * The set method stores the provided multipliers into the private
     * member #lambda which is used to evaluate the Augmented Lagrangian fcn.
     */
    void set_lambda(const hiopVectorPar* lambda_in);
    
    /**
     * Evaluates the penalty function residuals and gradient of the Lagrangian
     *
     *  p(x,s)  := [ce(x) - c_rhs; ci(x) - s] = 0
     * 
     *  d_La/d_x = df_x - J^T lam + 2rho J^T p(x,s)
     *  d_La/d_s =  0   - (-I) lam[cons_ineq_mapping] + (-I)2rho*p[cons_ineq_mapping]
     * 
     *  where p(x,s) is a penalty fcn, rho is the penalty param and
     *  J is the Jacobian of the original NLP constraints.
     */
    bool eval_residuals(const long long& n, const double* x_in,
                        const double *zL_in, const double *zU_in,
                        bool new_x, double *penalty, double* grad);

    /**
     * Projection of the gradient vector onto the rectangular box [l, u]
     *             min(0,g)  iff x == xl 
     * proj(g) =   g         iff x in (xl,xg) 
     *             max(0,g)  iff x == xu 
     */                    
    bool project_gradient(const double* x_in, double* grad);                        
    
    /** Objective function evaluation, this is the user objective function f(x) */
    bool eval_f_user(const long long& n, const double* x_in, bool new_x, double& obj_value);
    
    /**
    * IpoptApplication doesn't provide a method to access the solution.
    * The solution is passed to user in a callback finalize_solution() which
    * is implemented here in AugLagrNlpAdapter. The solution is cached in an.
    * array #_solutionIpopt. Thare is no guarantee that the solution
    * is correct or has been initialized, the user calling this function needs
    * to make sure Ipopt has finished successfuly. Only then the valid solution
    * will be returned.
    */
    void get_ipoptSolution(double *x) const;
    void get_ipoptBoundMultipliers(double *z_L, double *z_U) const;
    int  get_ipoptNumIters() const;

   /**
    * Returns scaling factor for dual infeasibility based on average norm of the dual variables
    * sd = |lambda|_1 + |z_l|_1 + |z_u|+1 / (m+2n)
    * sd = max(100, sd)/100
    */
    void get_dualScaling(double &sd);

protected:
    /** Allocates space for internal variables */
    bool initialize();

    /**
     * Evaluates the original NLP constraints L <= c(x) <= U and transforms the constraints 
     * into the penalty form p(x) = 0 appropriate for Augmented Lagrangian formulation.
     * The penalty terms consist of:
     * Equality constraints:  c(x) - c_rhs
     * Inequality constraints c(x) - s, where L <= s <= U
     * The evaluated penalty function is stored in member #_penaltyFcn
     */
    bool eval_penalty(const double *x_in, bool new_x);

    /**
     *  Evaluates Jacobian of the penalty function. Jacobian is stored in the
     *  member #_penaltyFcn_jacobian. The sparse structure is initialized during
     *  the first call.
     */
    bool eval_penalty_jac(const double *x_in, bool new_x);

    /** Gradient of the Lagrangian function L(x,s) = f(x) - lam^t p(x)
     *  d_L/d_x = df_x - J^T lam 
     *  d_L/d_s =  0   - (-I) lam[cons_ineq_mapping] 
     *  where J is the Jacobian of the original NLP constraints.
     *
     * @param[in] n Number of variables in Augmented Lagrangian formulation
     * @param[in] x_in Variables consisting of original NLP variables and additional slacks
     * @param[in] new_x
     * @param[out] gradLagr Returns gradient of the Lagrangian function L(x, lambda)
     * */
    bool eval_grad_Lagr(const long long& n, const double* x_in, bool new_x, double* gradLagr);

protected:
    //general nlp to be "adapted" to Augmented Lagrangian form Ipopt::TNLP.
    //Note that Ipopt uses type Index (aka int) and Number (aka double) in
    //its interface, while we are using long long and double. We need to be
    //careful about the former.
    NLP_CLASS_IN* nlp_in;

    //TODO adapt also from hiop::hiopInterfaceDenseConstraints
    //hiop::hiopInterfaceDenseConstraints* nlp;
    
    //specific variables of the augmented lagrangian formulation
    double rho; ///< penalty parameter for the quadratic penalty term ||p(x,s)||^2
    hiopVectorPar *lambda; ///< Lagrange multipliers
    hiopVectorPar *startingPoint; //< stored initial guess (including slack)

    //various sizes and properties of the original NLP problem
    long long n_vars; ///< number of primal variables x (original NLP problem)
    long long n_slacks; ///< number of slack variables s (equal to #ineq constr in original NLP)
    long long m_cons; ///< number of overall constraints (original NLP problem)
    long long m_cons_eq; ///< number of equality constraints (original NLP problem)
    long long m_cons_ineq; ///< number of inequality constraints (original NLP problem)
    long long nnz_jac; //< number of nonzeros in Jacobian of constraints (original NLP problem)
    long long nnz_hess; ///< number of nnz in Hessian of Lagrangian (original NLP prob)
    hiopVectorPar *xl, *xu; ///< x variable bounds (original NLP problem)
    hiopVectorPar *sl, *su; ///< slack variables bounds (equal to ineq. bounds in original NLP problem)

    //TODO: class for the constraints
    //auxiliary arrays for handling the original NLP constraints
    long long *cons_eq_mapping, *cons_ineq_mapping; ///< indices of eq. and ineq. constraints
    double *c_rhs; ///< rhs for the equality constraints

    hiopVectorPar *_solutionIpopt; ///< cached Ipopt solution from finalize_solution()
    hiopVectorPar *_zLowIpopt, *_zUppIpopt; ///< cached Ipopt multipliers for the bounds
    int _numItersIpopt; ///< cached number of Ipopt iterations

    //working memory for internal evaluations of the AL functions
    //motivation to have in on class level is to avoid alloc/dealloc
    //during each call of the evaluation routines
    hiopVectorPar *_penaltyFcn; ///< original constraints transformed  AL penalty function p(x)=0
    hiopMatrixSparse *_penaltyFcn_jacobian; ///< Jacobian of the the original NLP constraints, which is equivalent to the Jacobian of the penalty fcn. w.r.t the primal variables x (excluding slacks)
   
    hiopAugLagrHessian *_hessian; ///<hessian of the AL

public:
    /* outputing and debug-related functionality*/
    hiopRunStats runStats;
    hiopOptions* options;
    hiopLogger* log;
};

}

#endif
