#include "hiopAugLagrNlpAdapter.hpp"
#include "hiopLogger.hpp"

#ifdef HIOP_USE_MPI
#include "mpi.h"
#else 
#include <cstddef>
#endif

#include <stdlib.h>     /* exit, EXIT_FAILURE */
#include <cstring>      // memcpy
#include <cassert>

namespace hiop
{

hiopAugLagrNlpAdapter::hiopAugLagrNlpAdapter(NLP_CLASS_IN* nlp_in_):
    nlp_in(nlp_in_),
    rho(100.0),
    lambda(nullptr),
    x0_AugLagr(nullptr),
    n_vars(0),
    n_slacks(0),
    m_cons(0),
    m_cons_eq(0),
    m_cons_ineq(0),
    nnz_jac(0),
    xl(nullptr),
    xu(nullptr),
    sl(nullptr),
    su(nullptr),
    cons_eq_mapping(nullptr),
    cons_ineq_mapping(nullptr),
    c_rhs(nullptr),
    penalty_fcn(nullptr),
    penalty_fcn_jacobian(nullptr),
    log(new hiopLogger(this, stdout)),
    runStats(MPI_COMM_SELF),
    options(new hiopOptions(/*filename=NULL*/))
{
    //MPI_Comm comm = MPI_COMM_SELF;
    hiopOutVerbosity hov = (hiopOutVerbosity) options->GetInteger("verbosity_level");
    options->SetLog(log);
   
    // initializes the member variables
    initialize(); //TODO: Check if everything is allocated
}

hiopAugLagrNlpAdapter::~hiopAugLagrNlpAdapter()
{
    if(xl) delete xl; //TODO: can be deleted earlier, after get_vars_info?
    if(xu) delete xu;
    if(sl) delete sl;
    if(su) delete su;
    if(cons_eq_mapping)       delete cons_eq_mapping;
    if(cons_ineq_mapping)     delete cons_ineq_mapping;
    if(c_rhs)                 delete c_rhs;
    if(lambda)                delete lambda;
    if(x0_AugLagr)             delete x0_AugLagr;
    if(penalty_fcn)              delete penalty_fcn;
    if(penalty_fcn_jacobian)      delete penalty_fcn_jacobian;
    if(log)      delete log;
    if(options)  delete options;
}

/**
 * Analyzes the original NLP problem and determines size
 * of the Augmented Lagrangian (AL) formulation. Allocates
 * space for the vectors and matrices.
 * */
bool hiopAugLagrNlpAdapter::initialize()
{
    /**************************************************************************/
    /*  Analyze the original NLP variables and determine the new slack vars   */
    /**************************************************************************/
    //determine the original NLP problem size
    Ipopt::Index n_nlp, m_nlp, nnz_jac_nlp, dum1;
    Ipopt::IndexStyleEnum index_style = C_STYLE;
    //TODO Ipopt uses type Index*, is it compatible with long long?
    bool bret = nlp_in->get_nlp_info(n_nlp, m_nlp, nnz_jac_nlp, dum1, index_style);
    assert(bret);

    //create storage for variable and constraint bounds of the original NLP
    xl = new hiopVectorPar(n_nlp);
    xu = xl->alloc_clone();
    hiopVectorPar *gl = new hiopVectorPar(m_nlp); 
    hiopVectorPar *gu = new hiopVectorPar(m_nlp);
    
    double *xl_nlp= xl->local_data(), *xu_nlp= xu->local_data();
    double *gl_nlp=gl->local_data(),  *gu_nlp=gu->local_data();

    //get the bounds of the original NLP problem
    //TODO Ipopt uses type Index*, is it compatible with long long?
    bret = nlp_in->get_bounds_info(n_nlp, xl_nlp, xu_nlp, m_nlp, gl_nlp, gu_nlp);
    assert(bret);
    
    //analyze constraints to determine how many slacks are needed 
    //by splitting the constraints into equalies and inequalities
    //n_cons_eq=n_cons_ineq=0;// by default set in the constructor
    for(int i=0;i<m_nlp; i++) {
    if(gl_nlp[i]==gu_nlp[i]) n_cons_eq++;
    else                     n_cons_ineq++;
    }

    //update member properties
    n_vars = n_nlp; ///< number of primal variables x (not including slacks)
    m_cons = m_nlp; ///< number of eq and ineq constraints
    n_slacks = n_cons_ineq; ///< number of slack variables
    nnz_jac = nnz_jac_nlp; //< number of nonzeros in Jacobian ( w.r.t variables of the original NLP variables x, not slacks)

    //allocate space 
    lambda = new hiopVectorPar(m_cons);
    x0_AugLagr = new hiopVectorPar(n_vars+n_slacks);
    penalty_fcn = new hiopVectorPar(m_cons);
    penalty_fcn_jacobian = new hiopSparseMatrix(m_cons, n_vars, nnz_jac);
    sl = new hiopVectorPar(n_cons_ineq);
    su = new hiopVectorPar(n_cons_ineq);

    /**************************************************************************/
    /*              Analyze the original NLP constraints                      */
    /**************************************************************************/
    double *sl_nlp=sl->local_data(), *su_nlp=su->local_data();
    
    //We need to store the eq/ineq mapping
    //because we need to evaluate c(x) in the eval_f()
    //Constraint evaluations consist of:
    // ->  c(x) - c_rhs = 0 (eq. constr)
    // ->  c(x) - s     = 0 (ineq. constr)
    c_rhs = new hiopVectorPar(n_cons_eq);
    double *c_rhsvec=c_rhs->local_data();
    cons_eq_mapping   = new long long[n_cons_eq];
    cons_ineq_mapping = new long long[n_cons_ineq];
  
    // copy lower and upper bounds of the constraints and get indices of eq/ineq constraints
    int it_eq=0, it_ineq=0;
    for(int i=0;i<m_nlp; i++) {
      if(gl_vec[i]==gu_vec[i]) {
        c_rhsvec[it_eq] = gl_vec[i]; 
        cons_eq_mapping[it_eq]=i;
        it_eq++;
      } else {
  #ifdef HIOP_DEEPCHECKS
      assert(gl_vec[i] <= gu_vec[i] && "please fix the inconsistent inequality constraints, otherwise the problem is infeasible");
  #endif
        sl_nlp[it_ineq]=gl_vec[i]; su_nlp[it_ineq]=gu_vec[i]; 
        cons_ineq_mapping[it_ineq]=i;
        it_ineq++;
      }
    }
    assert(it_eq==n_cons_eq); assert(it_ineq==n_cons_ineq);

    delete gl;
    delete gu;

    return true;
}


/**
 * This functions transforms the decision variables of the original NLP problem
 * into the variables of the Augmented lagrangian formulation, including the slacks.
 * The new variable vector is xlow <= x <= xup where x consists of [x_nlp, s]
 * */
bool hiopAugLagrNlpAdapter::get_vars_info(const long long& n, double *xlow, double* xupp, NonlinearityType* type)
{
    assert(n == n_vars + n_slacks);

    //fill in lower and upper bounds (including slack variables)
    xl.copyTo(xlow);
    xu.copyTo(xupp);
    sl.copyTo(xlow + n_nlp);
    su.copyTo(xupp + n_nlp);

    //set variables type (x - nonlinear, s - linear)
    for(int i=0 i<n_vars; i++)  type[i] = hiopNonlinear;
    for(int i=n_vars; i<n; i++) type[i] = hiopLinear;

    //TODO: delete sl,xl here? not needed afterwards?

    return true;
}

/**
 * Evaluates the original NLP constraints L <= c(x) <= U and transforms the constraints 
 * into the penalty form p(x) = 0 appropriate for Augmented Lagrangian formulation.
 * The penalty terms consist of:
 * Equality constraints:  c(x) - c_rhs
 * Inequality constraints c(x) - s, where L <= s <= U
 */
bool hiopAugLagrNlpAdapter::eval_penalty(const double *x_in, bool new_x, double *penalty_data)
{
    const double *rhs_data = c_rhs->local_data_const();
    const double *slacks  = x_in + n_vars;

    //evaluate the original NLP constraints
    //TODO Ipopt uses type Index*, is it compatible with long long?
    bool bret = nlp_in->eval_g(n_vars, x_in, new_x, m_cons, penalty_data);
    assert(bret);

    //adjust equality constraints
    // c(x) - c_rhs
    for (int i = 0; i<m_cons_eq; i++)
    {
        penalty_data[cons_eq_mapping[i]] -= rhs_data[i]; 
    }
    
    //adjust inequality constraints
    // c(x) - s
    for (int i = 0; i<m_cons_ineq; i++)
    {
        penalty_data[cons_ineq_mapping[i]] -= slacks[i]; 
        assert(slacks[i] >= 0);
    }

    return true;
}

/** 
 * Objective function evaluation, this is the augmented lagrangian function
 * La(x,lambda,rho) = f(x) + lam^t p(x) + rho ||p(x)||^2.
 *
 * @param[in] n Number of variables in Augmented Lagrangian formulation
 * @param[in] x_in Variables consisting of original NLP variables and additional slacks
 * @param[in] new_x
 * @param[out] obj_value Returns Augmented lagrangian value La(x, lambda, rho)
 * */
bool hiopAugLagrNlpAdapter::eval_f(const long long& n, const double* x_in, bool new_x, double& obj_value)
{
    assert(n == n_vars + n_slacks);

    // evaluate the original NLP objective f(x), uses only firs n_vars entries of x_in
    double obj_nlp;
    bool bret = nlp_in->eval_f(n_vars, x_in, new_x, obj_nlp); 
    assert(bret);

    // evaluate and transform the constraints of the original NLP
    eval_penalty(x_in, new_x, penalty_fcn);

    // compute lam^t p(x)
    const double lagr_term = lambda.dotProductWith(penalty_fcn);
    
    // compute penalty term rho*||p(x)||^2
    const double penalty_term = rho * penalty_fcn.dotProductWith(penalty_fcn);

    //f(x) + lam^t p(x) + rho ||p(x)||^2
    obj_value = obj_nlp + lagr_term + penalty_term;

    return true;
}

/** Gradient of the Lagrangian function L(x,s)
 *  d_La/d_x = df_x + J^T lam 
 *  d_La/d_s =  0   + (-I) lam[cons_ineq_mapping] 
 *  where J is the Jacobian of the original NLP constraints.
 *
 * @param[in] n Number of variables in Augmented Lagrangian formulation
 * @param[in] x_in Variables consisting of original NLP variables and additional slacks
 * @param[in] new_x
 * @param[out] gradLagr Returns gradient of the Lagrangian function L(x, lambda)
 * */
bool hiopAugLagrNlpAdapter::eval_grad_Lagr(const long long& n, const double* x_in, bool new_x, double* gradLagr)
{
    assert(new_x == false); // we assume data in #penalty_fcn are up to date
    
    /****************************************************/
    /** Add contribution of the NLP objective function **/
    /****************************************************/
    //TODO Ipopt uses type Index*, is it compatible with long long?
    bool bret = nlp_in->eval_grad_f(n_vars, x_in, new_x, gradLagr);
    assert(bret);
    std::fill(gradLagr+n_vars, gradLagr+n, 0.0); //clear grad w.r.t slacks

    /****************************************************/
    /* Evaluate Jacobian of the original NLP problem */
    /****************************************************/
    
    //TODO Ipopt uses type Index*, is it compatible with long long?
    long long *iRow = penalty_fcn_jacobian->get_ia();
    long long *icol = penalty_fcn_jacobian->get_ja();
    double *values = penalty_fcn_jacobian->get_a();
    bret = nlp_in->eval_jac_g(n_vars, x_in, new_x, m_cons, iRow, jCol, values);
    assert(bret);

    /**************************************************/
    /**    Compute lagrangian term contribution       */
    /**************************************************/
    //y := alpha*A*x + beta*y sparse DGEMV
    //y := alpha*A'*x + beta*y sparse DGEMVt
    //penalty_fcn_jacobian.DGEMVt(alpha, x, beta, y);
    penalty_fcn_jacobian.DGEMVt(1.0, lam, 1.0, gradLagr);

    
    //Add the Jacobian w.r.t the slack variables (penalty_fcn_jacobian contains
    //only the jacobian w.r.t original x, we need to add Jac w.r.t slacks)
    //The structure of the La Jacobian is following (Note that Je, Ji
    //might be interleaved and not listed in order as shown below)
    //       | Je  0  |
    // Jac = |        |
    //       | Ji  -I |
    for (int i = 0; i<m_cons_ineq; i++)
    {
        gradLagr[n_vars + i] -= lambda[cons_ineq_mapping[i]];
    }

    return true;
}

/** Gradient of the Augmented Lagrangian function La(x,s)
 *  d_La/d_x = df_x + J^T lam + 2rho J^T p(x,s)
 *  d_La/d_s =  0   + (-I) lam[cons_ineq_mapping] + (-I)2rho*p[cons_ineq_mapping]
 *  where J is the Jacobian of the original NLP constraints.
 *
 * @param[in] n Number of variables in Augmented Lagrangian formulation
 * @param[in] x_in Variables consisting of original NLP variables and additional slacks
 * @param[in] new_x
 * @param[out] gradf Returns gradient of the Augmented Lagrangian function La(x, lambda, rho)
 * */
bool hiopAugLagrNlpAdapter::eval_grad_f(const long long& n, const double* x_in, bool new_x, double* gradf)
{
    assert(new_x == false); // we assume data in #penalty_fcn are up to date

    /********************************************************/
    /** Add contribution of the gradient of the Lagrangian **/
    /********************************************************/
    eval_grad_Lagr(n, x_in, new_x, gradf);
    
    /**************************************************/
    /**    Compute penalty term contribution          */
    /**************************************************/
    //y := alpha*A*x + beta*y sparse DGEMV
    //y := alpha*A'*x + beta*y sparse DGEMVt
    const double *penalty_data = penalty_fcn->local_data_const();
    //penalty_fcn_jacobian.DGEMVt(alpha, x, beta, y);
    penalty_fcn_jacobian.DGEMVt(2*rho, penalty_data, 1.0, gradf);

    //Add the Jacobian w.r.t the slack variables (penalty_fcn_jacobian contains
    //only the jacobian w.r.t original x, we need to add Jac w.r.t slacks)
    //The structure of the La Jacobian is following (Note that Je, Ji
    //might be interleaved and not listed in order as shown below)
    //       | Je  0  |
    // Jac = |        |
    //       | Ji  -I |
    for (int i = 0; i<m_cons_ineq; i++)
    {
        gradf[n_vars + i] -= 2*rho*penalty_fcn[cons_ineq_mapping[i]];
    }

    return true;
}

/**
 * The get method returns the value of the starting point x0
 * which was set from outside by the Solver and stored in #x0_AugLagr.
 * Motivation: every major iteration we want to reuse the previous
 * solution x_k, not start from the user point every time!!!
 */
bool hiopAugLagrNlpAdapter::get_starting_point(const long long &global_n, double* x0)
{
    memcpy(x0, x0_AugLagr, global_n*sizeof(double));
    return true;
}

/**
 * The set method stores the provided starting point into the private
 * member #x0_AugLagr
 * Motivation: every major iteration we want to reuse the previous
 * solution x_k, not start from the user point every time!!!
 */
bool hiopAugLagrNlpAdapter::set_starting_point(const long long &global_n, const double* x0_in)
{
    memcpy(x0_AugLagr, x0_in, global_n*sizeof(double));
    return true;
}


void hiopAugLagrNlpAdapter::set_lambda(const hiopVector* lambda_in)
{
    memcpy(lambda->local_data(), lambda_in->local_data_const(), m_cons*sizeof(double));
}

/**
 * The method returns true (and populates x0 with user provided TNLP starting point)
 * or returns false, in which case hiOP will set x0 to all zero.
 */
bool hiopAugLagrNlpAdapter::get_user_starting_point(const long long &global_n, double* x0)
{
    assert(global_n == n_vars + n_slacks);

    //call starting point from the adapted nlp
    bool bret = nlp_in->get_starting_point(n_vars, x0);
    if (!bret) std::fill(x0, x0+n_vars, 0.);
  
    //zero out the slack variables
    std::fill(x0+n_vars, x0+n_vars+n_slacks, 0.);
    //TODO is initialization by zero the best way?
    //can we set it at least within the bounds? or equal to c(x)?

    return bret;
}

//TODO: can we reuse the last jacobian instead of recomputing it? does hiop/ipopt evaluate the Jacobian in the last (xk + searchDir), a.k.a the solution? 
bool hiopAugLagrNlpAdapter::eval_residuals(const long long& n, const double* x_in, bool new_x, double *penalty, double* gradLagr)
{
   assert(n == n_vars);
   assert(new_x == false);

    /*****************************************/
    /*          Penalty Function             */
    /*        [ce(x) - c_rhs; ci(x) - s]     */
    /*****************************************/
    bool bret = eval_penalty(x_in, new_x, penalty);
    assert(bret);
   
    /**************************************************/
    /**        Compute the Lagrangian term            */
    /**     gradLagr := d_L/d_x = df_x + J^T lam      */
    /**************************************************/
    bret = eval_grad_Lagr(n, x_in, new_x, gradLagr);
    assert(bret);

    return bret;
}
}
