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
    _startingPoint(nullptr),
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
    _penaltyFcn(nullptr),
    _penaltyFcn_jacobian(nullptr),
    runStats(),
    options(new hiopOptions(/*filename=NULL*/)),
    log(new hiopLogger(options, stdout, 0))
{
    options->SetLog(log);
    //options->SetIntegerValue("verbosity_level", options->GetInteger("verbosity_level_major"));
    options->SetIntegerValue("verbosity_level", 5);
   
    // initializes the member variables
    initialize();
}

hiopAugLagrNlpAdapter::~hiopAugLagrNlpAdapter()
{
    //xl,xu,sl,su cannot be deleted earlier (e.g. after get_vars_info) because
    //we call inner problem repeatedly, thus get_var_info is called
    //multiple times
    if(xl) delete xl; 
    if(xu) delete xu;
    if(sl) delete sl;
    if(su) delete su;

    if(cons_eq_mapping)       delete cons_eq_mapping;
    if(cons_ineq_mapping)     delete cons_ineq_mapping;
    if(c_rhs)                 delete c_rhs;
    if(lambda)                delete lambda;
    if(_startingPoint)             delete _startingPoint;
    if(_penaltyFcn)              delete _penaltyFcn;
    if(_penaltyFcn_jacobian)      delete _penaltyFcn_jacobian;
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
    //determine the original NLP problem size
    Ipopt::Index n_nlp, m_nlp, nnz_jac_nlp, dum1;
    TNLP::IndexStyleEnum index_style = TNLP::C_STYLE;
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
    bret = nlp_in->get_bounds_info(n_nlp, xl_nlp, xu_nlp, m_nlp, gl_nlp, gu_nlp);
    assert(bret);
    
    /****************************************************************/
    /*  Analyze the original NLP and determine the slack vars count */
    /****************************************************************/
    for(Ipopt::Index i=0; i<m_nlp; i++) {
    if(gl_nlp[i]==gu_nlp[i]) m_cons_eq++;
    else                     m_cons_ineq++;
    }

    /****************************************************************/
    /*  Update the member properties and allocate the vectors       */
    /****************************************************************/
    n_vars = n_nlp;
    m_cons = m_nlp;
    n_slacks = m_cons_ineq;
    nnz_jac = nnz_jac_nlp;

    //allocate space 
    lambda = new hiopVectorPar(m_cons);
    _startingPoint = new hiopVectorPar(n_vars+n_slacks);
    _penaltyFcn = new hiopVectorPar(m_cons);
    _penaltyFcn_jacobian = new hiopMatrixSparse(m_cons, n_vars, nnz_jac);
    sl = new hiopVectorPar(m_cons_ineq);
    su = new hiopVectorPar(m_cons_ineq);

    /**************************************************************************/
    /*  Analyze the original NLP constraints and split them into eq/ineq      */
    /**************************************************************************/
    double *sl_nlp=sl->local_data(), *su_nlp=su->local_data();
    
    //We need to store the eq/ineq mapping
    //because we need to evaluate c(x) in eval_f()
    //Constraint evaluations consist of:
    // ->  c(x) - c_rhs = 0 (eq. constr)
    // ->  c(x) - s     = 0 (ineq. constr)
    c_rhs = new hiopVectorPar(m_cons_eq);
    double *c_rhsvec=c_rhs->local_data();
    cons_eq_mapping   = new long long[m_cons_eq];
    cons_ineq_mapping = new long long[m_cons_ineq];
  
    // copy lower and upper bounds of the constraints and get indices of eq/ineq constraints
    int it_eq=0, it_ineq=0;
    for(Ipopt::Index i=0; i<m_nlp; i++) {
      if(gl_nlp[i]==gu_nlp[i]) {
        c_rhsvec[it_eq] = gl_nlp[i]; 
        cons_eq_mapping[it_eq] = (long long) i;
        it_eq++;
      } else {
  #ifdef HIOP_DEEPCHECKS
      assert(gl_nlp[i] <= gu_nlp[i] && "please fix the inconsistent inequality constraints, otherwise the problem is infeasible");
  #endif
        sl_nlp[it_ineq]=gl_nlp[i]; su_nlp[it_ineq]=gu_nlp[i]; 
        cons_ineq_mapping[it_ineq] = (long long) i;
        it_ineq++;
      }
    }
    assert(it_eq==m_cons_eq); assert(it_ineq==m_cons_ineq);

    delete gl;
    delete gu;

    //std::cout << "Initialized AugLagrNlpAdapter with the following values:" << std::endl;
    //std::cout << "n_vars " << n_vars << std::endl;
    //std::cout << "n_slacks " << n_slacks << std::endl;
    //std::cout << "m_cons " << m_cons << std::endl;
    //std::cout << "m_cons_eq " << m_cons_eq << std::endl;
    //std::cout << "m_cons_ineq " << m_cons_ineq << std::endl;
    //std::cout << "nnz_jac " << nnz_jac << std::endl;

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
    xl->copyTo(xlow);
    xu->copyTo(xupp);
    sl->copyTo(xlow + n_vars);
    su->copyTo(xupp + n_vars);

    //set variable types (x - nonlinear, s - linear)
    for(long long i=0; i<n_vars; i++)  type[i] = hiopNonlinear;
    for(long long i=n_vars; i<n; i++)  type[i] = hiopLinear;

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
    bool bret = nlp_in->eval_g((Ipopt::Index)n_vars, x_in, new_x,
                               (Ipopt::Index)m_cons, penalty_data);
    assert(bret);

    //adjust equality constraints
    // c(x) - c_rhs
    for (long long i = 0; i<m_cons_eq; i++)
    {
        penalty_data[cons_eq_mapping[i]] -= rhs_data[i]; 
    }
    
    //adjust inequality constraints
    // c(x) - s
    for (long long i = 0; i<m_cons_ineq; i++)
    {
        penalty_data[cons_ineq_mapping[i]] -= slacks[i]; 
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

    runStats.tmEvalObj.start();

    // evaluate the original NLP objective f(x), uses only firs n_vars entries of x_in
    double obj_nlp;
    bool bret = nlp_in->eval_f((Ipopt::Index)n_vars, x_in, new_x, obj_nlp); 
    assert(bret);

    // evaluate and transform the constraints of the original NLP
    eval_penalty(x_in, new_x, _penaltyFcn->local_data());

    // compute lam^t p(x)
    const double lagr_term = lambda->dotProductWith(*_penaltyFcn);
    
    // compute penalty term rho*||p(x)||^2
    const double penalty_term = rho * _penaltyFcn->dotProductWith(*_penaltyFcn);

    //f(x) + lam^t p(x) + rho ||p(x)||^2
    obj_value = obj_nlp + lagr_term + penalty_term;

    runStats.tmEvalObj.stop();
    runStats.nEvalObj++;

    //std::cout << "Evaluating objective function:" << std::endl;
    //std::cout << "nlp_f: " << obj_nlp << std::endl;
    //std::cout << "lagr term: " << lagr_term << std::endl;
    //std::cout << "penalty term: " << penalty_term <<  std::endl;
    //std::cout << "penalty: " << rho <<  std::endl;
    //std::cout << "final f(x): " << obj_value <<  std::endl;


    return true;
}

bool hiopAugLagrNlpAdapter::eval_f_user(const long long& n, const double* x_in, bool new_x, double& obj_value)
{
    assert(n == n_vars + n_slacks);

    runStats.tmEvalObj.start();

    // evaluate the original NLP objective f(x), uses only firs n_vars entries of x_in
    bool bret = nlp_in->eval_f((Ipopt::Index)n_vars, x_in, new_x, obj_value); 
    assert(bret);
    
    runStats.tmEvalObj.stop();

    return bret;
}

/** Gradient of the Lagrangian function L(x,s)
 *  d_L/d_x = df_x + J^T lam 
 *  d_L/d_s =  0   + (-I) lam[cons_ineq_mapping] 
 *  where J is the Jacobian of the original NLP constraints.
 *
 * @param[in] n Number of variables in Augmented Lagrangian formulation
 * @param[in] x_in Variables consisting of original NLP variables and additional slacks
 * @param[in] new_x
 * @param[out] gradLagr Returns gradient of the Lagrangian function L(x, lambda)
 * */
bool hiopAugLagrNlpAdapter::eval_grad_Lagr(const long long& n, const double* x_in, bool new_x, double* gradLagr)
{
    //TODO new_x
    if (true) eval_penalty(x_in, new_x, _penaltyFcn->local_data());
    
    /****************************************************/
    /** Add contribution of the NLP objective function **/
    /****************************************************/
    bool bret = nlp_in->eval_grad_f((Ipopt::Index)n_vars, x_in, new_x, gradLagr);
    assert(bret);
    std::fill(gradLagr+n_vars, gradLagr+n, 0.0); //clear grad w.r.t slacks

    /****************************************************/
    /* Evaluate Jacobian of the original NLP problem */
    /****************************************************/
    static bool initializedStructure = false;
    if (!initializedStructure)
    {
      //initialize the structure during the first call, later update only the values
      int *iRow = _penaltyFcn_jacobian->get_iRow();
      int *jCol = _penaltyFcn_jacobian->get_jCol();
      bret = nlp_in->eval_jac_g((Ipopt::Index)n_vars, nullptr, new_x,
                              (Ipopt::Index)m_cons, nnz_jac, iRow, jCol, nullptr);
      assert(bret);
      initializedStructure = true;
    }

    double *values = _penaltyFcn_jacobian->get_values();
    bret = nlp_in->eval_jac_g((Ipopt::Index)n_vars, x_in, new_x,
                              (Ipopt::Index)m_cons, nnz_jac, nullptr, nullptr, values);
    assert(bret);


    const double *lambda_data = lambda->local_data_const();
    /**************************************************/
    /**    Compute Lagrangian term contribution       */
    /**************************************************/
    //_penaltyFcn_jacobian->transTimesVec(beta, y, alpha, x)
    _penaltyFcn_jacobian->transTimesVec(1.0, gradLagr, 1.0, lambda_data);

    
    //Add the Jacobian w.r.t the slack variables (_penaltyFcn_jacobian contains
    //only the jacobian w.r.t original x, we need to add Jac w.r.t slacks)
    //The structure of the La Jacobian is following (Note that Je, Ji
    //might be interleaved and not listed in order as shown below)
    //       | Je  0  |
    // Jac = |        |
    //       | Ji  -I |
    for (long long i = 0; i<m_cons_ineq; i++)
    {
        gradLagr[n_vars + i] -= lambda_data[cons_ineq_mapping[i]];
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
    
    runStats.tmEvalGrad_f.start();

    /********************************************************/
    /** Add contribution of the gradient of the Lagrangian **/
    /********************************************************/
    eval_grad_Lagr(n, x_in, new_x, gradf);
    
    const double *_penaltyFcn_data = _penaltyFcn->local_data_const();
    /**************************************************/
    /**    Compute penalty term contribution          */
    /**************************************************/
    //_penaltyFcn_jacobian->transTimesVec(beta, y, alpha, x)
    _penaltyFcn_jacobian->transTimesVec(1.0, gradf, 2*rho, _penaltyFcn_data);

    //Add the Jacobian w.r.t the slack variables (_penaltyFcn_jacobian contains
    //only the jacobian w.r.t original x, we need to add Jac w.r.t slacks)
    //The structure of the La Jacobian is following (Note that Je, Ji
    //might be interleaved and not listed in order as shown below)
    //       | Je  0  |
    // Jac = |        |
    //       | Ji  -I |
    for (long long i = 0; i<m_cons_ineq; i++)
    {
        gradf[n_vars + i] -= 2*rho*_penaltyFcn_data[cons_ineq_mapping[i]];
    }

    runStats.tmEvalGrad_f.stop();
    runStats.nEvalGrad_f++;

    return true;
}

/**
 * The get method returns the value of the starting point x0
 * which was set from outside by the Solver and stored in #_startingPoint.
 * Motivation: every major iteration we want to reuse the previous
 * solution x_k, not start from the user point every time!!!
 */
bool hiopAugLagrNlpAdapter::get_starting_point(const long long &global_n, double* x0) const
{
    assert(global_n == n_vars+n_slacks);

    memcpy(x0, _startingPoint->local_data_const(), global_n*sizeof(double));
    return true;
}

/**
 * The set method stores the provided starting point into the private
 * member #_startingPoint
 * Motivation: every major iteration we want to reuse the previous
 * solution x_k, not start from the user point every time!!!
 */
bool hiopAugLagrNlpAdapter::set_starting_point(const long long &global_n, const double* x0_in)
{
    assert(global_n == n_vars+n_slacks);

    memcpy(_startingPoint->local_data(), x0_in, global_n*sizeof(double));
    return true;
}

/**
 * The method returns true (and populates x0 with user provided TNLP starting point)
 * or returns false, in which case hiOP will set x0 to all zero.
 */
bool hiopAugLagrNlpAdapter::get_user_starting_point(const long long &global_n, double* x0)
{
    assert(global_n == n_vars + n_slacks);

    //call starting point from the adapted nlp
    bool bret = nlp_in->get_starting_point((Ipopt::Index)n_vars, true, x0,
                                           false, nullptr, nullptr,
                                           (Ipopt::Index)m_cons, false, nullptr);
    
    //if no user point provided, set it in between bounds,
    //or close to the bound if bounded only from one side
    const double* xl_ = xl->local_data_const();
    const double* xu_ = xu->local_data_const();
    for (long long i = 0; i < n_vars; i++)
    {
        if (xl_[i] < -1e20)
            if(xu_[i] > 1e20)
                x0[i] = 0.; //unbounded
            else
                x0[i] = xu_[i]-1e-4; //close to U
        else
            if(xu_[i] > 1e20)
                x0[i] = xl_[i]+1e-4; //close to L
            else
                x0[i] = (xl_[i]+xu_[i])/2.; //in-between the bounds
    }
    //if (!bret) std::fill(x0, x0+n_vars, 0.); //probably not the best way
  
    //initialize slacks close to the bound or in the middle
    const double* sl_ = sl->local_data_const();
    const double* su_ = su->local_data_const();
    for (long long i = 0; i < n_slacks; i++)
    {
        if (sl_[i] < -1e20)
            if(su_[i] > 1e20)
                x0[i+n_vars] = 0.; //unbounded
            else
                x0[i+n_vars] = su_[i]-1e-4; //close to U
        else
            if(su_[i] > 1e20)
                x0[i+n_vars] = sl_[i]+1e-4; //close to L
            else
                x0[i+n_vars] = (sl_[i]+su_[i])/2.; //in-between the bounds
    }
    
    //initialization by zero is probably not the best way
    //std::fill(x0+n_vars, x0+n_vars+n_slacks, 0.);

    return bret;
}

void hiopAugLagrNlpAdapter::set_lambda(const hiopVectorPar* lambda_in)
{
    assert(lambda_in->get_size() == m_cons);
    assert(lambda_in->get_size() == lambda->get_size());

    memcpy(lambda->local_data(), lambda_in->local_data_const(), m_cons*sizeof(double));
}

//TODO: can we reuse the last jacobian instead of recomputing it? does hiop/ipopt evaluate the Jacobian in the last (xk + searchDir), a.k.a the solution? 
bool hiopAugLagrNlpAdapter::eval_residuals(const long long& n, const double* x_in, bool new_x, double *penalty, double* gradLagr)
{
   assert(n == n_vars+n_slacks);

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
    //TODO: we could use new_x = false here, since the penalty is already evaluated
    //and cached in #_penaltyFcn
    bret = eval_grad_Lagr(n, x_in, new_x, gradLagr);
    assert(bret);

    return bret;
}
}
