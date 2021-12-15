#include "hiopAugLagrNlpAdapter.hpp"
#include "hiopVector.hpp"
#include "hiopMatrixSparse.hpp"
#include "hiopAugLagrHessian.hpp"

#include "IpIpoptCalculatedQuantities.hpp"


#ifdef HIOP_USE_MPI
#include "mpi.h"
#else 
#include <cstddef>
#endif

#include <stdlib.h>     /* exit, EXIT_FAILURE */
#include <cstring>      // memcpy
#include <cassert>
#include <iostream>


namespace hiop
{

hiopAugLagrNlpAdapter::hiopAugLagrNlpAdapter(NLP_CLASS_IN* nlp_in_):
    nlp_in(nlp_in_),
    rho(-1), lambda(nullptr),
    startingPoint(nullptr),
    n_vars(-1), n_slacks(-1), m_cons(-1),
    m_cons_eq(-1), m_cons_ineq(-1),
    nnz_jac(-1),
    nnz_hess(-1),
    xl(nullptr),
    xu(nullptr),
    sl(nullptr),
    su(nullptr),
    cons_eq_mapping(nullptr),
    cons_ineq_mapping(nullptr),
    c_rhs(nullptr),
    _solutionIpopt(nullptr), _zLowIpopt(nullptr), _zUppIpopt(nullptr), _numItersIpopt(-1),
    _penaltyFcn(nullptr),
    _penaltyFcn_jacobian(nullptr),
    _hessian(nullptr),
    runStats(),
    options(new hiopOptions(/*filename=NULL*/)),
    log(new hiopLogger(options, stdout, 0))
{
    options->SetLog(log);
    nlp_error_ = 1e+6;
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

    if(lambda)                   delete lambda;
    if(startingPoint)            delete startingPoint;
    if(_penaltyFcn)              delete _penaltyFcn;
    if(_penaltyFcn_jacobian)     delete _penaltyFcn_jacobian;
    if(_hessian)                 delete _hessian;
    if(cons_eq_mapping)       delete [] cons_eq_mapping;
    if(cons_ineq_mapping)     delete [] cons_ineq_mapping;
    if(c_rhs)                 delete [] c_rhs;
    if(_solutionIpopt)        delete  _solutionIpopt;
    if(_zLowIpopt)            delete  _zLowIpopt;
    if(_zUppIpopt)            delete  _zUppIpopt;
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
    Ipopt::Index n_nlp, m_nlp, nnz_jac_nlp, nnz_hess_nlp;
    TNLP::IndexStyleEnum index_style = TNLP::C_STYLE;
    bool bret = nlp_in->get_nlp_info(n_nlp, m_nlp, nnz_jac_nlp, nnz_hess_nlp, index_style);
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
    m_cons_eq = m_cons_ineq = 0;
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
    nnz_hess = nnz_hess_nlp;

    //allocate space 
    sl = new hiopVectorPar(m_cons_ineq);
    su = new hiopVectorPar(m_cons_ineq);
    lambda = new hiopVectorPar(m_cons);
    startingPoint = new hiopVectorPar(n_vars+n_slacks);
    _penaltyFcn = new hiopVectorPar(m_cons);
    _penaltyFcn_jacobian = new hiopMatrixSparse(m_cons, n_vars, nnz_jac);
    _hessian = new hiopAugLagrHessian(nlp_in, n_vars, n_slacks, m_cons, nnz_hess);

    _solutionIpopt = new hiopVectorPar(n_vars+n_slacks);
    _zLowIpopt = new hiopVectorPar(n_vars+n_slacks);
    _zUppIpopt = new hiopVectorPar(n_vars+n_slacks);

    /**************************************************************************/
    /*  Analyze the original NLP constraints and split them into eq/ineq      */
    /**************************************************************************/
    double *sl_nlp=sl->local_data(), *su_nlp=su->local_data();
    
    //We need to store the eq/ineq mapping
    //because we need to evaluate c(x) in eval_f()
    //Constraint evaluations consist of:
    // ->  c(x) - c_rhs = 0 (eq. constr)
    // ->  c(x) - s     = 0 (ineq. constr)
    c_rhs = new double[m_cons_eq];
    cons_eq_mapping   = new long long[m_cons_eq];
    cons_ineq_mapping = new long long[m_cons_ineq];
  
    // copy lower and upper bounds of the constraints and get indices of eq/ineq constraints
    int it_eq=0, it_ineq=0;
    for(Ipopt::Index i=0; i<m_nlp; i++) {
      if(gl_nlp[i]==gu_nlp[i]) {
        c_rhs[it_eq] = gl_nlp[i]; 
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

    std::cout << "Initialized AugLagrNlpAdapter with the following values:" << std::endl;
    std::cout << "n_vars " << n_vars << std::endl;
    std::cout << "n_slacks " << n_slacks << std::endl;
    std::cout << "m_cons " << m_cons << std::endl;
    std::cout << "m_cons_eq " << m_cons_eq << std::endl;
    std::cout << "m_cons_ineq " << m_cons_ineq << std::endl;
    std::cout << "nnz_jac " << nnz_jac << std::endl;
    std::cout << "nnz_hess " << nnz_hess << std::endl;

    return true;
}

    /***********************************************************************
     * HiOP Interface (overloaded from hip::hiopInterfaceDenseConstraints) *
     ***********************************************************************/                        
/**
 * This function returns info about the Augmented Lagrangian
 * */
bool hiopAugLagrNlpAdapter::get_prob_sizes(long long& n, long long& m)
{ n=n_vars+n_slacks; m=0; return true; }

/**
 * This function returns info about vables of the Augmented Lagrangian
 * formulation, including the slacks.
 * The new variable vector is xlow <= x <= xup where x consists of the
 * original nlp variables and new slack variables x = [x_nlp, s]
 * */
bool hiopAugLagrNlpAdapter::get_vars_info(const long long& n, double *xlow,
     double* xupp, NonlinearityType* type)
{
    assert(n == n_vars + n_slacks);

    //fill in lower and upper bounds (including slack variables)
    xl->copyTo(xlow);
    xu->copyTo(xupp);
    sl->copyTo(xlow + n_vars);
    su->copyTo(xupp + n_vars);

    //set variable types (x - nonlinear, s - linear)
    if (type != nullptr)
    {
      for(long long i=0; i<n_vars; i++)  type[i] = hiopNonlinear;
      for(long long i=n_vars; i<n; i++)  type[i] = hiopNonlinear;
    }

    return true;
}

/**
 * Evaluates the original NLP constraints L <= c(x) <= U and transforms the constraints 
 * into the penalty form p(x) = 0 appropriate for Augmented Lagrangian formulation.
 * The penalty terms consist of:
 * Equality constraints:  c(x) - c_rhs
 * Inequality constraints c(x) - s, where L <= s <= U
 * The evaluated penalty function is stored in member #_penaltyFcn
 */
bool hiopAugLagrNlpAdapter::eval_penalty(const double *x_in, bool new_x)
{
    const double *slacks  = &x_in[n_vars];
    double *penalty_data = _penaltyFcn->local_data();

    //evaluate the original NLP constraints
    bool bret = nlp_in->eval_g((Ipopt::Index)n_vars, x_in, new_x,
                               (Ipopt::Index)m_cons, penalty_data);
    assert(bret);

    //adjust equality constraints
    // c(x) - c_rhs
    for (long long i = 0; i<m_cons_eq; i++)
    {
        penalty_data[cons_eq_mapping[i]] -= c_rhs[i]; 
    }
    
    //adjust inequality constraints
    // c(x) - s
    assert(n_slacks == m_cons_ineq);
    for (long long i = 0; i<m_cons_ineq; i++)
    {
        penalty_data[cons_ineq_mapping[i]] -= slacks[i]; 
    }

    return true;
}

/**
 *  Evaluates Jacobian of the penalty function. Jacobian is stored in the
 *  member #_penaltyFcn_jacobian. The sparse structure is initialized during
 *  the first call.
 */
bool hiopAugLagrNlpAdapter::eval_penalty_jac(const double *x_in, bool new_x)
{
    //initialize the nonzero structure only during the first call
    static bool initializedStructure = false;
    if (!initializedStructure)
    {
      //initialize the structure during the first call, later update only the values
      int *iRow = _penaltyFcn_jacobian->get_iRow();
      int *jCol = _penaltyFcn_jacobian->get_jCol();
      bool bret = nlp_in->eval_jac_g((Ipopt::Index)n_vars, nullptr, new_x,
                              (Ipopt::Index)m_cons, nnz_jac, iRow, jCol, nullptr);


      assert(bret);
      initializedStructure = true;
    }

    // evaluate the nonzeros at the current x
    double *values = _penaltyFcn_jacobian->get_values();
    bool bret = nlp_in->eval_jac_g((Ipopt::Index)n_vars, x_in, new_x,
                              (Ipopt::Index)m_cons, nnz_jac, nullptr, nullptr, values);
    assert(bret);
      
    return bret;
}

/** 
 * Objective function evaluation, this is the augmented lagrangian function
 * La(x,lambda,rho) = f(x) - lam^t p(x) + rho ||p(x)||^2.
 *
 * @param[in] n Number of variables in Augmented Lagrangian formulation
 * @param[in] x_in Variables consisting of original NLP variables and additional slacks
 * @param[in] new_x
 * @param[out] obj_value Returns Augmented lagrangian value La(x, lambda, rho)
 * */
bool hiopAugLagrNlpAdapter::eval_f(const long long& n, const double* x_in,
     bool new_x, double& obj_value)
{
    assert(n == n_vars + n_slacks);

    runStats.tmEvalObj.start();

    // evaluate the original NLP objective f(x), uses only firs n_vars entries of x_in
    double obj_nlp;
    bool bret = nlp_in->eval_f((Ipopt::Index)n_vars, x_in, new_x, obj_nlp); 
    assert(bret);

    // evaluate and transform the constraints of the original NLP
    eval_penalty(x_in, new_x);//TODO: new_x

    // compute lam^t p(x)
    assert(lambda->get_size() == _penaltyFcn->get_size());
    const double lagr_term = lambda->dotProductWith(*_penaltyFcn);
    
    // compute penalty term rho*||p(x)||^2
    const double penalty_term = rho * _penaltyFcn->dotProductWith(*_penaltyFcn);

    //f(x) - lam^t p(x) + rho ||p(x)||^2
    obj_value = obj_nlp - lagr_term + penalty_term;
    
    runStats.tmEvalObj.stop();
    runStats.nEvalObj++;

    return true;
}

/** Objective function evaluation, this is the user objective function f(x) */
bool hiopAugLagrNlpAdapter::eval_f_user(const long long& n, const double* x_in,
     bool new_x, double& obj_value)
{
    assert(n == n_vars + n_slacks);

    runStats.tmEvalObj.start();

    // evaluate the original NLP objective f(x), uses only firs n_vars entries of x_in
    bool bret = nlp_in->eval_f((Ipopt::Index)n_vars, x_in, new_x, obj_value); 
    assert(bret);
    runStats.tmEvalObj.stop();

    return bret;
}

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
bool hiopAugLagrNlpAdapter::eval_grad_Lagr(const long long& n, const double* x_in,
     bool new_x, double* gradLagr)
{
    //TODO new_x
    if (true) eval_penalty(x_in, new_x);
    
    /****************************************************/
    /** Add contribution of the NLP objective function **/
    /****************************************************/
    bool bret = nlp_in->eval_grad_f((Ipopt::Index)n_vars, x_in, new_x, gradLagr);
    assert(bret);
    std::fill(gradLagr+n_vars, gradLagr+n, 0.0); //clear grad w.r.t slacks

    /****************************************************/
    /* Evaluate Jacobian of the original NLP problem */
    /****************************************************/
    //TODO new_x
    if (true) eval_penalty_jac(x_in, new_x);
    
    /**************************************************/
    /**    Compute Lagrangian term contribution       */
    /**     gradLagr = gradLagr - Jac' * lambda       */ 
    /**************************************************/
    const double *lambda_data = lambda->local_data_const();
    //_penaltyFcn_jacobian->transTimesVec(beta, y, alpha, x)
    assert(_penaltyFcn_jacobian->m() == m_cons);
    assert(_penaltyFcn_jacobian->n() == n_vars);

    _penaltyFcn_jacobian->transTimesVec(1.0, gradLagr, -1.0, lambda_data);
    //Add the Jacobian w.r.t the slack variables (_penaltyFcn_jacobian contains
    //only the jacobian w.r.t original x, we need to add Jac w.r.t slacks)
    //The structure of the La Jacobian is following (Note that Je, Ji
    //might be interleaved and not listed in order as shown below)
    //       | Je  0  |           | Je'  Ji' |
    // Jac = |        |   Jac' =  |          |
    //       | Ji  -I |           |  0   -I  |
    for (long long i = 0; i<m_cons_ineq; i++)
    {
        // gradLagr(n_vars:end) = 0 - (-I) * lambda_ineq
        gradLagr[n_vars + i] += lambda_data[cons_ineq_mapping[i]];
    }
    return true;
}

/** Gradient of the Augmented Lagrangian function La(x,s)
 *  d_La/d_x = df_x - J^T lam + 2rho J^T p(x,s)
 *  d_La/d_s =  0   - (-I) lam[cons_ineq_mapping] + (-I)2rho*p[cons_ineq_mapping]
 *  where p(x,s) is a penalty fcn and rho is the penalty param and
 *  J is the Jacobian of the original NLP constraints.
 *
 * @param[in] n Number of variables in Augmented Lagrangian formulation
 * @param[in] x_in Variables consisting of original NLP variables and additional slacks
 * @param[in] new_x
 * @param[out] gradf Returns gradient of the Augmented Lagrangian function La(x, lambda, rho)
 * */
bool hiopAugLagrNlpAdapter::eval_grad_f(const long long& n, const double* x_in,
     bool new_x, double* gradf)
{
    
    runStats.tmEvalGrad_f.start();

    /********************************************************/
    /** Add contribution of the gradient of the Lagrangian **/
    /**          gradf = df_x - Jac' *  lam                **/ 
    /********************************************************/
    eval_grad_Lagr(n, x_in, new_x, gradf);//TODO: new_x
    
    const double *_penaltyFcn_data = _penaltyFcn->local_data_const();
    /**************************************************/
    /**    Compute penalty term contribution          */
    /**  gradf = gradf + 2rho * Jac' * _penaltyFcn    */
    /**************************************************/
    //_penaltyFcn_jacobian->transTimesVec(beta, y, alpha, x)
    _penaltyFcn_jacobian->transTimesVec(1.0, gradf, 2*rho, _penaltyFcn_data);

    //Add the Jacobian w.r.t the slack variables (_penaltyFcn_jacobian contains
    //only the jacobian w.r.t original x, we need to add Jac w.r.t slacks)
    //The structure of the La Jacobian is following (Note that Je, Ji
    //might be interleaved and not listed in order as shown below)
    //       | Je  0  |           | Je'  Ji' |
    // Jac = |        |   Jac' =  |          |
    //       | Ji  -I |           |  0   -I  |
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
 * which was set from outside by the Solver and stored in #startingPoint.
 * Motivation: every major iteration we want to reuse the previous
 * solution x_k, not start from the user point every time!!!
 */
bool hiopAugLagrNlpAdapter::get_starting_point(const long long &global_n, double* x0)
{
    assert(global_n == n_vars+n_slacks);

    //memcpy(x0, startingPoint->local_data_const(), global_n*sizeof(double));
    startingPoint->copyTo(x0);
    return true;
}

    /***********************************************************************
     *            IPOPT interface (overloaded from Ipopt::TNLP)            *
     ***********************************************************************/                        
bool hiopAugLagrNlpAdapter::get_nlp_info(Index& n, Index& m,
     Index& nnz_jac_g, Index& nnz_h_lag, IndexStyleEnum& index_style)
{
    //If n,m are passed directly to get_prob_size() we get an error because
    //the n,m are implicitly converted from Index (aka int) to long long,
    //which exists only as a temporary rvalue.
    //  error: invalid initialization of non-const reference of type 'long long int&'
    //  from an rvalue of type 'long long int'
    //       get_prob_sizes(n, m);

    //hiop::hiopInterfaceDenseConstraints
    long long n_, m_;
    get_prob_sizes(n_, m_);
    n = n_; m = m_;

    nnz_jac_g = 0;
    index_style = TNLP::C_STYLE;

    
    //TODO: this is not needed for Quasi-Newton
    // need to call dummy hessian assembly to determine structure and nnz count
    eval_penalty_jac(startingPoint->local_data(), true);//need to init Jac. structure
    _hessian->assemble(startingPoint->local_data(), true, 1.0,
                        *lambda, rho, *_penaltyFcn,
		       *_penaltyFcn_jacobian, cons_ineq_mapping);

    nnz_h_lag = _hessian->nnz();
    //nnz_h_lag = 0;
    return true;
}

bool hiopAugLagrNlpAdapter::get_bounds_info(Index n, Number* x_l, Number* x_u,
     Index   m, Number* g_l, Number* g_u)
{
    //hiop::hiopInterfaceDenseConstraints
    get_vars_info((long long)n, (double*) x_l, (double*) x_u, nullptr);

    assert(m==0);
    return true;
}

bool hiopAugLagrNlpAdapter::get_starting_point( Index n, bool init_x, Number* x,
     bool init_z, Number* z_L, Number* z_U,
     Index m, bool init_lambda, Number* lambda )
{
    //hiop::hiopInterfaceDenseConstraints
    get_starting_point((long long) n, (double*) x);

    // returned cached bound multipliers
    if (init_z) log->printf(hovWarning, "Adapter: Initializing z_L, z_U!\n");
    if (init_z) get_ipoptBoundMultipliers(z_L, z_U);

    assert(m==0);
    return true;
}

bool hiopAugLagrNlpAdapter::eval_f(Index n, const Number* x,
     bool new_x, Number& obj_value)
{
    //hiop::hiopInterfaceDenseConstraints
    eval_f((long long) n, (double*) x, new_x, obj_value);
    //obj_value: this is ok because 'Number' is alias for 'double'

    return true;
}

bool hiopAugLagrNlpAdapter::eval_grad_f(Index n, const Number* x,
     bool new_x, Number* grad_f)
{
    //hiop::hiopInterfaceDenseConstraints
    eval_grad_f((long long) n, (double *)x, new_x, (double *) grad_f);
    
    return true;
}

bool hiopAugLagrNlpAdapter::eval_g(Index n, const Number* x, bool new_x,
     Index m, Number* g)
{
    assert(m == 0);
    return true;
}

bool hiopAugLagrNlpAdapter::eval_jac_g(Index n, const Number* x, bool new_x,
     Index m, Index nele_jac,
     Index* iRow, Index* jCol, Number* values)
{
    assert(m == 0);
    return true;
}


bool hiopAugLagrNlpAdapter::eval_h(Index n, const Number* x, bool new_x, Number obj_factor,
     Index m, const Number* lambda_ipopt, bool new_lambda,
     Index nele_hess, Index* iRow, Index* jCol, Number* values)
{
  //return true;
    assert(n == n_vars+n_slacks);
    assert(m == 0);
    assert(nele_hess == _hessian->nnz());

    if (iRow != NULL && jCol != NULL)
    {
      // copy structure from _hessian
      _hessian->getStructure(iRow, jCol);

    }
    else if (values != NULL)
    {
      // Evaluate #_penaltyFcn and  #_penaltyFcn_jacobian
      bool bret = eval_penalty(x, new_x);//TODO: new_x
      assert(bret);
      bret = eval_penalty_jac(x, new_x);//TODO: new_x
      assert(bret);

      _hessian->assemble(x, new_x, obj_factor, *lambda, rho, *_penaltyFcn,
                         *_penaltyFcn_jacobian, cons_ineq_mapping);
      //fill values
      _hessian->getValues(values);
    }

    return true;
}

void hiopAugLagrNlpAdapter::finalize_solution(SolverReturn status, Index n,
     const Number* x, const Number* z_L,
     const Number* z_U, Index m, const Number* g,
     const Number* lambda, Number obj_value,
     const IpoptData* ip_data,
     IpoptCalculatedQuantities* ip_cq)
{
    //SUCCESS         
    //MAXITER_EXCEEDED        
    //CPUTIME_EXCEEDED        
    //STOP_AT_TINY_STEP       
    //STOP_AT_ACCEPTABLE_POINT        
    //LOCAL_INFEASIBILITY     
    //USER_REQUESTED_STOP     
    //FEASIBLE_POINT_FOUND    
    //DIVERGING_ITERATES      
    //RESTORATION_FAILURE     
    //ERROR_IN_STEP_COMPUTATION       
    //INVALID_NUMBER_DETECTED         
    //TOO_FEW_DEGREES_OF_FREEDOM      
    //INVALID_OPTION  
    //OUT_OF_MEMORY   
    //INTERNAL_ERROR  
    //UNASSIGNED 
    if(status != SUCCESS) log->printf(hovMaxVerbose, "hiopAugLagrNlpApadpter::finalize_solution was called but Ipopt status is different from SUCCESS. The solution might not be valid.\n");

    //cache the Ipopt solution
    assert(n == n_vars+n_slacks);
    assert(m == 0);
    memcpy(_solutionIpopt->local_data(), x, n*sizeof(double));

    //cache the multipliers for the bounds z_L, z_U
    memcpy(_zLowIpopt->local_data(), z_L, n*sizeof(double));
    memcpy(_zUppIpopt->local_data(), z_U, n*sizeof(double));

    //cache the number of IPOPT iterations
    _numItersIpopt = ip_data->iter_count();

    nlp_error_ = ip_cq->curr_nlp_error();
    return;
}
    /***********************************************************************
     *     Other routines providing access to the internal data            *
     ***********************************************************************/ 

/**
* This method returns the cached IPOPT solution. No Guarantee that the solution
* is correct or has been initialized, the user calling this function needs
* to make sure Ipopt has finished successfuly. Only then the valid solution
* will be returned.
*/
void hiopAugLagrNlpAdapter::get_ipoptSolution(double *x) const
{
    memcpy(x, _solutionIpopt->local_data_const(), (n_vars+n_slacks)*sizeof(double));
}

void hiopAugLagrNlpAdapter::get_ipoptBoundMultipliers(double *z_L, double *z_U) const
{
    memcpy(z_L, _zLowIpopt->local_data_const(), (n_vars+n_slacks)*sizeof(double));
    memcpy(z_U, _zUppIpopt->local_data_const(), (n_vars+n_slacks)*sizeof(double));
}

void hiopAugLagrNlpAdapter::get_dualScaling(double &sd)
{
    //sd = |lambda|_1 + |z_l|_1 + |z_u|+1 / (m+2n)
    const double sLam = lambda->onenorm();
    const double szL = _zLowIpopt->onenorm();
    const double szU = _zUppIpopt->onenorm();
    const double avgNorm = (sLam + szL + szU)/(m_cons+2*(n_vars+n_slacks));

    //sd = max(100, sd)/100
    sd = std::max(100., avgNorm)/100.;
}

int hiopAugLagrNlpAdapter::get_ipoptNumIters() const
{
    return _numItersIpopt;
}

/**
 * The set method stores the provided starting point into the private
 * member #startingPoint
 * Motivation: every major iteration we want to reuse the previous
 * solution x_k, not start from the user point every time!!!
 */
bool hiopAugLagrNlpAdapter::set_starting_point(const long long &global_n, const double* x0_in)
{
    assert(global_n == n_vars+n_slacks);

    memcpy(startingPoint->local_data(), x0_in, global_n*sizeof(double));
    return true;
}

/**
 * The method returns true (and populates x0 with user provided TNLP starting point, x0 and possibly lambda0)
 * or returns false, in which case hiOP will set x0 to all zero.
 */
bool hiopAugLagrNlpAdapter::get_user_starting_point(const long long &global_n, double* x0, bool init_lambda, double* lambda)
{
    assert(global_n == n_vars + n_slacks);
    
    if (init_lambda) log->printf(hovWarning, "get_user_starting_point: hiOp is asking also for the initial values of the multipliers.\n");

    //get starting point from the adapted NLP
    bool bret = nlp_in->get_starting_point((Ipopt::Index)n_vars, true, x0,
                                           false, nullptr, nullptr,
                                           (Ipopt::Index)m_cons, init_lambda, lambda);
    assert(bret); //user might not provide lambda even though he promised to do so in the option file 
    
    // use alternative x0 if not provided by the user
    if (!bret) std::fill(x0, x0+n_vars, 0.); // TODO: solve PF or similar
  
    //slack initialization by zero is probably not the best way
    //std::fill(x0+n_vars, x0+global_n, 0.);

    //initialize the slack variables by the ineq. constr. values
    double *penalty_data = _penaltyFcn->local_data();
    bret = nlp_in->eval_g((Ipopt::Index)n_vars, x0, true,
                               (Ipopt::Index)m_cons, penalty_data);
    assert(bret);

    assert(n_slacks == m_cons_ineq);
    double *s0 = &x0[n_vars];
    for (long long i = 0; i<m_cons_ineq; i++)
    {
        s0[i] = penalty_data[cons_ineq_mapping[i]]; 
    }

    return true;
}

void hiopAugLagrNlpAdapter::set_lambda(const hiopVectorPar* lambda_in)
{
    assert(lambda_in->get_size() == m_cons);
    assert(lambda_in->get_size() == lambda->get_size());

    //memcpy(lambda->local_data(), lambda_in->local_data_const(), m_cons*sizeof(double));
    lambda->copyFrom(*lambda_in);
}

//TODO: can we reuse the last jacobian instead of recomputing it? does hiop/ipopt evaluate the Jacobian in the last (xk + searchDir), a.k.a the solution? 
bool hiopAugLagrNlpAdapter::eval_residuals(const long long& n, const double* x_in, const double *zL_in, const double *zU_in, bool new_x, double *penalty, double* grad)
{
   assert(n == n_vars+n_slacks);

    /*****************************************/
    /*          Penalty Function             */
    /*        [ce(x) - c_rhs; ci(x) - s]     */
    /*****************************************/
    bool bret = eval_penalty(x_in, new_x); //TODO: new_x
    assert(bret);
    _penaltyFcn->copyTo(penalty);
   
    /**************************************************/
    /**          Compute the AL gradient              */
    /*  d_La/d_x = df_x - J^T lam + 2rho J^T p(x,s)   */
    /*  d_La/d_s =  0 - (-I) lam[cons_ineq_mapping]   */
    /*                + (-I)2rho*p[cons_ineq_mapping] */
    /**************************************************/
    //TODO: we could use new_x = false here, since the penalty is already evaluated
    //and cached in #_penaltyFcn
    bret = eval_grad_f(n, x_in, new_x, grad); assert(bret); //TODO: new_x 

    /**************************************************/
    /**          Compute the AL gradient              */
    /*  d_La/d_x = df_x - J^T lam                     */
    /*  d_La/d_s =  0 - (-I) lam[cons_ineq_mapping]   */
    /**************************************************/
    //TODO: we could use new_x = false here, since the penalty is already evaluated
    //and cached in #_penaltyFcn
    //eval_grad_Lagr(n, x_in, new_x, grad); //TODO: new_x

    // std::cerr << "grad_f = [";
    // for(int it=0; it<n_vars+n_slacks; it++)  fprintf(stderr, "%22.16e ; ", grad[it]);
    // std::cerr << "];";


    //add multipliers for the l < x < u bound constraints, z_L and z_U
    //TODO: multipliers are not ititialized during the inital evaluation
    //      of the error at iteration 0, they are obtain as a subproblem solution
    for (int i = 0; i < n; i++)
    {
      grad[i] += (zU_in[i] - zL_in[i]);
    }

    std::string name = "z_L.txt";
    //FILE *f33=fopen(name.c_str(),"w");
    //_zLowIpopt->print(stdout);
    //fclose(f33);

    name = "z_u.txt";
    //FILE *f331=fopen(name.c_str(),"w");
    //_zUppIpopt->print(stdout);
    //fclose(f331);

    //project gradient onto rectangular box [l,u]
    // remove gradient with respect to the fixed variables, such that l=u
    project_gradient(x_in, grad);

    return bret;
}

bool hiopAugLagrNlpAdapter::project_gradient(const double* x_in, double* grad)
{
    //const double EPS = options->GetNumeric("fixed_var_tolerance");
    const double EPS = 1e-8;

    const double *x_low = xl->local_data_const();
    const double *x_upp = xu->local_data_const();

    for (int i=0; i<n_vars; i++)
    {
        if (fabs(x_upp[i] - x_low[i]) < EPS) //x == lb
        {
            grad[i] = 0.;
        }
    }

    const double *s_in = &x_in[n_vars];
    double *grad_s = &grad[n_vars];
    const double *s_low = sl->local_data_const();
    const double *s_upp = su->local_data_const();

    for (int i=0; i<n_slacks; i++)
    {
        if (fabs(s_upp[i] - s_low[i]) < EPS) //s == lb
        {
            grad_s[i] = 0.;
        }
    }

    return true;
}
}
