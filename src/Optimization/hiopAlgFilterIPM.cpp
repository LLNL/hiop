// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory (LLNL).
// Written by Cosmin G. Petra, petra1@llnl.gov.
// LLNL-CODE-742473. All rights reserved.
//
// This file is part of HiOp. For details, see https://github.com/LLNL/hiop. HiOp
// is released under the BSD 3-clause license (https://opensource.org/licenses/BSD-3-Clause).
// Please also read "Additional BSD Notice" below.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
// i. Redistributions of source code must retain the above copyright notice, this list
// of conditions and the disclaimer below.
// ii. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the disclaimer (as noted below) in the documentation and/or
// other materials provided with the distribution.
// iii. Neither the name of the LLNS/LLNL nor the names of its contributors may be used to
// endorse or promote products derived from this software without specific prior written
// permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
// SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
// AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
// EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Additional BSD Notice
// 1. This notice is required to be provided under our contract with the U.S. Department
// of Energy (DOE). This work was produced at Lawrence Livermore National Laboratory under
// Contract No. DE-AC52-07NA27344 with the DOE.
// 2. Neither the United States Government nor Lawrence Livermore National Security, LLC
// nor any of their employees, makes any warranty, express or implied, or assumes any
// liability or responsibility for the accuracy, completeness, or usefulness of any
// information, apparatus, product, or process disclosed, or represents that its use would
// not infringe privately-owned rights.
// 3. Also, reference herein to any specific commercial products, process, or services by
// trade name, trademark, manufacturer or otherwise does not necessarily constitute or
// imply its endorsement, recommendation, or favoring by the United States Government or
// Lawrence Livermore National Security, LLC. The views and opinions of authors expressed
// herein do not necessarily state or reflect those of the United States Government or
// Lawrence Livermore National Security, LLC, and shall not be used for advertising or
// product endorsement purposes.
#include "hiopAlgFilterIPM.hpp"

#include "hiopKKTLinSys.hpp"
#include "hiopKKTLinSysDense.hpp"
#include "hiopKKTLinSysMDS.hpp"
#include "hiopKKTLinSysSparse.hpp"

#include "hiopCppStdUtils.hpp"

#include <cmath>
#include <cstring>
#include <cassert>

namespace hiop
{

hiopAlgFilterIPMBase::hiopAlgFilterIPMBase(hiopNlpFormulation* nlp_)
{
  nlp = nlp_;
  //force completion of the nlp's initialization
  nlp->finalizeInitialization();
  reloadOptions();

  it_curr = new hiopIterate(nlp);
  it_trial= it_curr->alloc_clone();
  dir     = it_curr->alloc_clone();

  if(nlp->options->GetString("KKTLinsys")=="full")
  {
    it_curr->selectPattern();
    it_trial->selectPattern(); 
    dir->selectPattern();
  }

  logbar = new hiopLogBarProblem(nlp);

  _f_nlp = _f_log = 0;
  _c = nlp->alloc_dual_eq_vec();
  _d = nlp->alloc_dual_ineq_vec();

  _grad_f  = nlp->alloc_primal_vec();
  _Jac_c   = nlp->alloc_Jac_c();
  _Jac_d   = nlp->alloc_Jac_d();

  _f_nlp_trial = _f_log_trial = 0;
  _c_trial = nlp->alloc_dual_eq_vec();
  _d_trial = nlp->alloc_dual_ineq_vec();

  _grad_f_trial  = nlp->alloc_primal_vec();
  _Jac_c_trial   = nlp->alloc_Jac_c();
  _Jac_d_trial   = nlp->alloc_Jac_d();

  _Hess_Lagr = nlp->alloc_Hess_Lagr();

  resid = new hiopResidual(nlp);
  resid_trial = new hiopResidual(nlp);

  //parameter based initialization
  if(duals_update_type==0) {
    dualsUpdate = nlp->alloc_duals_lsq_updater();
  } else if(duals_update_type==1) {
    dualsUpdate = new hiopDualsNewtonLinearUpdate(nlp);
  } else {
    assert(false && "duals_update_type has an unrecognized value");
  }

  resetSolverStatus();
}
void hiopAlgFilterIPMBase::destructorPart()
{
  if(it_curr)  delete it_curr;
  if(it_trial) delete it_trial;
  if(dir)      delete dir;

  if(_c)       delete _c;
  if(_d)       delete _d;
  if(_grad_f)  delete _grad_f;
  if(_Jac_c)   delete _Jac_c;
  if(_Jac_d)   delete _Jac_d;

  if(_Hess_Lagr) delete _Hess_Lagr;

  if(resid)    delete resid;

  if(_c_trial)       delete _c_trial;
  if(_d_trial)       delete _d_trial;
  if(_grad_f_trial)  delete _grad_f_trial;
  if(_Jac_c_trial)   delete _Jac_c_trial;
  if(_Jac_d_trial)   delete _Jac_d_trial;

  if(resid_trial)    delete resid_trial;

  if(logbar) delete logbar;

  if(dualsUpdate) delete dualsUpdate;
}
hiopAlgFilterIPMBase::~hiopAlgFilterIPMBase()
{
  if(it_curr)  delete it_curr;
  if(it_trial) delete it_trial;
  if(dir)      delete dir;

  if(_c)       delete _c;
  if(_d)       delete _d;
  if(_grad_f)  delete _grad_f;
  if(_Jac_c)   delete _Jac_c;
  if(_Jac_d)   delete _Jac_d;

  if(_Hess_Lagr) delete _Hess_Lagr;

  if(resid)    delete resid;

  if(_c_trial)       delete _c_trial;
  if(_d_trial)       delete _d_trial;
  if(_grad_f_trial)  delete _grad_f_trial;
  if(_Jac_c_trial)   delete _Jac_c_trial;
  if(_Jac_d_trial)   delete _Jac_d_trial;

  if(resid_trial)    delete resid_trial;

  if(logbar) delete logbar;

  if(dualsUpdate) delete dualsUpdate;
}

void hiopAlgFilterIPMBase::reInitializeNlpObjects()
{
  destructorPart();

  it_curr = new hiopIterate(nlp);
  it_trial= it_curr->alloc_clone();
  dir     = it_curr->alloc_clone();

  logbar = new hiopLogBarProblem(nlp);

  _f_nlp = _f_log = 0;
  _c = nlp->alloc_dual_eq_vec();
  _d = nlp->alloc_dual_ineq_vec();

  _grad_f  = nlp->alloc_primal_vec();
  _Jac_c   = nlp->alloc_Jac_c();
  _Jac_d   = nlp->alloc_Jac_d();

  _f_nlp_trial = _f_log_trial = 0;
  _c_trial = nlp->alloc_dual_eq_vec();
  _d_trial = nlp->alloc_dual_ineq_vec();

  _grad_f_trial  = nlp->alloc_primal_vec();
  _Jac_c_trial   = nlp->alloc_Jac_c();
  _Jac_d_trial   = nlp->alloc_Jac_d();

  _Hess_Lagr = nlp->alloc_Hess_Lagr();

  resid = new hiopResidual(nlp);
  resid_trial = new hiopResidual(nlp);

  //0 LSQ (default), 1 linear update (more stable)
  duals_update_type = nlp->options->GetString("duals_update_type")=="lsq"?0:1;
  //0 LSQ (default), 1 set to zero
  dualsInitializ = nlp->options->GetString("duals_init")=="lsq"?0:1;

  if(duals_update_type==0) {
    hiopNlpDenseConstraints* nlpd = dynamic_cast<hiopNlpDenseConstraints*>(nlp);
    if(NULL==nlpd) {
      duals_update_type = 1;
      dualsInitializ = 1;
      nlp->log->printf(hovWarning,
		       "Option duals_update_type=lsq not compatible with the requested NLP formulation and will "
		       "be set to duals_update_type=linear together with duals_init=zero\n");
    }
  }

  //parameter based initialization
  if(duals_update_type==0) {
    //lsq update
    //dualsUpdate = new hiopDualsLsqUpdate(nlp);
    dualsUpdate = nlp->alloc_duals_lsq_updater();
  } else {
    if(duals_update_type==1) {
      dualsUpdate = new hiopDualsNewtonLinearUpdate(nlp);
    } else { assert(false && "duals_update_type has an unrecognized value"); }
  }
}

void hiopAlgFilterIPMBase::reloadOptions()
{
  //algorithm parameters parameters
  mu0=_mu  = nlp->options->GetNumeric("mu0");
  kappa_mu = nlp->options->GetNumeric("kappa_mu");         //linear decrease factor
  theta_mu = nlp->options->GetNumeric("theta_mu");         //exponent for higher than linear decrease of mu
  tau_min  = nlp->options->GetNumeric("tau_min");          //min value for the fraction-to-the-boundary
  eps_tol  = nlp->options->GetNumeric("tolerance");        //absolute error for the nlp
  eps_rtol = nlp->options->GetNumeric("rel_tolerance");    //relative error (to errors for the initial point)
  kappa_eps= nlp->options->GetNumeric("kappa_eps");        //relative (to mu) error for the log barrier

  kappa1   = nlp->options->GetNumeric("kappa1");          //projection params for starting point (default 1e-2)
  kappa2   = nlp->options->GetNumeric("kappa2");
  p_smax   = nlp->options->GetNumeric("smax");            //threshold for the magnitude of the multipliers

  max_n_it  = nlp->options->GetInteger("max_iter");

  accep_n_it    = nlp->options->GetInteger("acceptable_iterations");
  eps_tol_accep = nlp->options->GetNumeric("acceptable_tolerance");

  //0 LSQ (default), 1 linear update (more stable)
  duals_update_type = nlp->options->GetString("duals_update_type")=="lsq"?0:1;
  //0 LSQ (default), 1 set to zero
  dualsInitializ = nlp->options->GetString("duals_init")=="lsq"?0:1;

  if(duals_update_type==0) {
    hiopNlpDenseConstraints* nlpd = dynamic_cast<hiopNlpDenseConstraints*>(nlp);
    if(NULL==nlpd){
      // this is sparse or mds linear algebra
      duals_update_type = 1;
      nlp->log->printf(hovWarning,
		       "Option duals_update_type=lsq not compatible with the requested NLP formulation. "
		       " Will use duals_update_type=linear.\n");
    }
  }

  gamma_theta = 1e-5; //sufficient progress parameters for the feasibility violation
  gamma_phi=1e-8;     //and log barrier objective
  s_theta=1.1;        //parameters in the switch condition of
  s_phi=2.3;          // the linearsearch (equation 19) in
  delta=1.;           // the WachterBiegler paper
  // parameter in the Armijo rule
  eta_phi=nlp->options->GetNumeric("eta_phi");
  kappa_Sigma = 1e10; //parameter in resetting the duals to guarantee closedness of the primal-dual logbar Hessian to the primal logbar Hessian
  _tau=fmax(tau_min,1.0-_mu);
  theta_max = 1e7; //temporary - will be updated after ini pt is computed
  theta_min = 1e7; //temporary - will be updated after ini pt is computed

  perf_report_kkt_ = "on"==hiop::tolower(nlp->options->GetString("time_kkt"));

  // Set memory space for computations
  hiop::LinearAlgebraFactory::set_mem_space(nlp->options->GetString("mem_space"));
}

void hiopAlgFilterIPMBase::resetSolverStatus()
{
  n_accep_iters_ = 0;
  solver_status_ = NlpSolve_IncompleteInit;
  filter.clear();
}

int hiopAlgFilterIPMBase::
startingProcedure(hiopIterate& it_ini,
		  double &f, hiopVector& c, hiopVector& d,
		  hiopVector& gradf,  hiopMatrix& Jac_c,  hiopMatrix& Jac_d)
{
  bool duals_avail = false;
  if(!nlp->get_starting_point(*it_ini.get_x(),
			      duals_avail,
			      *it_ini.get_zl(), *it_ini.get_zu(),
			      *it_ini.get_yc(), *it_ini.get_yd())) {

    nlp->log->printf(hovWarning, "user did not provide a starting point; will be set to all zeros\n");
    it_ini.get_x()->setToZero();
    //in case user wrongly set this to true when he/she returned false
    duals_avail = false;
  }

  nlp->runStats.tmSolverInternal.start();
  nlp->runStats.tmStartingPoint.start();

  it_ini.projectPrimalsXIntoBounds(kappa1, kappa2);

  nlp->runStats.tmStartingPoint.stop();
  nlp->runStats.tmSolverInternal.stop();

  // before evaluating the NLP, make sure that iterate, including dual variables are initialized
  // to zero; many of these will be updated later in this method, but we want to make sure
  // that the user's NLP evaluator functions, in particular the Hessian of the Lagrangian,
  // receives initialized arrays


  if(!duals_avail) {
    // initialization for yc and yd
    it_ini.setEqualityDualsToConstant(0.);
  } else {
    // yc and yd were provided by the user
  }

  if(!this->evalNlp_noHess(it_ini, f, c, d, gradf, Jac_c, Jac_d)) {
    nlp->log->printf(hovError, "Failure in evaluating user provided NLP functions.");
    assert(false);
    return false;
  }
  
  if(nlp->apply_scaling(c, d, gradf, Jac_c, Jac_d)){
    // do function evaluation again after add scaling 
    if(!this->evalNlp_noHess(it_ini, f, c, d, gradf, Jac_c, Jac_d)) {
      nlp->log->printf(hovError, "Failure in evaluating user provided NLP functions.");
      assert(false);
      return false;
    }
  }

  nlp->runStats.tmSolverInternal.start();
  nlp->runStats.tmStartingPoint.start();

  it_ini.get_d()->copyFrom(d);

  it_ini.projectPrimalsDIntoBounds(kappa1, kappa2);

  it_ini.determineSlacks();

  if(!duals_avail) {
    // initialization for zl, zu, vl, vu
    it_ini.setBoundsDualsToConstant(1.);
  } else {
    // zl and zu were provided by the user

    // compute vl and vu from vl = mu e ./ sdl and vu = mu e ./ sdu
    // sdl and sdu were initialized above in 'determineSlacks'

    it_ini.determineDualsBounds_d(mu0);
  }

  if(!duals_avail) {
    if(0==dualsInitializ) {
      //LSQ-based initialization of yc and yd

      //is the dualsUpdate already the LSQ-based updater?
      hiopDualsLsqUpdate* updater = dynamic_cast<hiopDualsLsqUpdate*>(dualsUpdate);
      bool deleteUpdater = false;
      if(updater == NULL) {
	//updater = new hiopDualsLsqUpdate(nlp);
        updater = nlp->alloc_duals_lsq_updater();
	deleteUpdater = true;
      }

      //this will update yc and yd in it_ini
      updater->computeInitialDualsEq(it_ini, gradf, Jac_c, Jac_d);

      if(deleteUpdater) delete updater;
    } else {
      it_ini.setEqualityDualsToConstant(0.);
    }
  } // end of if(!duals_avail)
  else {
    // duals eq ('yc' and 'yd') were provided by the user
  }

  //we have the duals
  if(!this->evalNlp_HessOnly(it_ini, *_Hess_Lagr)) {
    assert(false);
    return false;
  }

  nlp->log->write("Using initial point:", it_ini, hovIteration);
  nlp->runStats.tmStartingPoint.stop();
  nlp->runStats.tmSolverInternal.stop();

  solver_status_ = NlpSolve_SolveNotCalled;

  return true;
}

bool hiopAlgFilterIPMBase::
evalNlp(hiopIterate& iter,
	double &f, hiopVector& c, hiopVector& d,
	hiopVector& gradf,  hiopMatrix& Jac_c,  hiopMatrix& Jac_d,
	hiopMatrix& Hess_L)
{
  bool new_x=true;
  // hiopVector& it_x = *iter.get_x();
  // double* x = it_x.local_data();//local_data_const();
  // //f(x)
  // if(!nlp->eval_f(x, new_x, f)) {
  hiopVector& x = *iter.get_x();
  //f(x)
  if(!nlp->eval_f(x, new_x, f)) {
    nlp->log->printf(hovError, "Error occured in user objective evaluation\n");
    return false;
  }
  new_x= false; //same x for the rest

  if(!nlp->eval_grad_f(x, new_x, gradf)) {
    nlp->log->printf(hovError, "Error occured in user gradient evaluation\n");
    return false;
  }

  //bret = nlp->eval_c        (x, new_x, c.local_data());  assert(bret);
  //bret = nlp->eval_d        (x, new_x, d.local_data());  assert(bret);
  if(!nlp->eval_c_d(x, new_x, c, d)) {
    nlp->log->printf(hovError, "Error occured in user constraint(s) function evaluation\n");
    return false;
  }

  //nlp->log->write("Eq   body c:", c, hovFcnEval);
  //nlp->log->write("Ineq body d:", d, hovFcnEval);

  //bret = nlp->eval_Jac_c    (x, new_x, Jac_c);           assert(bret);
  //bret = nlp->eval_Jac_d    (x, new_x, Jac_d);           assert(bret);
  if(!nlp->eval_Jac_c_d(x, new_x, Jac_c, Jac_d)) {
    nlp->log->printf(hovError, "Error occured in user Jacobian function evaluation\n");
    return false;
  }
  const hiopVector* yc = iter.get_yc(); assert(yc);
  const hiopVector* yd = iter.get_yd(); assert(yd);
  const int new_lambda = true;

  if(!nlp->eval_Hess_Lagr(x, new_x,
			  1., *yc, *yd, new_lambda,
			  Hess_L)) {
    nlp->log->printf(hovError, "Error occured in user Hessian function evaluation\n");
    return false;
  }
  return true;
}

bool hiopAlgFilterIPMBase::
evalNlp_noHess(hiopIterate& iter,
	       double &f, hiopVector& c, hiopVector& d,
	       hiopVector& gradf,  hiopMatrix& Jac_c,  hiopMatrix& Jac_d)
{
  bool new_x=true;
  //hiopVectorPar& it_x = dynamic_cast<hiopVectorPar&>(*iter.get_x());
  //hiopVectorPar& c=dynamic_cast<hiopVectorPar&>(c_);
  //hiopVectorPar& d=dynamic_cast<hiopVectorPar&>(d_);
  //hiopVectorPar& gradf=dynamic_cast<hiopVectorPar&>(gradf_);
  hiopVector& x = *iter.get_x();
  //f(x)
  if(!nlp->eval_f(x, new_x, f)) {
    nlp->log->printf(hovError, "Error occured in user objective evaluation\n");
    return false;
  }
  new_x= false; //same x for the rest

  if(!nlp->eval_grad_f(x, new_x, gradf)) {
    nlp->log->printf(hovError, "Error occured in user gradient evaluation\n");
    return false;
  }

  //bret = nlp->eval_c        (x, new_x, c.local_data());  assert(bret);
  //bret = nlp->eval_d        (x, new_x, d.local_data());  assert(bret);
  if(!nlp->eval_c_d(x, new_x, c, d)) {
    nlp->log->printf(hovError, "Error occured in user constraint(s) function evaluation\n");
    return false;
  }

  //nlp->log->write("Eq   body c:", c, hovFcnEval);
  //nlp->log->write("Ineq body d:", d, hovFcnEval);

  //bret = nlp->eval_Jac_c    (x, new_x, Jac_c);           assert(bret);
  //bret = nlp->eval_Jac_d    (x, new_x, Jac_d);           assert(bret);
  if(!nlp->eval_Jac_c_d(x, new_x, Jac_c, Jac_d)) {
    nlp->log->printf(hovError, "Error occured in user Jacobian function evaluation\n");
    return false;
  }
  return true;
}

bool hiopAlgFilterIPMBase::evalNlp_HessOnly(hiopIterate& iter,
					    hiopMatrix& Hess_L)
{
  const bool new_x = false; //precondition is that 'evalNlp_noHess' was called just before

  const hiopVector* yc = iter.get_yc(); assert(yc);
  const hiopVector* yd = iter.get_yd(); assert(yd);
  const int new_lambda = true;

  hiopVector& x = *iter.get_x();
  if(!nlp->eval_Hess_Lagr(x, new_x,
			  1., *yc, *yd, new_lambda,
			  Hess_L)) {
    nlp->log->printf(hovError, "Error occured in user Hessian function evaluation\n");
    return false;
  }
  return true;
}

bool hiopAlgFilterIPMBase::
updateLogBarrierParameters(const hiopIterate& it, const double& mu_curr, const double& tau_curr,
			   double& mu_new, double& tau_new)
{
  double new_mu = fmax(eps_tol/10, fmin(kappa_mu*mu_curr, pow(mu_curr,theta_mu)));
  if(fabs(new_mu-mu_curr)<1e-16) return false;
  mu_new  = new_mu;
  tau_new = fmax(tau_min,1.0-mu_new);
  return true;
}

double hiopAlgFilterIPMBase::thetaLogBarrier(const hiopIterate& it, const hiopResidual& resid, const double& mu)
{
  //actual nlp errors
  double optim, feas, complem;
  resid.getNlpErrors(optim, feas, complem);
  return feas;
}


bool hiopAlgFilterIPMBase::
evalNlpAndLogErrors(const hiopIterate& it, const hiopResidual& resid, const double& mu,
		    double& nlpoptim, double& nlpfeas, double& nlpcomplem, double& nlpoverall,
		    double& logoptim, double& logfeas, double& logcomplem, double& logoverall)
{
  nlp->runStats.tmSolverInternal.start();

  long long n=nlp->n_complem(), m=nlp->m();
  //the one norms
  //double nrmDualBou=it.normOneOfBoundDuals();
  //double nrmDualEqu=it.normOneOfEqualityDuals();
  double nrmDualBou, nrmDualEqu;
  it.normOneOfDuals(nrmDualEqu, nrmDualBou);

  nlp->log->printf(hovScalars, "nrmOneDualEqu %g   nrmOneDualBo %g\n", nrmDualEqu, nrmDualBou);
  if(nrmDualBou>1e+10) {
    nlp->log->printf(hovWarning, "Unusually large bound dual variables (norm1=%g) occured, "
		     "which may cause numerical instabilities if it persists. Convergence "
		     " issues or inacurate optimal solutions may be experienced. Possible causes: "
		     " tight bounds or bad scaling of the optimization variables.\n",
		     nrmDualBou);
    if(nlp->options->GetString("fixed_var")=="remove") {
      nlp->log->printf(hovWarning, "For example, increase 'fixed_var_tolerance' to remove "
		       "additional variables.\n");
    } else {
      if(nlp->options->GetString("fixed_var")=="relax") {
	nlp->log->printf(hovWarning, "For example, increase 'fixed_var_tolerance' to relax "
			 "aditional (tight) variables and/or increase 'fixed_var_perturb' "
			 "to decrease the tightness.\n");
      } else
	nlp->log->printf(hovWarning, "Potential fixes: fix or relax variables with tight bounds "
			 "(see 'fixed_var' option) or rescale variables.\n");
    }
  }

  //scaling factors
  double sd = fmax(p_smax,(nrmDualBou+nrmDualEqu)/(n+m)) / p_smax;
  double sc = n==0?0:fmax(p_smax,nrmDualBou/n) / p_smax;

  sd = fmin(sd, 1e+8);
  sc = fmin(sc, 1e+8);

  //actual nlp errors
  resid.getNlpErrors(nlpoptim, nlpfeas, nlpcomplem);

  //finally, the scaled nlp error
  nlpoverall = fmax(nlpoptim/sd, fmax(nlpfeas, nlpcomplem/sc));

  nlp->log->printf(hovScalars,
		   "nlpoverall %g  nloptim %g  sd %g  nlpfeas %g  nlpcomplem %g  sc %g\n",
		   nlpoverall, nlpoptim, sd, nlpfeas, nlpcomplem, sc);

  //actual log errors
  resid.getBarrierErrors(logoptim, logfeas, logcomplem);

  //finally, the scaled barrier error
  logoverall = fmax(logoptim/sd, fmax(logfeas, logcomplem/sc));
  nlp->runStats.tmSolverInternal.start();
  return true;
}

bool hiopAlgFilterIPMBase::evalNlp_funcOnly(hiopIterate& iter,
					    double& f, hiopVector& c, hiopVector& d)
{
  bool new_x=true;
  // hiopVector& it_x = *iter.get_x();
  // double* x = it_x.local_data();
  // if(!nlp->eval_f(x, new_x, f)) {
  hiopVector& x = *iter.get_x();
  if(!nlp->eval_f(x, new_x, f)) {
    nlp->log->printf(hovError, "Error occured in user objective evaluation\n");
    return false;
  }
  new_x= false; //same x for the rest
  if(!nlp->eval_c_d(x, new_x, c, d)) {
    nlp->log->printf(hovError, "Error occured in user constraint(s) function evaluation\n");
    return false;
  }
  return true;
}

bool hiopAlgFilterIPMBase::evalNlp_derivOnly(hiopIterate& iter,
					     hiopVector& gradf,
					     hiopMatrix& Jac_c,
					     hiopMatrix& Jac_d,
					     hiopMatrix& Hess_L)
{
  bool new_x=false; //functions were previously evaluated in the line search
  // hiopVector& it_x = *iter.get_x();
  // double* x = it_x.local_data();
  hiopVector& x = *iter.get_x();
  if(!nlp->eval_grad_f(x, new_x, gradf)) {
    nlp->log->printf(hovError, "Error occured in user gradient evaluation\n");
    return false;
  }
  if(!nlp->eval_Jac_c_d(x, new_x, Jac_c, Jac_d)) {
    nlp->log->printf(hovError, "Error occured in user Jacobian function evaluation\n");
    return false;
  }

  const hiopVector* yc = iter.get_yc(); assert(yc);
  const hiopVector* yd = iter.get_yd(); assert(yd);
  const int new_lambda = true;
  if(!nlp->eval_Hess_Lagr(x, new_x,
			  1., *yc, *yd, new_lambda,
			  Hess_L)) {
    nlp->log->printf(hovError, "Error occured in user Hessian function evaluation\n");
    return false;
  }
  return true;
}

/* returns the objective value; valid only after 'run' method has been called */
double hiopAlgFilterIPMBase::getObjective() const
{
  if(solver_status_==NlpSolve_IncompleteInit || solver_status_ == NlpSolve_SolveNotCalled) {
    nlp->log->
      printf(hovError, "getObjective: HiOp did not initialize entirely or the 'run' function was not called.");
  }
  if(solver_status_==NlpSolve_Pending) {
    nlp->log->
      printf(hovWarning, "getObjective: HiOp has not completed and objective value may not be optimal.");
  }
  return nlp->user_obj(_f_nlp);
}
/* returns the primal vector x; valid only after 'run' method has been called */
void hiopAlgFilterIPMBase::getSolution(double* x) const
{
  if(solver_status_==NlpSolve_IncompleteInit || solver_status_ == NlpSolve_SolveNotCalled) {
    nlp->log->
      printf(hovError, "getSolution: HiOp did not initialize entirely or the 'run' function was not called.");
  }
  if(solver_status_==NlpSolve_Pending) {
    nlp->log->
      printf(hovWarning, "getSolution: HiOp has not completed yet and solution returned may not be optimal.");
  }
  hiopVector& it_x = *it_curr->get_x();
  //it_curr->get_x()->copyTo(x);
  nlp->user_x(it_x, x);
}

void hiopAlgFilterIPMBase::getDualSolutions(double* zl_a, double* zu_a, double* lambda_a)
{
  if(solver_status_==NlpSolve_IncompleteInit || solver_status_ == NlpSolve_SolveNotCalled) {
    nlp->log->
      printf(hovError,
	     "getDualSolutions: HiOp did not initialize entirely or the 'run' function was not called.");
  }
  if(solver_status_==NlpSolve_Pending) {
    nlp->log->
      printf(hovWarning,
	     "getSolution: HiOp has not completed yet and solution returned may not be optimal.");
  }
  hiopVector& zl = *it_curr->get_zl();
  hiopVector& zu = *it_curr->get_zu();

  nlp->get_dual_solutions(*it_curr, zl_a, zu_a, lambda_a);
}

int hiopAlgFilterIPMBase::getNumIterations() const
{
  if(solver_status_==NlpSolve_IncompleteInit || solver_status_ == NlpSolve_SolveNotCalled)
    nlp->log->
      printf(hovError, "getNumIterations: HiOp did not initialize or the 'run' function was not called.");
  if(solver_status_==NlpSolve_Pending)
    nlp->log->
      printf(hovWarning, "getNumIterations: HiOp has not completed upon this call of 'getNumIterations'");
  return nlp->runStats.nIter;
}


bool hiopAlgFilterIPMBase::
checkTermination(const double& err_nlp, const int& iter_num, hiopSolveStatus& status)
{
  if(err_nlp<=eps_tol)   { solver_status_ = Solve_Success;     return true; }
  if(iter_num>=max_n_it) { solver_status_ = Max_Iter_Exceeded; return true; }

  if(eps_rtol>0) {
    if(_err_nlp_optim   <= eps_rtol * _err_nlp_optim0 &&
       _err_nlp_optim   <= eps_rtol * _err_nlp_optim0 &&
       _err_nlp_complem <= std::max(eps_rtol,1e-6) * std::min(1.,_err_nlp_complem0)) {
      solver_status_ = Solve_Success_RelTol;
      return true;
    }
  }

  if(err_nlp<=eps_tol_accep) n_accep_iters_++;
  else n_accep_iters_ = 0;

  if(n_accep_iters_>=accep_n_it) { solver_status_ = Solve_Acceptable_Level; return true; }

  return false;
}
/***** Termination message *****/
void hiopAlgFilterIPMBase::displayTerminationMsg()
{
  std::string strStatsReport = nlp->runStats.get_summary() + nlp->runStats.kkt.get_summary_total();
  switch(solver_status_) {
  case Solve_Success:
    {
      nlp->log->printf(hovSummary, "Successfull termination.\n%s\n", strStatsReport.c_str());
      break;
    }
  case Solve_Success_RelTol:
    {
      nlp->log->printf(hovSummary,
		       "Successfull termination (error within the relative tolerance).\n%s\n",
		        strStatsReport.c_str());
      break;
    }
  case Solve_Acceptable_Level:
    {
      nlp->log->printf(hovSummary,
		       "Solve to only to the acceptable tolerance(s).\n%s\n",
		       strStatsReport.c_str());
      break;
    }
  case Max_Iter_Exceeded:
    {
      nlp->log->printf(hovSummary,
		       "Maximum number of iterations reached.\n%s\n",
		       strStatsReport.c_str());//nlp->runStats.getSummary().c_str());
      break;
    }
  case Steplength_Too_Small:
    {
      nlp->log->printf(hovSummary, "Couldn't solve the problem.\n");
      nlp->log->printf(hovSummary, "Linesearch returned unsuccessfully (small step). Probable cause: "
		       "inaccurate gradients/Jacobians or infeasible problem.\n");
      nlp->log->printf(hovSummary, "%s\n", strStatsReport.c_str());
      break;
    }
  case User_Stopped:
    {
      nlp->log->printf(hovSummary,
		       "Stopped by the user through the user provided iterate callback.\n%s\n",
		       strStatsReport.c_str());
      break;
    }
  default:
    {
      nlp->log->printf(hovSummary, "Do not know why HiOp stopped. This shouldn't happen. :)\n%s\n",
		       strStatsReport.c_str());
      assert(false && "Do not know why hiop stopped. This shouldn't happen.");
      break;
    }
  };
}








hiopAlgFilterIPMQuasiNewton::hiopAlgFilterIPMQuasiNewton(hiopNlpDenseConstraints* nlp_)
  : hiopAlgFilterIPMBase(nlp_)
{
  nlpdc = nlp_;
  //_Hess = new hiopHessianLowRank(nlpdc, nlpdc->options->GetInteger("secant_memory_len"));
}

hiopAlgFilterIPMQuasiNewton::~hiopAlgFilterIPMQuasiNewton()
{
  //if(_Hess) delete _Hess;
}

hiopSolveStatus hiopAlgFilterIPMQuasiNewton::run()
{
  //hiopNlpFormulation nlp may need an update since user may have changed options and
  //reruning with the same hiopAlgFilterIPMNewton instance
  nlp->finalizeInitialization();
  //also reload options
  reloadOptions();

  //if nlp changed internally, we need to reinitialize 'this'
  if(it_curr->get_x()->get_size()!=nlp->n() ||
     //Jac_c->get_local_size_n()!=nlpdc->n_local()) { <- this is prone to racing conditions
     _Jac_c->n()!=nlpdc->n()) {
    //size of the nlp changed internally ->  reInitializeNlpObjects();
    reInitializeNlpObjects();
  }
  resetSolverStatus();

  //types of linear algebra objects are known now
  hiopMatrixDense* Jac_c = dynamic_cast<hiopMatrixDense*>(_Jac_c);
  hiopMatrixDense* Jac_d = dynamic_cast<hiopMatrixDense*>(_Jac_d);
  hiopHessianLowRank* Hess = dynamic_cast<hiopHessianLowRank*>(_Hess_Lagr);

  nlp->runStats.initialize();
  ////////////////////////////////////////////////////////////////////////////////////
  // run baby run
  ////////////////////////////////////////////////////////////////////////////////////

  nlp->log->printf(hovSummary, "===============\nHiop SOLVER\n===============\n");
  nlp->log->write(NULL, *nlp->options, hovSummary);

#ifdef HIOP_USE_MPI
  nlp->log->printf(hovSummary, "Using %d MPI ranks.\n", nlp->get_num_ranks());
#endif
  nlp->log->write("---------------\nProblem Summary\n---------------", *nlp, hovSummary);

  nlp->runStats.tmOptimizTotal.start();

  startingProcedure(*it_curr, _f_nlp, *_c, *_d, *_grad_f, *_Jac_c, *_Jac_d); //this also evaluates the nlp
  _mu=mu0;

  //update log bar
  logbar->updateWithNlpInfo(*it_curr, _mu, _f_nlp, *_c, *_d, *_grad_f, *_Jac_c, *_Jac_d);
  nlp->log->printf(hovScalars, "log bar obj: %g\n", logbar->f_logbar);
  //recompute the residuals
  resid->update(*it_curr,_f_nlp, *_c, *_d,*_grad_f,*_Jac_c,*_Jac_d, *logbar);

  nlp->log->write("First residual-------------", *resid, hovIteration);

  iter_num=0; nlp->runStats.nIter=iter_num;

  theta_max=1e+4*fmax(1.0,resid->get_theta());
  theta_min=1e-4*fmax(1.0,resid->get_theta());

  hiopKKTLinSysLowRank* kkt=new hiopKKTLinSysLowRank(nlp);

  _alpha_primal = _alpha_dual = 0;

  _err_nlp_optim0=-1.; _err_nlp_feas0=-1.; _err_nlp_complem0=-1;

  // --- Algorithm status 'algStatus ----
  //-1 couldn't solve the problem (most likely because small search step. Restauration phase likely needed)
  // 0 stopped due to tolerances, including acceptable tolerance, or relative tolerance
  // 1 max iter reached
  // 2 user stop via the iteration callback

  //int algStatus=0;
  bool bret=true; int lsStatus=-1, lsNum=0;
  int num_adjusted_bounds = 0;
  solver_status_ = NlpSolve_Pending;
  while(true) {

    bret = evalNlpAndLogErrors(*it_curr, *resid, _mu,
			       _err_nlp_optim, _err_nlp_feas, _err_nlp_complem, _err_nlp,
			       _err_log_optim, _err_log_feas, _err_log_complem, _err_log);
    if(!bret) {
      solver_status_ = Error_In_User_Function;
      return Error_In_User_Function;
    }

    nlp->log->printf(hovScalars,
		     "  Nlp    errs: pr-infeas:%23.17e   dual-infeas:%23.17e  comp:%23.17e  overall:%23.17e\n",
		     _err_nlp_feas, _err_nlp_optim, _err_nlp_complem, _err_nlp);
    nlp->log->printf(hovScalars,
		     "  LogBar errs: pr-infeas:%23.17e   dual-infeas:%23.17e  comp:%23.17e  overall:%23.17e\n",
		     _err_log_feas, _err_log_optim, _err_log_complem, _err_log);
    outputIteration(lsStatus, lsNum);

    if(_err_nlp_optim0<0) { // && _err_nlp_feas0<0 && _err_nlp_complem0<0
      _err_nlp_optim0=_err_nlp_optim; _err_nlp_feas0=_err_nlp_feas; _err_nlp_complem0=_err_nlp_complem;
    }

    //user callback
    if(!nlp->user_callback_iterate(iter_num, _f_nlp,
				   *it_curr->get_x(),
				   *it_curr->get_zl(),
				   *it_curr->get_zu(),
				   *_c,*_d,
				   *it_curr->get_yc(),  *it_curr->get_yd(), //lambda,
				   _err_nlp_feas, _err_nlp_optim,
				   _mu,
				   _alpha_dual, _alpha_primal,  lsNum)) {
      solver_status_ = User_Stopped; break;
    }

    /*************************************************
     * Termination check
     ************************************************/
    if(checkTermination(_err_nlp, iter_num, solver_status_)) {
      break;
    }
    if(NlpSolve_Pending!=solver_status_) break; //failure of the line search or user stopped.

    /************************************************
     * update mu and other parameters
     ************************************************/
    while(_err_log<=kappa_eps * _mu) {
      //update mu and tau (fraction-to-boundary)
      bret = updateLogBarrierParameters(*it_curr, _mu, _tau, _mu, _tau);
      if(!bret) break; //no update is necessary
      nlp->log->printf(hovScalars, "Iter[%d] barrier params reduced: mu=%g tau=%g\n", iter_num, _mu, _tau);

      //update only logbar problem  and residual (the NLP didn't change)
      logbar->updateWithNlpInfo(*it_curr, _mu, _f_nlp, *_c, *_d, *_grad_f, *_Jac_c, *_Jac_d);

      //! should perform only a partial update since NLP didn't change
      resid->update(*it_curr,_f_nlp, *_c, *_d,*_grad_f,*_Jac_c,*_Jac_d, *logbar);
      bret = evalNlpAndLogErrors(*it_curr, *resid, _mu,
				 _err_nlp_optim, _err_nlp_feas, _err_nlp_complem, _err_nlp,
				 _err_log_optim, _err_log_feas, _err_log_complem, _err_log);
      if(!bret) {
	solver_status_ = Error_In_User_Function;
	return Error_In_User_Function;
      }
      nlp->log->printf(hovScalars,
		       "  Nlp    errs: pr-infeas:%23.17e   dual-infeas:%23.17e  comp:%23.17e  overall:%23.17e\n",
		       _err_nlp_feas, _err_nlp_optim, _err_nlp_complem, _err_nlp);
      nlp->log->printf(hovScalars,
		       "  LogBar errs: pr-infeas:%23.17e   dual-infeas:%23.17e  comp:%23.17e  overall:%23.17e\n",
		       _err_log_feas, _err_log_optim, _err_log_complem, _err_log);

      filter.reinitialize(theta_max);
      //recheck residuals at the first iteration in case the starting pt is  very good
      //if(iter_num==0) {
      //	continue;
      //}
    }
    nlp->log->printf(hovScalars, "Iter[%d] logbarObj=%23.17e (mu=%12.5e)\n", iter_num, logbar->f_logbar,_mu);
    /****************************************************
     * Search direction calculation
     ***************************************************/
    //first update the Hessian and kkt system
    Hess->update(*it_curr,*_grad_f,*_Jac_c,*_Jac_d);
    kkt->update(it_curr, _grad_f, Jac_c, Jac_d, Hess);
    bret = kkt->computeDirections(resid,dir); assert(bret==true);

    nlp->log->printf(hovIteration, "Iter[%d] full search direction -------------\n", iter_num);
    nlp->log->write("", *dir, hovIteration);
    /***************************************************************
     * backtracking line search
     ****************************************************************/
    nlp->runStats.tmSolverInternal.start();

    //maximum  step
    bret = it_curr->fractionToTheBdry(*dir, _tau, _alpha_primal, _alpha_dual); assert(bret);
    double theta = resid->getInfeasInfNorm(); //at it_curr
    double theta_trial;
    nlp->runStats.tmSolverInternal.stop();

    //lsStatus: line search status for the accepted trial point. Needed to update the filter
    //-1 uninitialized (first iteration)
    //0 unsuccessful (small step size)
    //1 "sufficient decrease" when far away from solution (theta_trial>theta_min)
    //2 close to solution but switching condition does not hold, so trial accepted based on "sufficient decrease"
    //3 close to solution and switching condition is true; trial accepted based on Armijo
    lsStatus=0; lsNum=0;

    bool grad_phi_dx_computed=false, iniStep=true; double grad_phi_dx;

    //this will cache the primal infeasibility norm for (reuse)use in the dual updating
    double infeas_nrm_trial=-1.;

    //
    //this is the linesearch loop
    //
    while(true) {
      nlp->runStats.tmSolverInternal.start(); //---

      // check the step against the minimum step size, but accept small
      // fractionToTheBdry since these may occur for tight bounds at the first iteration(s)
      if(!iniStep && _alpha_primal<1e-16) {
	nlp->log->write("Panic: minimum step size reached. The problem may be infeasible or the "
			"gradient inaccurate. Will exit here.",hovError);
	solver_status_ = Steplength_Too_Small;
	break;
      }
      iniStep=false;
      bret = it_trial->takeStep_primals(*it_curr, *dir, _alpha_primal, _alpha_dual); assert(bret);
      num_adjusted_bounds = it_trial->adjust_small_slacks(*it_curr, _mu);
      nlp->runStats.tmSolverInternal.stop(); //---

      //evaluate the problem at the trial iterate (functions only)
      if(!this->evalNlp_funcOnly(*it_trial, _f_nlp_trial, *_c_trial, *_d_trial)) {
	solver_status_ = Error_In_User_Function;
	return Error_In_User_Function;
      }

      logbar->updateWithNlpInfo_trial_funcOnly(*it_trial, _f_nlp_trial, *_c_trial, *_d_trial);

      nlp->runStats.tmSolverInternal.start(); //---
      //compute infeasibility theta at trial point.
      infeas_nrm_trial = theta_trial = resid->compute_nlp_norms(*it_trial, *_c_trial, *_d_trial);

      lsNum++;

      nlp->log->printf(hovLinesearch, "  trial point %d: alphaPrimal=%14.8e barier:(%22.16e)>%15.9e theta:(%22.16e)>%22.16e\n",
		       lsNum, _alpha_primal, logbar->f_logbar, logbar->f_logbar_trial, theta, theta_trial);

      //let's do the cheap, "sufficient progress" test first, before more involved/expensive tests.
      // This simple test is good enough when iterate is far away from solution
      if(theta>=theta_min) {
	//check the filter and the sufficient decrease condition (18)
	if(!filter.contains(theta_trial,logbar->f_logbar_trial)) {
	  if(theta_trial<=(1-gamma_theta)*theta || logbar->f_logbar_trial<=logbar->f_logbar - gamma_phi*theta) {
	    //trial good to go
	    nlp->log->printf(hovLinesearchVerb,
			     "Linesearch: accepting based on suff. decrease (far from solution)\n");
	    lsStatus=1;
	    break;
	  } else {
	    //there is no sufficient progress
	    _alpha_primal *= 0.5;
	    continue;
	  }
	} else {
	  //it is in the filter
	  _alpha_primal *= 0.5;
	  continue;
	}
	nlp->log->write("Warning (close to panic): I got to a point where I wasn't supposed to be. (1)", hovWarning);
      } else {
	// if(theta<theta_min,  then check the switching condition and, if true, rely on Armijo rule.
	// first compute grad_phi^T d_x if it hasn't already been computed
	if(!grad_phi_dx_computed) {
	  nlp->runStats.tmSolverInternal.stop(); //---
	  grad_phi_dx = logbar->directionalDerivative(*dir);
	  grad_phi_dx_computed=true;
	  nlp->runStats.tmSolverInternal.start(); //---
	}
	nlp->log->printf(hovLinesearch, "Linesearch: grad_phi_dx = %22.15e\n", grad_phi_dx);
	//nlp->log->printf(hovSummary, "Linesearch: grad_phi_dx = %22.15e      %22.15e >   %22.15e  \n", grad_phi_dx, _alpha_primal*pow(-grad_phi_dx,s_phi), delta*pow(theta,s_theta));
	//nlp->log->printf(hovSummary, "Linesearch: s_phi=%22.15e;   s_theta=%22.15e; theta=%22.15e; delta=%22.15e \n", s_phi, s_theta, theta, delta);
	//this is the actual switching condition
	if(grad_phi_dx<0 && _alpha_primal*pow(-grad_phi_dx,s_phi)>delta*pow(theta,s_theta)) {

	  if(logbar->f_logbar_trial <= logbar->f_logbar + eta_phi*_alpha_primal*grad_phi_dx) {
	    lsStatus=3;
	    nlp->log->printf(hovLinesearchVerb,
			     "Linesearch: accepting based on Armijo (switch cond also passed)\n");
	    break; //iterate good to go since it satisfies Armijo
	  } else {  //Armijo is not satisfied
	    _alpha_primal *= 0.5; //reduce step and try again
	    continue;
	  }
	} else {//switching condition does not hold

	  //ok to go with  "sufficient progress" condition even when close to solution, provided the switching condition is not satisfied
	  //check the filter and the sufficient decrease condition (18)
	  if(!filter.contains(theta_trial,logbar->f_logbar_trial)) {
	    if(theta_trial<=(1-gamma_theta)*theta
	       || logbar->f_logbar_trial<=logbar->f_logbar - gamma_phi*theta) {

	      //trial good to go
	      nlp->log->printf(hovLinesearchVerb,
			       "Linesearch: accepting based on suff. decrease (switch cond also passed)\n");
	      lsStatus=2;
	      break;
	    } else {
	      //there is no sufficient progress
	      _alpha_primal *= 0.5;
	      continue;
	    }
	  } else {
	    //it is in the filter
	    _alpha_primal *= 0.5;
	    continue;
	  }
	} // end of else: switching condition does not hold

	nlp->log->write("Warning (close to panic): I got to a point where I wasn't supposed to be. (2)", hovWarning);

      } //end of else: theta_trial<theta_min
    } //end of while for the linesearch loop
    nlp->runStats.tmSolverInternal.stop();

    if(num_adjusted_bounds > 0) {
      nlp->log->printf(hovWarning, "%d slacks are too small. Adjust corresponding variable bounds!\n", num_adjusted_bounds);
    }

    //post line-search stuff
    //filter is augmented whenever the switching condition or Armijo rule do not hold for the trial point that was just accepted
    if(lsStatus==1) {
      //need to check switching cond and Armijo to decide if filter is augmented
      if(!grad_phi_dx_computed) {
	grad_phi_dx = logbar->directionalDerivative(*dir);
	grad_phi_dx_computed=true;
      }

      //this is the actual switching condition
      if(grad_phi_dx<0 && _alpha_primal*pow(-grad_phi_dx,s_phi)>delta*pow(theta,s_theta)) {
	//check armijo
	if(logbar->f_logbar_trial <= logbar->f_logbar + eta_phi*_alpha_primal*grad_phi_dx) {
	  //filter does not change
	} else {
	  //Armijo does not hold
	  filter.add(theta_trial, logbar->f_logbar_trial);
	}
      } else { //switching condition does not hold
	filter.add(theta_trial, logbar->f_logbar_trial);
      }

    } else if(lsStatus==2) {
      //switching condition does not hold for the trial
      filter.add(theta_trial, logbar->f_logbar_trial);
    } else if(lsStatus==3) {
      //Armijo (and switching condition) hold, nothing to do.
    } else if(lsStatus==0) {
      //small step; take the update; if the update doesn't pass the convergence test, the optimiz. loop will exit.
    } else
      assert(false && "unrecognized value for lsStatus");

    nlp->log->printf(hovScalars, "Iter[%d] -> accepted step primal=[%17.11e] dual=[%17.11e]\n", iter_num, _alpha_primal, _alpha_dual);
    iter_num++; nlp->runStats.nIter=iter_num;
    //evaluate derivatives at the trial (and to be accepted) trial point
    if(!this->evalNlp_derivOnly(*it_trial, *_grad_f, *_Jac_c, *_Jac_d, *_Hess_Lagr)){
	solver_status_ = Error_In_User_Function;
	return Error_In_User_Function;
      }

    nlp->runStats.tmSolverInternal.start(); //-----
    //reuse function values
    _f_nlp=_f_nlp_trial; hiopVector* pvec=_c_trial; _c_trial=_c; _c=pvec; pvec=_d_trial; _d_trial=_d; _d=pvec;

    //update and adjust the duals
    //it_trial->takeStep_duals(*it_curr, *dir, _alpha_primal, _alpha_dual); assert(bret);
    //bret = it_trial->adjustDuals_primalLogHessian(_mu,kappa_Sigma); assert(bret);
    assert(infeas_nrm_trial>=0 && "this should not happen");
    bret = dualsUpdate->go(*it_curr, *it_trial,
			   _f_nlp, *_c, *_d, *_grad_f, *_Jac_c, *_Jac_d, *dir,
			   _alpha_primal, _alpha_dual, _mu, kappa_Sigma, infeas_nrm_trial); assert(bret);

    //update current iterate (do a fast swap of the pointers)
    hiopIterate* pit=it_curr; it_curr=it_trial; it_trial=pit;
    nlp->log->printf(hovIteration, "Iter[%d] -> full iterate:", iter_num);
    nlp->log->write("", *it_curr, hovIteration);
    nlp->runStats.tmSolverInternal.stop(); //-----

    //notify logbar about the changes
    logbar->updateWithNlpInfo(*it_curr, _mu, _f_nlp, *_c, *_d, *_grad_f, *_Jac_c, *_Jac_d);
    //update residual
    resid->update(*it_curr,_f_nlp, *_c, *_d,*_grad_f,*_Jac_c,*_Jac_d, *logbar);
    nlp->log->printf(hovIteration, "Iter[%d] full residual:-------------\n", iter_num);
    nlp->log->write("", *resid, hovIteration);
  }

  nlp->runStats.tmOptimizTotal.stop();

  //solver_status_ contains the termination information
  displayTerminationMsg();

  //user callback
  nlp->user_callback_solution(solver_status_,
			      *it_curr->get_x(),
			      *it_curr->get_zl(),
			      *it_curr->get_zu(),
			      *_c,*_d,
			      *it_curr->get_yc(),  *it_curr->get_yd(),
			      _f_nlp);
  delete kkt;

  return solver_status_;
}

void hiopAlgFilterIPMQuasiNewton::outputIteration(int lsStatus, int lsNum)
{
  if(iter_num/10*10==iter_num)
    nlp->log->printf(hovSummary, "iter    objective     inf_pr     inf_du   lg(mu)  alpha_du   alpha_pr linesrch\n");

  if(lsStatus==-1)
    nlp->log->printf(hovSummary, "%4d %14.7e %7.3e  %7.3e %6.2f  %7.3e  %7.3e  -(-)\n",
		     iter_num, _f_nlp, _err_nlp_feas, _err_nlp_optim, log10(_mu), _alpha_dual, _alpha_primal);
  else {
    char stepType[2];
    if(lsStatus==1) strcpy(stepType, "s");
    else if(lsStatus==2) strcpy(stepType, "h");
    else if(lsStatus==3) strcpy(stepType, "f");
    else strcpy(stepType, "?");
    nlp->log->printf(hovSummary, "%4d %14.7e %7.3e  %7.3e %6.2f  %7.3e  %7.3e  %d(%s)\n",
		     iter_num, _f_nlp, _err_nlp_feas, _err_nlp_optim, log10(_mu), _alpha_dual, _alpha_primal, lsNum, stepType);
  }
}


/******************************************************************************************************
 * FULL NEWTON IPM
 *****************************************************************************************************/
hiopAlgFilterIPMNewton::hiopAlgFilterIPMNewton(hiopNlpFormulation* nlp_)
  : hiopAlgFilterIPMBase(nlp_),
    fact_acceptor_{nullptr}
{
}

hiopAlgFilterIPMNewton::~hiopAlgFilterIPMNewton()
{
  if(fact_acceptor_){
    delete fact_acceptor_;
  }
  fact_acceptor_ = nullptr;
}

hiopKKTLinSys* hiopAlgFilterIPMNewton::
decideAndCreateLinearSystem(hiopNlpFormulation* nlp)
{
  //hiopNlpMDS* nlpMDS = NULL;
  hiopNlpMDS* nlpMDS = dynamic_cast<hiopNlpMDS*>(nlp);

  if(NULL == nlpMDS)
  {
    hiopNlpSparse* nlpSp = dynamic_cast<hiopNlpSparse*>(nlp);
    if(NULL == nlpSp)
    {
      // this is dense linear system. This is the default case.
      std::string strKKT = nlp->options->GetString("KKTLinsys");
      if(strKKT == "xdycyd")
        return new hiopKKTLinSysDenseXDYcYd(nlp);
      else //'auto' or 'XYcYd'
        return new hiopKKTLinSysDenseXYcYd(nlp);
    } else {
#ifdef HIOP_SPARSE
      // this is Sparse linear system
      std::string strKKT = nlp->options->GetString("KKTLinsys");
      if(strKKT == "full")
        return new hiopKKTLinSysSparseFull(nlp);
      else if(strKKT == "xdycyd")
        return new hiopKKTLinSysCompressedSparseXDYcYd(nlp);
      else //'auto' or 'XYcYd'
        return new hiopKKTLinSysCompressedSparseXYcYd(nlp);
#endif
    }
  } else {
    return new hiopKKTLinSysCompressedMDSXYcYd(nlp);
  }
  assert(false && 
	 "Could not match linear algebra to NLP formulation. Likely, HiOp was"
	 "not built with all linear algebra modules/options");

  return NULL;
}

hiopFactAcceptor* hiopAlgFilterIPMNewton::
decideAndCreateFactAcceptor(hiopPDPerturbation* p, hiopNlpFormulation* nlp)
{
  std::string strKKT = nlp->options->GetString("fact_acceptor");
  if(strKKT == "inertia_correction")
  {
    return new hiopFactAcceptorIC(p,nlp->m_eq()+nlp->m_ineq());
  }else{
    assert(false &&
    "Inertia-free approach hasn't been implemented yet.");
    return new hiopFactAcceptorIC(p,nlp->m_eq()+nlp->m_ineq());
  } 
}

hiopSolveStatus hiopAlgFilterIPMNewton::run()
{
  //hiopNlpFormulation nlp may need an update since user may have changed options and
  //reruning with the same hiopAlgFilterIPMNewton instance
  nlp->finalizeInitialization();

  //also reload options
  reloadOptions();

  //if nlp changed internally, we need to reinitialize 'this'
  if(it_curr->get_x()->get_size()!=nlp->n() ||
     //Jac_c->get_local_size_n()!=nlpdc->n_local()) { <- this is prone to racing conditions
     _Jac_c->n()!=nlp->n()) {
    //size of the nlp changed internally ->  reInitializeNlpObjects();
    reInitializeNlpObjects();
  }
  resetSolverStatus();

  nlp->runStats.initialize();
  nlp->runStats.kkt.initialize();

  if(!pd_perturb_.initialize(nlp)) {
    return SolveInitializationError;
  }

  ////////////////////////////////////////////////////////////////////////////////////
  // run baby run
  ////////////////////////////////////////////////////////////////////////////////////

  nlp->log->printf(hovSummary, "===============\nHiop SOLVER\n===============\n");
  nlp->log->write(NULL, *nlp->options, hovSummary);

#ifdef HIOP_USE_MPI
  nlp->log->printf(hovSummary, "Using %d MPI ranks.\n", nlp->get_num_ranks());
#endif
  nlp->log->write("---------------\nProblem Summary\n---------------", *nlp, hovSummary);

  nlp->runStats.tmOptimizTotal.start();

  startingProcedure(*it_curr, _f_nlp, *_c, *_d, *_grad_f, *_Jac_c, *_Jac_d); //this also evaluates the nlp
  _mu=mu0;

  //update log bar
  logbar->updateWithNlpInfo(*it_curr, _mu, _f_nlp, *_c, *_d, *_grad_f, *_Jac_c, *_Jac_d);
  nlp->log->printf(hovScalars, "log bar obj: %g\n", logbar->f_logbar);
  //recompute the residuals
  resid->update(*it_curr,_f_nlp, *_c, *_d,*_grad_f,*_Jac_c,*_Jac_d, *logbar);

  nlp->log->write("First residual-------------", *resid, hovIteration);
  //nlp->log->printf(hovSummary, "Iter[%d] -> full iterate -------------", iter_num);
  //nlp->log->write("", *it_curr, hovSummary);

  iter_num=0; nlp->runStats.nIter=iter_num;
  bool disableLS = nlp->options->GetString("accept_every_trial_step")=="yes";

  theta_max=1e+4*fmax(1.0,resid->get_theta());
  theta_min=1e-4*fmax(1.0,resid->get_theta());

  hiopKKTLinSys* kkt = decideAndCreateLinearSystem(nlp);
  assert(kkt != NULL);
  kkt->set_PD_perturb_calc(&pd_perturb_);

  if(fact_acceptor_)
  {
    delete fact_acceptor_;
    fact_acceptor_ = nullptr;
  }
  fact_acceptor_ = decideAndCreateFactAcceptor(&pd_perturb_,nlp);
  kkt->set_fact_acceptor(fact_acceptor_);
  
  _alpha_primal = _alpha_dual = 0;

  _err_nlp_optim0=-1.; _err_nlp_feas0=-1.; _err_nlp_complem0=-1;

  // --- Algorithm status 'algStatus ----
  //-1 couldn't solve the problem (most likely because small search step. Restauration phase likely needed)
  // 0 stopped due to tolerances, including acceptable tolerance, or relative tolerance
  // 1 max iter reached
  // 2 user stop via the iteration callback

  bool bret=true;
  int lsStatus=-1, lsNum=0;
  int num_adjusted_bounds = 0;

  int linsol_safe_mode_lastiter = -1;
  bool linsol_safe_mode_on = "stable"==hiop::tolower(nlp->options->GetString("linsol_mode"));
  bool linsol_forcequick = "forcequick"==hiop::tolower(nlp->options->GetString("linsol_mode"));

  solver_status_ = NlpSolve_Pending;
  while(true) {

    bret = evalNlpAndLogErrors(*it_curr, *resid, _mu,
			       _err_nlp_optim, _err_nlp_feas, _err_nlp_complem, _err_nlp,
			       _err_log_optim, _err_log_feas, _err_log_complem, _err_log);
    if(!bret) {
      solver_status_ = Error_In_User_Function;
      return Error_In_User_Function;
    }

    nlp->log->
      printf(hovScalars,
	     "  Nlp    errs: pr-infeas:%23.17e   dual-infeas:%23.17e  comp:%23.17e  overall:%23.17e\n",
	     _err_nlp_feas, _err_nlp_optim, _err_nlp_complem, _err_nlp);
    nlp->log->
      printf(hovScalars,
	     "  LogBar errs: pr-infeas:%23.17e   dual-infeas:%23.17e  comp:%23.17e  overall:%23.17e\n",
	     _err_log_feas, _err_log_optim, _err_log_complem, _err_log);
    outputIteration(lsStatus, lsNum);

    if(_err_nlp_optim0<0) { // && _err_nlp_feas0<0 && _err_nlp_complem0<0
      _err_nlp_optim0=_err_nlp_optim; _err_nlp_feas0=_err_nlp_feas; _err_nlp_complem0=_err_nlp_complem;
    }

    //user callback
    if(!nlp->user_callback_iterate(iter_num, _f_nlp,
				   *it_curr->get_x(),
				   *it_curr->get_zl(),
				   *it_curr->get_zu(),
				   *_c,*_d,
				   *it_curr->get_yc(),  *it_curr->get_yd(), //lambda,
				   _err_nlp_feas, _err_nlp_optim,
				   _mu,
				   _alpha_dual, _alpha_primal,  lsNum)) {
      solver_status_ = User_Stopped; break;
    }

    /*************************************************
     * Termination check
     ************************************************/
    if(checkTermination(_err_nlp, iter_num, solver_status_)) {
      break;
    }
    if(NlpSolve_Pending!=solver_status_) break; //failure of the line search or user stopped.

    /************************************************
     * update mu and other parameters
     ************************************************/
    while(_err_log<=kappa_eps * _mu) {
      //update mu and tau (fraction-to-boundary)
      bret = updateLogBarrierParameters(*it_curr, _mu, _tau, _mu, _tau);
      if(!bret) break; //no update is necessary
      nlp->log->printf(hovScalars, "Iter[%d] barrier params reduced: mu=%g tau=%g\n", iter_num, _mu, _tau);

      //update only logbar problem  and residual (the NLP didn't change)
      logbar->updateWithNlpInfo(*it_curr, _mu, _f_nlp, *_c, *_d, *_grad_f, *_Jac_c, *_Jac_d);

      //! should perform only a partial update since NLP didn't change
      resid->update(*it_curr,_f_nlp, *_c, *_d,*_grad_f,*_Jac_c,*_Jac_d, *logbar);

      bret = evalNlpAndLogErrors(*it_curr, *resid, _mu,
				 _err_nlp_optim, _err_nlp_feas, _err_nlp_complem, _err_nlp,
				 _err_log_optim, _err_log_feas, _err_log_complem, _err_log);
      if(!bret) {
        solver_status_ = Error_In_User_Function;
        return Error_In_User_Function;
      }
      nlp->log->
        printf(hovScalars,
               "  Nlp    errs: pr-infeas:%23.17e   dual-infeas:%23.17e  comp:%23.17e  overall:%23.17e\n",
               _err_nlp_feas, _err_nlp_optim, _err_nlp_complem, _err_nlp);
      nlp->log->
        printf(hovScalars,
               "  LogBar errs: pr-infeas:%23.17e   dual-infeas:%23.17e  comp:%23.17e  overall:%23.17e\n",
               _err_log_feas, _err_log_optim, _err_log_complem, _err_log);

      filter.reinitialize(theta_max);
      //recheck residuals at the first iteration in case the starting pt is  very good
      //if(iter_num==0) {
      //	continue;
      //}
    }
    nlp->log->printf(hovScalars, "Iter[%d] logbarObj=%23.17e (mu=%12.5e)\n", iter_num, logbar->f_logbar,_mu);
    /****************************************************
     * Search direction calculation
     ***************************************************/
    pd_perturb_.set_mu(_mu);

    //this will cache the primal infeasibility norm for (re)use in the dual updating
    double infeas_nrm_trial;

    //
    // this is the linear solve (computeDirections) loop that iterates at most two times
    //
    //  - two times when the step is small (search direction is assumed to be invalid, of ascent): first time
    // linear solve with safe mode (=addtl accuracy and stability) off failed; second times with safe mode on
    //  - one time when the linear solve with the safe mode off is successfull (descent search direction)
    //

    {
      if(linsol_forcequick) {
        linsol_safe_mode_on = false;
      } else {
        //safe mode on for the first three iterations
        if(iter_num<=2) {
          linsol_safe_mode_on=true;
        } else {
          // to do	  if(linsol_safe_mode_on || noinertiacorr)
          if("speculative"==hiop::tolower(nlp->options->GetString("linsol_mode"))) {
            linsol_safe_mode_on=false;
          }
        }
      }
    }

    for(int linsolve=1; linsolve<=2; ++linsolve) {

      nlp->runStats.kkt.start_optimiz_iteration();

      kkt->set_safe_mode(linsol_safe_mode_on);
      //
      //update the Hessian and kkt system; usually a matrix factorization occurs
      //
      if(!kkt->update(it_curr, _grad_f, _Jac_c, _Jac_d, _Hess_Lagr)) {

        nlp->runStats.kkt.end_optimiz_iteration();

        if(linsol_safe_mode_on) {
          nlp->log->write("Unrecoverable error in step computation (factorization) [1]. Will exit here.",
                          hovError);
          return solver_status_ = Err_Step_Computation;
        } else {

          //failed with 'linsol_mode'='forcequick' means unrecoverable
          if(linsol_forcequick) {

            nlp->log->write("Unrecoverable error in step computation (factorization) [2]. Will exit here.",
                            hovError);
            return solver_status_ = Err_Step_Computation;
          }

          linsol_safe_mode_on = true;
          linsol_safe_mode_lastiter = iter_num;

          nlp->log->printf(hovWarning,
                           "Requesting additional accuracy and stability from the KKT linear system "
                           "at iteration %d (safe mode ON)\n", iter_num);

          // repeat linear solve (computeDirections) in safe mode (meaning additional accuracy
          // and stability is requested)
          continue;
        }
      } // end of if(!kkt->update(it_curr, _grad_f, _Jac_c, _Jac_d, _Hess_Lagr))

      //
      // solve for search directions
      //
      if(!kkt->computeDirections(resid, dir)) {

        nlp->runStats.kkt.start_optimiz_iteration();

        if(linsol_safe_mode_on) {
          nlp->log->write("Unrecoverable error in step computation (solve)[1]. Will exit here.", hovError);
          return solver_status_ = Err_Step_Computation;
        } else {
          if(linsol_forcequick) {
            nlp->log->write("Unrecoverable error in step computation (solve)[2]. Will exit here.", hovError);
            return solver_status_ = Err_Step_Computation;
          }
          linsol_safe_mode_on = true;
          linsol_safe_mode_lastiter = iter_num;

          nlp->log->printf(hovWarning,
                           "Requesting additional accuracy and stability from the KKT linear system "
                           "at iteration %d (safe mode ON)\n", iter_num);

          // repeat linear solve (computeDirections) in safe mode (meaning additional accuracy
          // and stability is requested)
          continue;

        }
      } // end of if(!kkt->computeDirections(resid, dir))

      //at this point all is good in terms of searchDirections computations as far as the linear solve
      //is concerned; the search direction can be of ascent because some fast factorizations do not
      //support inertia calculation; this case will be handled later on in this loop
      //( //! todo nopiv inertia calculation ))
      nlp->runStats.kkt.end_optimiz_iteration();

      if(perf_report_kkt_) {
        nlp->log->printf(hovSummary, "%s", nlp->runStats.kkt.get_summary_last_iter().c_str());
      }

      nlp->log->printf(hovIteration, "Iter[%d] full search direction -------------\n", iter_num);
      nlp->log->write("", *dir, hovIteration);
      /***************************************************************
       * backtracking line search
       ****************************************************************/
      nlp->runStats.tmSolverInternal.start();

      //maximum  step
      bret = it_curr->fractionToTheBdry(*dir, _tau, _alpha_primal, _alpha_dual); assert(bret);
      double theta = resid->get_theta(); //at it_curr
      double theta_trial;
      nlp->runStats.tmSolverInternal.stop();

      //lsStatus: line search status for the accepted trial point. Needed to update the filter
      //-1 uninitialized (first iteration)
      //0 unsuccessful (small step size)
      //1 "sufficient decrease" when far away from solution (theta_trial>theta_min)
      //2 close to solution but switching condition does not hold; trial accepted based on "sufficient decrease"
      //3 close to solution and switching condition is true; trial accepted based on Armijo
      lsStatus=0; lsNum=0;

      bool grad_phi_dx_computed=false, iniStep=true; double grad_phi_dx;

      //this will cache the primal infeasibility norm for (re)use in the dual updating
      infeas_nrm_trial=-1.;
      //
      // linesearch loop
      //
      while(true) {
        nlp->runStats.tmSolverInternal.start(); //---

        // check the step against the minimum step size, but accept small
        // fractionToTheBdry since these may occur for tight bounds at the first iteration(s)
        if(!iniStep && _alpha_primal<1e-16) {

          if(linsol_safe_mode_on) {
            nlp->log->write("Panic: minimum step size reached. The problem may be infeasible or the "
                            "gradient inaccurate. Will exit here.", hovError);
            solver_status_ = Steplength_Too_Small;
            break;
          } else {
            lsStatus = 0;
            break;
          }
        }
        iniStep=false;
        bret = it_trial->takeStep_primals(*it_curr, *dir, _alpha_primal, _alpha_dual); assert(bret);
        num_adjusted_bounds = it_trial->adjust_small_slacks(*it_curr, _mu);
        nlp->runStats.tmSolverInternal.stop(); //---

        //evaluate the problem at the trial iterate (functions only)
        if(!this->evalNlp_funcOnly(*it_trial, _f_nlp_trial, *_c_trial, *_d_trial)) {
          solver_status_ = Error_In_User_Function;
          return Error_In_User_Function;
        }

        logbar->updateWithNlpInfo_trial_funcOnly(*it_trial, _f_nlp_trial, *_c_trial, *_d_trial);

        nlp->runStats.tmSolverInternal.start(); //---
        //compute infeasibility theta at trial point.
        infeas_nrm_trial = theta_trial = resid->compute_nlp_norms(*it_trial, *_c_trial, *_d_trial);

        lsNum++;

        nlp->log->printf(hovLinesearch, "  trial point %d: alphaPrimal=%14.8e barier:(%22.16e)>%15.9e "
                         "theta:(%22.16e)>%22.16e\n",
                         lsNum, _alpha_primal, logbar->f_logbar, logbar->f_logbar_trial, theta, theta_trial);

        if(disableLS) break;

        nlp->log->write("Filter IPM: ", filter, hovLinesearch);

        // Do the cheap, "sufficient progress" test first, before more involved/expensive tests.
        // This simple test is good enough when iterate is far away from solution
        if(theta>=theta_min) {
          //check the filter and the sufficient decrease condition (18)
          if(!filter.contains(theta_trial,logbar->f_logbar_trial)) {
            if(theta_trial<=(1-gamma_theta)*theta ||
               logbar->f_logbar_trial<=logbar->f_logbar - gamma_phi*theta) {
              //trial good to go
              nlp->log->printf(hovLinesearchVerb, "Linesearch: accepting based on suff. decrease "
                               "(far from solution)\n");
              lsStatus=1;
              break;
            } else {
              //there is no sufficient progress
              _alpha_primal *= 0.5;
              continue;
            }
          } else {
            //it is in the filter
            _alpha_primal *= 0.5;
            continue;
          }
          nlp->log->write("Warning (close to panic): got to a point I wasn't supposed reach. (1)",
                          hovWarning);
        } else {
          // if(theta<theta_min,  then check the switching condition and, if true, rely on Armijo rule.
          // first compute grad_phi^T d_x if it hasn't already been computed
          if(!grad_phi_dx_computed) {
            nlp->runStats.tmSolverInternal.stop(); //---
            grad_phi_dx = logbar->directionalDerivative(*dir);
            grad_phi_dx_computed=true;
            nlp->runStats.tmSolverInternal.start(); //---
          }
          nlp->log->printf(hovLinesearch, "Linesearch: grad_phi_dx = %22.15e\n", grad_phi_dx);

          // nlp->log->printf(hovSummary,
          // 		 "Linesearch: grad_phi_dx = %22.15e      %22.15e >   %22.15e  \n",
          // 		 grad_phi_dx, _alpha_primal*pow(-grad_phi_dx,s_phi), delta*pow(theta,s_theta));
          // nlp->log->printf(hovSummary,
          // 		 "Linesearch: s_phi=%22.15e;   s_theta=%22.15e; theta=%22.15e; delta=%22.15e\n",
          // 		 s_phi, s_theta, theta, delta);

          // this is the actual switching condition
          if(grad_phi_dx<0 && _alpha_primal*pow(-grad_phi_dx,s_phi)>delta*pow(theta,s_theta)) {

            if(logbar->f_logbar_trial <= logbar->f_logbar + eta_phi*_alpha_primal*grad_phi_dx) {
              lsStatus=3;
              nlp->log->printf(hovLinesearchVerb,
                               "Linesearch: accepting based on Armijo (switch cond also passed)\n");

            //iterate good to go since it satisfies Armijo
              break;
            } else {
              //Armijo is not satisfied
              _alpha_primal *= 0.5; //reduce step and try again
              continue;
            }
          } else {//switching condition does not hold

            //ok to go with  "sufficient progress" condition even when close to solution, provided the
            //switching condition is not satisfied

            //check the filter and the sufficient decrease condition (18)
            if(!filter.contains(theta_trial,logbar->f_logbar_trial)) {
              if(theta_trial<=(1-gamma_theta)*theta ||
                 logbar->f_logbar_trial <= logbar->f_logbar - gamma_phi*theta) {

                //trial good to go
                nlp->log->printf(hovLinesearchVerb,
                                 "Linesearch: accepting based on suff. decrease (switch cond also passed)\n");
                lsStatus=2;
                break;
              } else {
                //there is no sufficient progress
                _alpha_primal *= 0.5;
                continue;
              }
            } else {
              //it is in the filter
              _alpha_primal *= 0.5;
              continue;
            }
          } // end of else: switching condition does not hold

          nlp->log->write("Warning (close to panic): got to a point I wasn't supposed to reach. (2)",
                          hovWarning);

        } //end of else: theta_trial<theta_min
      } //end of while for the linesearch loop
      nlp->runStats.tmSolverInternal.stop();

      if(num_adjusted_bounds > 0) {
        nlp->log->printf(hovWarning, "%d slacks are too small. Adjust corresponding variable bounds!\n", num_adjusted_bounds);
      }
      // post line-search: filter is augmented whenever the switching condition or Armijo rule do not
      // hold for the trial point that was just accepted
      if(lsStatus==1) {

        //need to check switching cond and Armijo to decide if filter is augmented
        if(!grad_phi_dx_computed) {
          grad_phi_dx = logbar->directionalDerivative(*dir);
          grad_phi_dx_computed=true;
        }

        //this is the actual switching condition
        if(grad_phi_dx<0 && (_alpha_primal*pow(-grad_phi_dx,s_phi) > delta*pow(theta,s_theta))) {
          //check armijo
          if(logbar->f_logbar_trial <= logbar->f_logbar + eta_phi*_alpha_primal*grad_phi_dx) {
            //filter does not change
          } else {
            //Armijo does not hold
            filter.add(theta_trial, logbar->f_logbar_trial);
          }
        } else { //switching condition does not hold
          filter.add(theta_trial, logbar->f_logbar_trial);
        }
        break; //from the linear solve (computeDirections) loop

      } else if(lsStatus==2) {
        //switching condition does not hold for the trial
        filter.add(theta_trial, logbar->f_logbar_trial);

        break; //from the linear solve (computeDirections) loop

      } else if(lsStatus==3) {
        //Armijo (and switching condition) hold, nothing to do.

        break; //from the linear solve (computeDirections) loop

      } else if(lsStatus==0) {

        //
        //small step
        //

        if(linsol_safe_mode_on) {

          // this is  likely catastrophic
          // however take the update;
          // if the update doesn't pass the convergence test, the optimiz. loop will exit

          // first exit the linear solve (computeDirections) loop
          break;

        } else {
          //here false == linsol_safe_mode_on
          if(linsol_forcequick) {
            // this is  likely catastrophic as under 'linsol_mode'='forcequick' we deliberately
            //won't switch to safe mode
            //
            // however take the update;
            // if the update doesn't pass the convergence test, the optimiz. loop will exit

            // first exit the linear solve (computeDirections) loop
            break;
          }

          linsol_safe_mode_on = true;
          linsol_safe_mode_lastiter = iter_num;

          nlp->log->printf(hovWarning,
                           "Requesting additional accuracy and stability from the KKT linear system "
                           "at iteration %d (safe mode ON)\n", iter_num);

          // repeat linear solve (computeDirections) in safe mode (meaning additional accuracy
          // and stability is requested)
          continue;

        }
      } else {
        assert(false && "unrecognized value for lsStatus");
      }
    } // end of the linear solve (computeDirections) loop

    nlp->log->printf(hovScalars,
		     "Iter[%d] -> accepted step primal=[%17.11e] dual=[%17.11e]\n",
		     iter_num, _alpha_primal, _alpha_dual);
    iter_num++;
    nlp->runStats.nIter=iter_num;

    // update and adjust the duals
    // this needs to be done before evalNlp_derivOnly so that the user's NLP functions
    // get the updated duals
    assert(infeas_nrm_trial>=0 && "this should not happen");
    bret = dualsUpdate->go(*it_curr, *it_trial,
			   _f_nlp, *_c, *_d, *_grad_f, *_Jac_c, *_Jac_d, *dir,
			   _alpha_primal, _alpha_dual, _mu, kappa_Sigma, infeas_nrm_trial); assert(bret);

    //evaluate derivatives at the trial (and to be accepted) trial point
    if(!this->evalNlp_derivOnly(*it_trial, *_grad_f, *_Jac_c, *_Jac_d, *_Hess_Lagr)) {
      solver_status_ = Error_In_User_Function;
      return Error_In_User_Function;
    }

    nlp->runStats.tmSolverInternal.start(); //-----
    //reuse function values
    _f_nlp=_f_nlp_trial;
    hiopVector* pvec=_c_trial; _c_trial=_c; _c=pvec; pvec=_d_trial; _d_trial=_d; _d=pvec;

    //
    //update current iterate (do a fast swap of the pointers)
    //
    hiopIterate* pit=it_curr; it_curr=it_trial; it_trial=pit;

    nlp->log->printf(hovIteration, "Iter[%d] -> full iterate:", iter_num);
    nlp->log->write("", *it_curr, hovIteration);

    nlp->runStats.tmSolverInternal.stop(); //-----

    //notify logbar about the changes
    _f_log = _f_nlp;

    logbar->updateWithNlpInfo(*it_curr, _mu, _f_log, *_c, *_d, *_grad_f, *_Jac_c, *_Jac_d);
    //update residual
    resid->update(*it_curr,_f_nlp, *_c, *_d,*_grad_f,*_Jac_c,*_Jac_d, *logbar);

    nlp->log->printf(hovIteration, "Iter[%d] full residual:-------------\n", iter_num);
    nlp->log->write("", *resid, hovIteration);
  }

  nlp->runStats.tmOptimizTotal.stop();

  //solver_status_ contains the termination information
  displayTerminationMsg();

  //user callback
  nlp->user_callback_solution(solver_status_,
			      *it_curr->get_x(),
			      *it_curr->get_zl(),
			      *it_curr->get_zu(),
			      *_c,*_d,
			      *it_curr->get_yc(), *it_curr->get_yd(),
			      _f_nlp);
  delete kkt;

  return solver_status_;
}

void hiopAlgFilterIPMNewton::outputIteration(int lsStatus, int lsNum)
{
  if(iter_num/10*10==iter_num)
    nlp->log->printf(hovSummary,
		     "iter    objective     inf_pr     inf_du   lg(mu)  alpha_du   alpha_pr linesrch\n");

  if(lsStatus==-1)
    nlp->log->printf(hovSummary, "%4d %14.7e %7.3e  %7.3e %6.2f  %7.3e  %7.3e  -(-)\n",
		     iter_num, _f_nlp/nlp->get_obj_scale(), _err_nlp_feas, _err_nlp_optim, log10(_mu), _alpha_dual, _alpha_primal);
  else {
    char stepType[2];

    if(lsStatus==1) strcpy(stepType, "s");
    else if(lsStatus==2) strcpy(stepType, "h");
    else if(lsStatus==3) strcpy(stepType, "f");
    else strcpy(stepType, "?");

    nlp->log->printf(hovSummary, "%4d %14.7e %7.3e  %7.3e %6.2f  %7.3e  %7.3e  %d(%s)\n",
		     iter_num, _f_nlp/nlp->get_obj_scale(), _err_nlp_feas,
		     _err_nlp_optim, log10(_mu),
		     _alpha_dual, _alpha_primal,
		     lsNum, stepType);
  }
}
} //end namespace
