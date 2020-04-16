// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory (LLNL).
// Written by Cosmin G. Petra, petra1@llnl.gov.
// LLNL-CODE-742473. All rights reserved.
//
// This file is part of HiOp. For details, see https://github.com/LLNL/hiop. HiOp 
// is released under the BSD 3-clause license (https://opensource.org/licenses/BSD-3-Clause). 
// Please also read “Additional BSD Notice” below.
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

#include "hiopNlpFormulation.hpp"
#include "hiopHessianLowRank.hpp"
#include "hiopLogger.hpp"

#ifdef HIOP_USE_MPI
#include "mpi.h"
#else 
#include <cstddef>
#endif

#include <stdlib.h>     /* exit, EXIT_FAILURE */

#include <cassert>
namespace hiop
{

hiopNlpFormulation::hiopNlpFormulation(hiopInterfaceBase& interface_)
#ifdef HIOP_USE_MPI
  : mpi_init_called(false), interface_base(interface_)
#else 
  : interface_base(interface_)
#endif
{
  strFixedVars = ""; //uninitialized
  dFixedVarsTol=-1.; //uninitialized
  bool bret;
#ifdef HIOP_USE_MPI
  bret = interface_base.get_MPI_comm(comm); assert(bret);

  int nret;
  //MPI may not be initialized: this occurs when a serial driver call HiOp built with MPI support on
  int initialized;
  nret = MPI_Initialized( &initialized );
  if(!initialized) {
    mpi_init_called=true;
    nret = MPI_Init(NULL,NULL);
    assert(MPI_SUCCESS==nret);
  } 
  
  nret=MPI_Comm_rank(comm, &rank); assert(MPI_SUCCESS==nret);
  nret=MPI_Comm_size(comm, &num_ranks); assert(MPI_SUCCESS==nret);
#else
  //fake communicator (defined by hiop)
  MPI_Comm comm = MPI_COMM_SELF;
#endif

  options = new hiopOptions(/*filename=NULL*/);

  hiopOutVerbosity hov = (hiopOutVerbosity) options->GetInteger("verbosity_level");
  log = new hiopLogger(this, stdout);

  options->SetLog(log);
  //log->write(NULL, *options, hovSummary);//! comment this at some point

  runStats = hiopRunStats(comm);

  /* NLP members intialization */
  bret = interface_base.get_prob_sizes(n_vars, n_cons); assert(bret);
  xl=NULL;
  xu=NULL;
  vars_type=NULL;
  ixl=NULL;
  ixu=NULL;
  c_rhs=NULL;
  cons_eq_type=NULL;
  dl=NULL;
  du=NULL;
  cons_ineq_type=NULL;
  cons_eq_mapping=NULL;
  cons_ineq_mapping=NULL;
  idl=NULL;
  idu=NULL;
#ifdef HIOP_USE_MPI
  vec_distrib=NULL;
#endif
  cons_eval_type_ = -1;
  cons_body_ = NULL;
  cons_Jac_ = NULL;
}

hiopNlpFormulation::~hiopNlpFormulation()
{  
  if(xl)   delete xl;
  if(xu)   delete xu;
  if(ixl)  delete ixl;
  if(ixu)  delete ixu;
  if(c_rhs)delete c_rhs;
  if(dl)   delete dl;
  if(du)   delete du;
  if(idl)  delete idl;
  if(idu)  delete idu;

  if(vars_type)      delete[] vars_type;
  if(cons_ineq_type) delete[] cons_ineq_type;
  if(cons_eq_type)   delete[] cons_eq_type;

  if(cons_eq_mapping)   delete[] cons_eq_mapping;
  if(cons_ineq_mapping) delete[] cons_ineq_mapping;
#ifdef HIOP_USE_MPI
  if(vec_distrib) delete[] vec_distrib;
#endif
  delete log;
  delete options;

#ifdef HIOP_USE_MPI
  //some (serial) drivers call (MPI) HiOp repeatedly in an outer loop
  //if we finalize here, subsequent calls to MPI will fail and break this outer loop. So we don't finalize
  //if(mpi_init_called) { 
  //  int nret=MPI_Finalize(); assert(MPI_SUCCESS==nret);
  //}
#endif
  delete[] cons_body_;
  delete cons_Jac_;
}

bool hiopNlpFormulation::finalizeInitialization()
{
  //check if there was a change in the user options that requires reinitialization of 'this'
  bool doinit = false; 
  if(strFixedVars != options->GetString("fixed_var")) {
    doinit=true;
  }
  const double fixedVarTol = options->GetNumeric("fixed_var_tolerance");
  if(dFixedVarsTol != fixedVarTol) {
    doinit=true;
  }
  //more tests here (for example change in the rescaling)
  if(!doinit) {
    return true;
  } else {
    
  }

  bool bret = interface_base.get_prob_sizes(n_vars, n_cons); assert(bret);

  nlp_transformations.clear();
  nlp_transformations.setUserNlpNumVars(n_vars);

  if(xl) delete xl;
  if(xu) delete xu;
  if(vars_type) delete[] vars_type;
#ifdef HIOP_USE_MPI
  if(vec_distrib) delete[] vec_distrib;
  vec_distrib=new long long[num_ranks+1];
  if(true==interface_base.get_vecdistrib_info(n_vars,vec_distrib)) {
    xl = new hiopVectorPar(n_vars, vec_distrib, comm);
  } else {
    xl = new hiopVectorPar(n_vars);   
    delete[] vec_distrib;
    vec_distrib = NULL;
  }
#else
  xl   = new hiopVectorPar(n_vars);
#endif  
  xu = xl->alloc_clone();

  int nlocal=xl->get_local_size();

  nlp_transformations.setUserNlpNumLocalVars(nlocal);

  double  *xl_vec= xl->local_data(),  *xu_vec= xu->local_data();
  vars_type = new hiopInterfaceBase::NonlinearityType[nlocal];

  bret=interface_base.get_vars_info(n_vars,xl_vec,xu_vec,vars_type); assert(bret);

  //allocate and build ixl(ow) and ix(upp) vectors
  if(ixl) delete ixl; if(ixu) delete ixu;
  ixl = xu->alloc_clone(); ixu = xu->alloc_clone();
  n_bnds_low_local = n_bnds_upp_local = 0;
  n_bnds_lu = 0;
  long long nfixed_vars_local=0;
  double *ixl_vec=ixl->local_data(), *ixu_vec=ixu->local_data();
#ifdef HIOP_DEEPCHECKS
  const int maxBndsCloseMsgs=3; int nBndsClose=0;
#endif
  for(int i=0;i<nlocal; i++) {
    if(xl_vec[i]>-1e20) { 
      ixl_vec[i]=1.; n_bnds_low_local++;
      if(xu_vec[i]< 1e20) n_bnds_lu++;
    } else ixl_vec[i]=0.;

    if(xu_vec[i]< 1e20) { 
      ixu_vec[i]=1.; n_bnds_upp_local++;
    } else ixu_vec[i]=0.;

#ifdef HIOP_DEEPCHECKS
    assert(xl_vec[i] <= xu_vec[i] && "please fix the inconsistent bounds, otherwise the problem is infeasible");
#endif

    //if(xl_vec[i]==xu_vec[i]) {
    if(fabs(xl_vec[i]-xu_vec[i])<= fixedVarTol*fmax(1.,fabs(xu_vec[i]))) {
      nfixed_vars_local++;
    } else {
#ifdef HIOP_DEEPCHECKS
      #define min_dist 1e-8
      if(fixedVarTol<min_dist) { 
	if(nBndsClose<maxBndsCloseMsgs) {
	  if(fabs(xl_vec[i]-xu_vec[i]) / std::max(1.,fabs(xu_vec[i]))<min_dist) {
	    log->printf(hovWarning, 
			"Lower (%g) and upper bound (%g) for variable %d are very close. "
			"Consider fixing this variable or increase 'fixed_var_tolerance'.\n",
			i, xl_vec[i], xu_vec[i]);
	    nBndsClose++;
	  }
	} 
	if(nBndsClose==maxBndsCloseMsgs) {
	  log->printf(hovWarning, "[further messages were surpressed]\n");
	  nBndsClose++;
	}
      }
#endif
    }
  }

  dFixedVarsTol = fixedVarTol;
  
  long long nfixed_vars=nfixed_vars_local;
#ifdef HIOP_USE_MPI
  int ierr = MPI_Allreduce(&nfixed_vars_local, &nfixed_vars, 1, MPI_LONG_LONG, MPI_SUM, comm); 
  assert(MPI_SUCCESS==ierr);
#endif
  hiopFixedVarsRemover* fixedVarsRemover = NULL;
  if(nfixed_vars>0) {
    log->printf(hovWarning, "Detected %lld fixed variables out of a total of %lld.\n", nfixed_vars, n_vars);

    if(options->GetString("fixed_var")=="remove") {
      log->printf(hovWarning, "Fixed variables will be removed internally.\n");

      fixedVarsRemover = new hiopFixedVarsRemover(*xl, *xu, fixedVarTol, nfixed_vars, nfixed_vars_local);
      

#ifdef HIOP_USE_MPI
      fixedVarsRemover->setFSVectorDistrib(vec_distrib,num_ranks);
      fixedVarsRemover->setMPIComm(comm);
#endif
      bret = fixedVarsRemover->setupDecisionVectorPart(); 
      assert(bret && "error while removing fixed variables");
    
      n_vars = fixedVarsRemover->rs_n();
#ifdef HIOP_USE_MPI
      long long* vec_distrib_rs = fixedVarsRemover->allocRSVectorDistrib();
      if(vec_distrib) delete[] vec_distrib;
      vec_distrib = vec_distrib_rs;
#endif
    
      hiopVectorPar* xl_rs;
#ifdef HIOP_USE_MPI
      if(vec_distrib!=NULL) {
	xl_rs = new hiopVectorPar(n_vars, vec_distrib, comm);
      } else {
	xl_rs = new hiopVectorPar(n_vars);   
      }
#else
      xl_rs = new hiopVectorPar(n_vars); 
#endif
      
      hiopVectorPar* xu_rs  = xl_rs->alloc_clone();
      hiopVectorPar* ixl_rs = xl_rs->alloc_clone(); 
      hiopVectorPar* ixu_rs = xu_rs->alloc_clone();
      
      fixedVarsRemover->copyFsToRs( *xl, *xl_rs);
      fixedVarsRemover->copyFsToRs( *xu, *xu_rs);
      fixedVarsRemover->copyFsToRs(*ixl, *ixl_rs);
      fixedVarsRemover->copyFsToRs(*ixu, *ixu_rs);
      
      nlocal=xl_rs->get_local_size();
      hiopInterfaceBase::NonlinearityType* vars_type_rs = new hiopInterfaceBase::NonlinearityType[nlocal];
      fixedVarsRemover->copyFsToRs(vars_type, vars_type_rs);
      
      delete xl; delete xu; delete ixl; delete ixu; delete[] vars_type;
      xl = xl_rs; xu = xu_rs; ixl = ixl_rs; ixu = ixu_rs; vars_type = vars_type_rs;
      
      n_bnds_low_local -= nfixed_vars_local;
      n_bnds_upp_local -= nfixed_vars_local;
      n_bnds_lu        -= nfixed_vars_local;
      
      nlp_transformations.append(fixedVarsRemover);
    } else {
      if(options->GetString("fixed_var")=="relax") {
	log->printf(hovWarning, "Fixed variables will be relaxed internally.\n");
	hiopFixedVarsRelaxer* fixedVarsRelaxer = new hiopFixedVarsRelaxer(*xl, *xu, nfixed_vars, nfixed_vars_local);
	fixedVarsRelaxer->setup();
	fixedVarsRelaxer->relax(options->GetNumeric("fixed_var_tolerance"), options->GetNumeric("fixed_var_perturb"), *xl, *xu);

	nlp_transformations.append(fixedVarsRelaxer);

      } else {
	log->printf(hovError,  
		    "detected fixed variables but was not instructed "
		    "how to deal with them (option 'fixed_var' is 'none').\n");
	exit(EXIT_FAILURE);
      }
    }
  }
  /* split the constraints */
  hiopVectorPar* gl = new hiopVectorPar(n_cons); 
  hiopVectorPar* gu = new hiopVectorPar(n_cons);
  double *gl_vec=gl->local_data(), *gu_vec=gu->local_data();
  hiopInterfaceBase::NonlinearityType* cons_type = new hiopInterfaceBase::NonlinearityType[n_cons];
  bret = interface_base.get_cons_info(n_cons, gl_vec, gu_vec, cons_type); assert(bret);

  assert(gl->get_local_size()==n_cons);
  assert(gl->get_local_size()==n_cons);
  n_cons_eq=n_cons_ineq=0; 
  for(int i=0;i<n_cons; i++) {
    if(gl_vec[i]==gu_vec[i]) n_cons_eq++;
    else                     n_cons_ineq++;
  }

  if(c_rhs) delete c_rhs; 
  if(cons_eq_type) delete[] cons_eq_type;
  if(dl) delete dl; if(du) delete du;
  if(cons_ineq_type) delete[] cons_ineq_type;
  if(cons_eq_mapping) delete[] cons_eq_mapping;
  if(cons_ineq_mapping) delete[] cons_ineq_mapping;
  /* allocate c_rhs, dl, and du (all serial in this formulation) */
  c_rhs = new hiopVectorPar(n_cons_eq);
  cons_eq_type = new  hiopInterfaceBase::NonlinearityType[n_cons_eq];
  dl    = new hiopVectorPar(n_cons_ineq);
  du    = new hiopVectorPar(n_cons_ineq);
  cons_ineq_type = new  hiopInterfaceBase::NonlinearityType[n_cons_ineq];
  cons_eq_mapping   = new long long[n_cons_eq];
  cons_ineq_mapping = new long long[n_cons_ineq];

  /* copy lower and upper bounds - constraints */
  double *dlvec=dl->local_data(), *duvec=du->local_data(), *c_rhsvec=c_rhs->local_data();
  int it_eq=0, it_ineq=0;
  for(int i=0;i<n_cons; i++) {
    if(gl_vec[i]==gu_vec[i]) {
      cons_eq_type[it_eq]=cons_type[i]; 
      c_rhsvec[it_eq] = gl_vec[i]; 
      cons_eq_mapping[it_eq]=i;
      it_eq++;
    } else {
#ifdef HIOP_DEEPCHECKS
    assert(gl_vec[i] <= gu_vec[i] && "please fix the inconsistent inequality constraints, otherwise the problem is infeasible");
#endif
      cons_ineq_type[it_ineq]=cons_type[i];
      dlvec[it_ineq]=gl_vec[i]; duvec[it_ineq]=gu_vec[i]; 
      cons_ineq_mapping[it_ineq]=i;
      it_ineq++;
    }
  }
  assert(it_eq==n_cons_eq); assert(it_ineq==n_cons_ineq);
  /* delete the temporary buffers */
  delete gl; delete gu; delete[] cons_type;

  if(idl) delete idl; if(idu) delete idu;
  /* iterate over the inequalities and build the idl(ow) and idu(pp) vectors */
  idl = dl->alloc_clone(); idu=du->alloc_clone();
  n_ineq_low=n_ineq_upp=0; n_ineq_lu=0;
  double* idl_vec=idl->local_data(); double* idu_vec=idu->local_data();
  double* dl_vec = dl->local_data(); double* du_vec = du->local_data();
  for(int i=0; i<n_cons_ineq; i++) {
    if(dl_vec[i]>-1e20) { 
      idl_vec[i]=1.; n_ineq_low++; 
      if(du_vec[i]< 1e20) n_ineq_lu++;
    }
    else idl_vec[i]=0.;

    if(du_vec[i]< 1e20) { 
      idu_vec[i]=1.; n_ineq_upp++; 
    } else idu_vec[i]=0.;
    //idl_vec[i] = dl_vec[i]<=-1e20?0.:1.;
    //idu_vec[i] = du_vec[i]>= 1e20?0.:1.;
  }

  if(fixedVarsRemover) {
    fixedVarsRemover->setupConstraintsPart(n_cons_eq, n_cons_ineq);
  }
  strFixedVars = options->GetString("fixed_var");

  //compute the overall n_low and n_upp
#ifdef HIOP_USE_MPI
  long long aux[3]={n_bnds_low_local, n_bnds_upp_local, n_bnds_lu}, aux_g[3];
  ierr=MPI_Allreduce(aux, aux_g, 3, MPI_LONG_LONG, MPI_SUM, comm); assert(MPI_SUCCESS==ierr);
  n_bnds_low=aux_g[0]; n_bnds_upp=aux_g[1]; n_bnds_lu=aux_g[2];
#else
  n_bnds_low=n_bnds_low_local; n_bnds_upp=n_bnds_upp_local; //n_bnds_lu is ok
#endif

  //reset/release info and data related to one-call constraints evaluation
  cons_eval_type_ = -1;
  delete[] cons_body_;
  cons_body_ = NULL;
  delete cons_Jac_;
  cons_Jac_ = NULL;
  return bret;
}


hiopVector* hiopNlpFormulation::alloc_primal_vec() const
{
  return xl->alloc_clone();
}

hiopVector* hiopNlpFormulation::alloc_dual_eq_vec() const
{
  return c_rhs->alloc_clone();
}
hiopVector* hiopNlpFormulation::alloc_dual_ineq_vec() const
{
  return dl->alloc_clone();
}
hiopVector* hiopNlpFormulation::alloc_dual_vec() const
{
  hiopVectorPar* ret=new hiopVectorPar(n_cons);
#ifdef HIOP_DEEPCHECKS
  assert(ret!=NULL);
#endif
  return ret;
}

bool hiopNlpFormulation::eval_f(double* x, bool new_x, double& f)
{
  double* xx = nlp_transformations.applyTox(x, new_x);

  runStats.tmEvalObj.start();
  bool bret = interface_base.eval_f(nlp_transformations.n_post(),xx,new_x,f);
  runStats.tmEvalObj.stop(); runStats.nEvalObj++;

  f = nlp_transformations.applyToObj(f);
  return bret;
}
bool hiopNlpFormulation::eval_grad_f(double* x, bool new_x, double* gradf)
{
  double* xx     = nlp_transformations.applyTox(x, new_x);
  double* gradff = nlp_transformations.applyToGradObj(gradf);
  bool bret; 
  runStats.tmEvalGrad_f.start();
  bret = interface_base.eval_grad_f(nlp_transformations.n_post(),xx,new_x,gradff);
  runStats.tmEvalGrad_f.stop(); runStats.nEvalGrad_f++;

  gradf = nlp_transformations.applyInvToGradObj(gradff);
  return bret;
}

bool hiopNlpFormulation::get_starting_point(hiopVector& x0_)
{
  hiopVectorPar &x0_for_hiop = dynamic_cast<hiopVectorPar&>(x0_);
  bool bret; 

  double* x0_for_user = nlp_transformations.applyTox(x0_for_hiop.local_data(),true);

  bret = interface_base.get_starting_point(nlp_transformations.n_post(), x0_for_user);

  nlp_transformations.applyInvTox(x0_for_user, x0_for_hiop);
  return bret;
}

bool hiopNlpFormulation::eval_c(double*x, bool new_x, double* c)
{
  double* xx = nlp_transformations.applyTox(x, new_x);
  double* cc = c;//nlp_transformations.applyToCons(c, n_cons_eq); //not needed for now

  runStats.tmEvalCons.start();
  bool bret = interface_base.eval_cons(nlp_transformations.n_post(),n_cons,n_cons_eq,cons_eq_mapping,xx,new_x,cc);
  runStats.tmEvalCons.stop(); runStats.nEvalCons_eq++;

  //c = nlp_transformations.applyInvToCons(c, n_cons_eq); //not needed for now
  return bret;
}
bool hiopNlpFormulation::eval_d(double*x, bool new_x, double* d)
{
  double* xx = nlp_transformations.applyTox(x, new_x);
  double* dd = d;//nlp_transformations.applyToCons(d, n_cons_ineq); //not needed for now

  runStats.tmEvalCons.start();
  bool bret = interface_base.eval_cons(nlp_transformations.n_post(),
				       n_cons, n_cons_ineq, cons_ineq_mapping,
				       xx, new_x, dd);
  runStats.tmEvalCons.stop(); runStats.nEvalCons_ineq++;

  //d = nlp_transformations.applyInvToCons(d, n_cons_ineq); //not needed for now
  return bret;
}

bool hiopNlpFormulation::eval_c_d(double*x, bool new_x, double* c, double* d)
{
  bool do_eval_c = true;
  if(-1 == cons_eval_type_) {
    assert(cons_body_ == NULL);
    assert(NULL == cons_Jac_);
    if(!eval_c(x, new_x, c)) {
      //test if eval_d also fails; this means we should use one-call constraints/Jacobian evaluation
      if(!eval_d(x, new_x, d)) {
	cons_eval_type_ = 1;
	cons_body_ = new double[n_cons];
	cons_Jac_ = alloc_Jac_cons();
      } else {
	cons_eval_type_ = 0;
	return false;
      }
    } else {
      cons_eval_type_ = 0;
      do_eval_c = false;
    }
  }

  if(0 == cons_eval_type_) {
    if(do_eval_c) if(!eval_c(x, new_x, c)) { return false; }
    if(!eval_d(x, new_x, d)) { return false; }
    return true;
  } else {
    assert(1 == cons_eval_type_);
    assert(cons_body_ != NULL);

    double* xx = nlp_transformations.applyTox(x, new_x);
    double* body = cons_body_;//nlp_transformations.applyToCons(d, n_cons_ineq); //not needed for now

    runStats.tmEvalCons.start();
    bool bret = interface_base.eval_cons(nlp_transformations.n_post(),
					 n_cons, 
					 xx, new_x, body);
    //copy back to c and d
    for(int i=0; i<n_cons_eq; ++i) {
      c[i] = body[cons_eq_mapping[i]];
    }
    for(int i=0; i<n_cons_ineq; ++i) {
      d[i] = body[cons_ineq_mapping[i]];
    }
    
    runStats.tmEvalCons.stop();
    runStats.nEvalCons_eq++;
    runStats.nEvalCons_ineq++;
    
    //d = nlp_transformations.applyInvToCons(d, n_cons_ineq); //not needed for now
    return bret;
  }
}

bool hiopNlpFormulation::eval_Jac_c_d(double* x, bool new_x, hiopMatrix& Jac_c, hiopMatrix& Jac_d)
{
  bool do_eval_Jac_c = true;
  if(-1 == cons_eval_type_) {
    assert(cons_body_ == NULL);
    assert(NULL == cons_Jac_);
    if(!eval_Jac_c(x, new_x, Jac_c)) {
      //test if eval_d also fails; this means we should use one-call constraints/Jacobian evaluation
      if(!eval_Jac_d(x, new_x, Jac_d)) {
	cons_eval_type_ = 1;
	cons_body_ = new double[n_cons];
	cons_Jac_ = alloc_Jac_cons();
      } else {
	cons_eval_type_ = 0;
	return false;
      }
    } else {
      cons_eval_type_ = 0;
      do_eval_Jac_c = false;
    }
  }

  if(0 == cons_eval_type_) {
    if(do_eval_Jac_c)
      if(!eval_Jac_c(x, new_x, Jac_c))
	return false; 
    if(!eval_Jac_d(x, new_x, Jac_d)) { return false; }
    return true;
  } else {
    assert(1 == cons_eval_type_);
    assert(cons_body_);
    assert(cons_Jac_);
    
    return eval_Jac_c_d_interface_impl(x, new_x, Jac_c, Jac_d);
  }
  return true;
}

void hiopNlpFormulation::user_callback_solution(hiopSolveStatus status,
						const hiopVector& x,
						const hiopVector& z_L,
						const hiopVector& z_U,
						const hiopVector& c,
						const hiopVector& d,
						const hiopVector& yc,
						const hiopVector& yd,
						double obj_value) 
{
  const hiopVectorPar& xp = dynamic_cast<const hiopVectorPar&>(x);
  const hiopVectorPar& zl = dynamic_cast<const hiopVectorPar&>(z_L);
  const hiopVectorPar& zu = dynamic_cast<const hiopVectorPar&>(z_U);
  assert(xp.get_size()==n_vars);
  assert(c.get_size()+d.get_size()==n_cons);
  //!petra: to do: assemble (c,d) into cons and (yc,yd) into lambda based on cons_eq_mapping
  // and cons_ineq_mapping
  interface_base.solution_callback(status, 
				   (int)n_vars, xp.local_data_const(),
				   zl.local_data_const(), zu.local_data_const(),
				   (int)n_cons, NULL, //cons, 
				   NULL, //lambda,
				   obj_value);
}

bool hiopNlpFormulation::user_callback_iterate(int iter,
					       double obj_value,
					       const hiopVector& x,
					       const hiopVector& z_L,
					       const hiopVector& z_U,
					       const hiopVector& c,
					       const hiopVector& d,
					       const hiopVector& yc,
					       const hiopVector& yd,
					       double inf_pr,
					       double inf_du,
					       double mu,
					       double alpha_du,
					       double alpha_pr,
					       int ls_trials)
{
  const hiopVectorPar& xp = dynamic_cast<const hiopVectorPar&>(x);
  const hiopVectorPar& zl = dynamic_cast<const hiopVectorPar&>(z_L);
  const hiopVectorPar& zu = dynamic_cast<const hiopVectorPar&>(z_U);
  assert(xp.get_size()==n_vars);
  assert(c.get_size()+d.get_size()==n_cons);
  //!petra: to do: assemble (c,d) into cons and (yc,yd) into lambda based on cons_eq_mapping
  //and cons_ineq_mapping
  return interface_base.iterate_callback(iter, obj_value, 
					 (int)n_vars, xp.local_data_const(),
					 zl.local_data_const(), zu.local_data_const(),
					 (int)n_cons, NULL, //cons, 
					 NULL, //lambda,
					 inf_pr, inf_du, mu, alpha_du, alpha_pr,  ls_trials);
}

void hiopNlpFormulation::print(FILE* f, const char* msg, int rank) const
{
   int myrank=0; 
#ifdef HIOP_USE_MPI
   if(rank>=0) {
     int ierr = MPI_Comm_rank(comm, &myrank); assert(ierr==MPI_SUCCESS); 
   }
#endif
  if(myrank==rank || rank==-1) {
    if(NULL==f) f=stdout;

    if(msg) {
      fprintf(f, "%s\n", msg);
    } else { 
      fprintf(f, "NLP summary\n");
    }
    fprintf(f, "Total number of variables: %lld\n", n_vars);
    fprintf(f, "     lower/upper/lower_and_upper bounds: %lld / %lld / %lld\n",
	    n_bnds_low, n_bnds_upp, n_bnds_lu);
    fprintf(f, "Total number of equality constraints: %lld\n", n_cons_eq);
    fprintf(f, "Total number of inequality constraints: %lld\n", n_cons_ineq );
    fprintf(f, "     lower/upper/lower_and_upper bounds: %lld / %lld / %lld\n",
	    n_ineq_low, n_ineq_upp, n_ineq_lu);
  } 
}

/* ***********************************************************************************
 *    hiopNlpDenseConstraints class implementation 
 * ***********************************************************************************
*/

hiopNlpDenseConstraints::hiopNlpDenseConstraints(hiopInterfaceDenseConstraints& interface_)
  : hiopNlpFormulation(interface_), interface(interface_)
{
}

hiopNlpDenseConstraints::~hiopNlpDenseConstraints()
{
}

bool hiopNlpDenseConstraints::finalizeInitialization()
{
  return hiopNlpFormulation::finalizeInitialization();
}

bool hiopNlpDenseConstraints::eval_Jac_c(double* x, bool new_x, double** Jac_c)
{
  double*  x_user      = nlp_transformations.applyTox(x, new_x);
  double** Jac_c_user = nlp_transformations.applyToJacobEq(Jac_c, n_cons_eq);

  runStats.tmEvalJac_con.start();
  bool bret = interface.eval_Jac_cons(nlp_transformations.n_post(),n_cons,n_cons_eq,cons_eq_mapping,
				      x_user,new_x,Jac_c_user);
  runStats.tmEvalJac_con.stop(); runStats.nEvalJac_con_eq++;

  Jac_c = nlp_transformations.applyInvToJacobEq(Jac_c_user, n_cons_eq);
  return bret;
}
bool hiopNlpDenseConstraints::eval_Jac_d(double* x, bool new_x, double** Jac_d)
{
  double* x_user      = nlp_transformations.applyTox(x, new_x);
  double** Jac_d_user = nlp_transformations.applyToJacobIneq(Jac_d, n_cons_ineq);
 
  runStats.tmEvalJac_con.start();
  bool bret = interface.eval_Jac_cons(nlp_transformations.n_post(),n_cons,n_cons_ineq,cons_ineq_mapping,
				      x_user,new_x,Jac_d_user);
  runStats.tmEvalJac_con.stop(); runStats.nEvalJac_con_ineq++;

  Jac_d = nlp_transformations.applyInvToJacobIneq(Jac_d_user, n_cons_ineq);
  return bret;
}

bool hiopNlpDenseConstraints::eval_Jac_c_d_interface_impl(double* x, bool new_x,
							  hiopMatrix& Jac_c,
							  hiopMatrix& Jac_d)
{
  hiopMatrixDense* Jac_cde = dynamic_cast<hiopMatrixDense*>(&Jac_c);
  hiopMatrixDense* Jac_dde = dynamic_cast<hiopMatrixDense*>(&Jac_d);
  if(Jac_cde==NULL || Jac_dde==NULL) {
    log->printf(hovError, "[internal error] hiopNlpDenseConstraints NLP works only with dense matrices\n");
    return false;
  }
  hiopMatrixDense* cons_Jac_de = dynamic_cast<hiopMatrixDense*>(cons_Jac_);
  if(cons_Jac_de == NULL) {
    log->printf(hovError, "[internal error] hiopNlpDenseConstraints NLP received an unexpected matrix\n");
    return false;
  }

  double* x_user = nlp_transformations.applyTox(x, new_x);
  double** Jac_consde = cons_Jac_de->local_data();
  double** Jac_user = nlp_transformations.applyToJacobCons(Jac_consde, n_cons);

  runStats.tmEvalJac_con.start();
  bool bret = interface.eval_Jac_cons(nlp_transformations.n_post(), n_cons,
				      x_user, new_x,
				      Jac_user);
  
  Jac_consde = nlp_transformations.applyInvToJacobCons(Jac_user, n_cons);
  assert(cons_Jac_de->local_data() == Jac_consde &&
	 "mismatch between Jacobian mem adress pre- and post-transformations should not happen");

  Jac_cde->copyRowsFrom(*cons_Jac_, cons_eq_mapping, n_cons_eq);
  Jac_dde->copyRowsFrom(*cons_Jac_, cons_ineq_mapping, n_cons_ineq);
  
  runStats.tmEvalJac_con.stop();
  runStats.nEvalJac_con_eq++;
  runStats.nEvalJac_con_ineq++;

  return bret;
}

bool hiopNlpDenseConstraints::eval_Jac_c(double* x, bool new_x, hiopMatrix& Jac_c)
{
  hiopMatrixDense* Jac_cde = dynamic_cast<hiopMatrixDense*>(&Jac_c);
  if(Jac_cde==NULL) {
    log->printf(hovError, "[internal error] hiopNlpDenseConstraints NLP works only with dense matrices\n");
    return false;
  } else {
    return this->eval_Jac_c(x, new_x, Jac_cde->local_data());
  }
}

bool hiopNlpDenseConstraints::eval_Jac_d(double* x, bool new_x, hiopMatrix& Jac_d)
{
  hiopMatrixDense* Jac_dde = dynamic_cast<hiopMatrixDense*>(&Jac_d);
  if(Jac_dde==NULL) {
    log->printf(hovError, "[internal error] hiopNlpDenseConstraints NLP works only with dense matrices\n");
    return false;
  } else {
    return this->eval_Jac_d(x, new_x, Jac_dde->local_data());
  }
}

hiopMatrixDense* hiopNlpDenseConstraints::alloc_Jac_c()
{
  return alloc_multivector_primal(n_cons_eq);
}

hiopMatrixDense* hiopNlpDenseConstraints::alloc_Jac_d()
{
  return alloc_multivector_primal(n_cons_ineq);
}

hiopMatrixDense* hiopNlpDenseConstraints::alloc_Jac_cons()
{
  return alloc_multivector_primal(n_cons);
}

hiopMatrix* hiopNlpDenseConstraints::alloc_Hess_Lagr()
{
  return new hiopHessianLowRank(this, this->options->GetInteger("secant_memory_len"));
}

hiopMatrixDense* hiopNlpDenseConstraints::alloc_multivector_primal(int nrows, int maxrows/*=-1*/) const
{
  hiopMatrixDense* M;
#ifdef HIOP_USE_MPI
  //long long* vec_distrib=new long long[num_ranks+1];
  //if(true==interface.get_vecdistrib_info(n_vars,vec_distrib)) 
  if(vec_distrib)
  {
    M = new hiopMatrixDense(nrows, n_vars, vec_distrib, comm, maxrows);
  } else {
    //the if is not really needed, but let's keep it clear, costs only a comparison
    if(-1==maxrows)
      M = new hiopMatrixDense(nrows, n_vars);   
    else
      M = new hiopMatrixDense(nrows, n_vars, NULL, MPI_COMM_SELF, maxrows);
  }
#else
  //the if is not really needed, but let's keep it clear, costs only a comparison
  if(-1==maxrows)
    M = new hiopMatrixDense(nrows, n_vars);   
  else
    M = new hiopMatrixDense(nrows, n_vars, NULL, MPI_COMM_SELF, maxrows);
#endif
  return M;
}

/* ***********************************************************************************
 *    hiopNlpMDS class implementation 
 * ***********************************************************************************
*/
bool hiopNlpMDS::eval_Jac_c(double* x, bool new_x, hiopMatrix& Jac_c)
{
  hiopMatrixMDS* pJac_c = dynamic_cast<hiopMatrixMDS*>(&Jac_c);
  assert(pJac_c);
  if(pJac_c) {
    double* x_user = nlp_transformations.applyTox(x, new_x);
    //! todo -> need hiopNlpTransformation::applyToJacobXXX to work with MDS Jacobian
    //double** Jac_c_user = nlp_transformations.applyToJacobEq(Jac_c, n_cons_eq); //!
    
    runStats.tmEvalJac_con.start();
    
    int nnz = pJac_c->sp_nnz();
    bool bret = interface.eval_Jac_cons(n_vars, n_cons, 
					n_cons_eq, cons_eq_mapping, 
					x_user, new_x,
					pJac_c->n_sp(), pJac_c->n_de(), 
					nnz, pJac_c->sp_irow(), pJac_c->sp_jcol(), pJac_c->sp_M(),
					pJac_c->de_local_data());

    //! todo -> need hiopNlpTransformation::applyInvToJacobXXX to work with MDS Jacobian
    //Jac_c = nlp_transformations.applyInvToJacobEq(Jac_c_user, n_cons_eq); //!
    runStats.tmEvalJac_con.stop();
    runStats.nEvalJac_con_eq++;
    return bret;
  } else {
    return false;
  }
}
bool hiopNlpMDS::eval_Jac_d(double* x, bool new_x, hiopMatrix& Jac_d)
{
  hiopMatrixMDS* pJac_d = dynamic_cast<hiopMatrixMDS*>(&Jac_d);
  assert(pJac_d);
  if(pJac_d) {
    double* x_user      = nlp_transformations.applyTox(x, new_x);
    //! todo -> need hiopNlpTransformation::applyToJacobXXX to work with MDS Jacobian
    //double** Jac_d_user = nlp_transformations.applyToJacobIneq(Jac_d, n_cons_ineq);
    
    runStats.tmEvalJac_con.start();
  
    int nnz = pJac_d->sp_nnz();
    bool bret =  interface.eval_Jac_cons(n_vars, n_cons, 
					 n_cons_ineq, cons_ineq_mapping, 
					 x_user, new_x,
					 pJac_d->n_sp(), pJac_d->n_de(), 
					 nnz, pJac_d->sp_irow(), pJac_d->sp_jcol(), pJac_d->sp_M(),
					 pJac_d->de_local_data());

    //! todo -> need hiopNlpTransformation::applyInvToJacobXXX to work with MDS Jacobian
    //Jac_d = nlp_transformations.applyInvToJacobIneq(Jac_d_user, n_cons_ineq);
    runStats.tmEvalJac_con.stop();
    runStats.nEvalJac_con_ineq++;
    return bret;
  } else {
    return false;
  }
}
bool hiopNlpMDS::eval_Jac_c_d_interface_impl(double* x,
					     bool new_x,
					     hiopMatrix& Jac_c,
					     hiopMatrix& Jac_d)
{
  hiopMatrixMDS* pJac_c = dynamic_cast<hiopMatrixMDS*>(&Jac_c);
  hiopMatrixMDS* pJac_d = dynamic_cast<hiopMatrixMDS*>(&Jac_d);
  hiopMatrixMDS* cons_Jac = dynamic_cast<hiopMatrixMDS*>(cons_Jac_);
  if(pJac_c && pJac_d) {
    assert(cons_Jac);
    if(NULL == cons_Jac)
      return false;

    assert(cons_Jac->n_de() == pJac_d->n_de());
    assert(cons_Jac->n_sp() == pJac_d->n_sp());
    assert(cons_Jac->sp_nnz() == pJac_c->sp_nnz() + pJac_d->sp_nnz());
    
    double* x_user      = nlp_transformations.applyTox(x, new_x);
    //! todo -> need hiopNlpTransformation::applyInvToJacobIneq to work with MDS Jacobian
    //double** Jac_d_user = nlp_transformations.applyToJacobIneq(Jac_d, n_cons_ineq);
    
    runStats.tmEvalJac_con.start();
  
    int nnz = cons_Jac->sp_nnz();
    bool bret = interface.eval_Jac_cons(n_vars, n_cons, 
					x_user, new_x,
					pJac_d->n_sp(), pJac_d->n_de(), 
					nnz, cons_Jac->sp_irow(), cons_Jac->sp_jcol(), cons_Jac->sp_M(),
					cons_Jac->de_local_data());
    //! todo -> need hiopNlpTransformation::applyInvToJacobIneq to work with MDS Jacobian
    //Jac_d = nlp_transformations.applyInvToJacobIneq(Jac_d_user, n_cons_ineq);
    
    //copy back to Jac_c and Jac_d
    pJac_c->copyRowsFrom(*cons_Jac, cons_eq_mapping, n_cons_eq);
    pJac_d->copyRowsFrom(*cons_Jac, cons_ineq_mapping, n_cons_ineq);
    
    runStats.tmEvalJac_con.stop();
    runStats.nEvalJac_con_eq++;
    runStats.nEvalJac_con_ineq++;
    
    return bret;
  } else {
    return false;
  }
  return true;
}

bool hiopNlpMDS::eval_Hess_Lagr(const double* x, bool new_x, const double& obj_factor,
			      const double* lambda_eq, const double* lambda_ineq, bool new_lambdas,
			      hiopMatrix& Hess_L)
{
  hiopMatrixSymBlockDiagMDS* pHessL = dynamic_cast<hiopMatrixSymBlockDiagMDS*>(&Hess_L);
  assert(pHessL);
  if(pHessL) {
    
    if(n_cons_eq + n_cons_ineq != _buf_lambda->get_size()) {
      delete _buf_lambda;
      _buf_lambda = NULL;
	_buf_lambda = new hiopVectorPar(n_cons_eq + n_cons_ineq);
    }
    assert(_buf_lambda);
    _buf_lambda->copyFromStarting(0,         lambda_eq,   n_cons_eq);
    _buf_lambda->copyFromStarting(n_cons_eq, lambda_ineq, n_cons_ineq);
    
    int nnzHSS = pHessL->sp_nnz(), nnzHSD = 0;
    bool bret = interface.eval_Hess_Lagr(n_vars, n_cons, x, new_x, 
					 obj_factor, _buf_lambda->local_data(), new_lambdas, 
					 pHessL->n_sp(), pHessL->n_de(),
					 nnzHSS, pHessL->sp_irow(), pHessL->sp_jcol(), pHessL->sp_M(),
					 pHessL->de_local_data(),
					 nnzHSD, NULL, NULL, NULL);
    assert(nnzHSD==0);
    assert(nnzHSS==pHessL->sp_nnz());
    return bret;
  } else {
    return false;
  }
}

bool hiopNlpMDS::finalizeInitialization()
{
  if(!interface.get_sparse_dense_blocks_info(nx_sparse, nx_dense,
					     nnz_sparse_Jaceq, nnz_sparse_Jacineq,
					     nnz_sparse_Hess_Lagr_SS, 
					     nnz_sparse_Hess_Lagr_SD)) {
    return false;
  }
  assert(0==nnz_sparse_Hess_Lagr_SD);
  return hiopNlpFormulation::finalizeInitialization();
}

};
