#include "hiopNlpFormulation.hpp"

#include "hiopLogger.hpp"

#ifdef WITH_MPI
#include "mpi.h"
#else 
#include <cstddef>
#endif

#include <cassert>

hiopNlpFormulation::hiopNlpFormulation(hiopInterfaceBase& interface)
{
  //log = new hiopLogger(hovLinAlgScalars,stdout);
  log = new hiopLogger(this,hovSummary,stdout);
  //log = new hiopLogger(hovLinesearch,stdout);
  //log = new hiopLogger(this,hovScalars,stdout);

#ifdef WITH_MPI
  assert(interface.get_MPI_comm(comm));
  assert(MPI_SUCCESS==MPI_Comm_rank(comm, &rank));
#endif
}

hiopNlpFormulation::~hiopNlpFormulation()
{
  delete log;
}


hiopNlpDenseConstraints::hiopNlpDenseConstraints(hiopInterfaceDenseConstraints& interface_)
  : hiopNlpFormulation(interface_), interface(interface_)
{
  assert(interface.get_prob_sizes(n_vars, n_cons));
#ifdef WITH_MPI
  
  int numRanks; 
  int ierr=MPI_Comm_size(comm, &numRanks); assert(MPI_SUCCESS==ierr);
  long long* columns_partitioning=new long long[numRanks+1];
  if(true==interface.get_vecdistrib_info(n_vars,columns_partitioning)) {
    xl = new hiopVectorPar(n_vars, columns_partitioning, comm);
  } else {
    xu = new hiopVectorPar(n_vars);   
  }
  delete[] columns_partitioning;
#else
  xl   = new hiopVectorPar(n_vars);
#endif  
  xu = xl->alloc_clone();

  int nlocal=xl->get_local_size();
  double  *xl_vec= xl->local_data(),  *xu_vec= xu->local_data();
  vars_type = new hiopInterfaceBase::NonlinearityType[nlocal];
  bool bret=interface.get_vars_info(n_vars,xl_vec,xu_vec,vars_type); assert(bret);
  //allocate and build ixl(ow) and ix(upp) vectors
  ixl = xu->alloc_clone(); ixu = xu->alloc_clone();
  n_bnds_low_local = n_bnds_upp_local = 0;
  double *ixl_vec=ixl->local_data(), *ixu_vec=ixu->local_data();
  for(int i=0;i<nlocal; i++) {
    if(xl_vec[i]>-1e20){ ixl_vec[i]=1.; n_bnds_low_local++;}
    else                 ixl_vec[i]=0.;
    if(xu_vec[i]< 1e20){ ixu_vec[i]=1.; n_bnds_upp_local++;}
    else                 ixu_vec[i]=0.;
    //ixl_vec[i] = xl_vec[i]<=-1e20?0.:1.;
    //ixu_vec[i] = xu_vec[i]>= 1e20?0.:1.;
  }

  /* split the constraints */
  hiopVectorPar* gl = new hiopVectorPar(n_cons); 
  hiopVectorPar* gu = new hiopVectorPar(n_cons);
  double *gl_vec=gl->local_data(), *gu_vec=gu->local_data();
  hiopInterfaceBase::NonlinearityType* cons_type = new hiopInterfaceBase::NonlinearityType[n_cons];
  bret = interface.get_cons_info(n_cons, gl_vec, gu_vec, cons_type); assert(bret);

  assert(gl->get_local_size()==n_cons);
  assert(gl->get_local_size()==n_cons);
  n_cons_eq=n_cons_ineq=0; 
  for(int i=0;i<n_cons; i++) {
    if(gl_vec[i]==gu_vec[i]) n_cons_eq++;
    else                     n_cons_ineq++;
  }

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
      cons_ineq_type[it_ineq]=cons_type[i];
      dlvec[it_ineq]=gl_vec[i]; duvec[it_ineq]=gu_vec[i]; 
      cons_ineq_mapping[it_ineq]=i;
      it_ineq++;
    }
  }
  assert(it_eq==n_cons_eq); assert(it_ineq==n_cons_ineq);
  /* delete the temporary buffers */
  delete gl,gu; delete[] cons_type;

  /* iterate over the inequalities and build the idl(ow) and idu(pp) vectors */
  idl = dl->alloc_clone(); idu=du->alloc_clone();
  n_ineq_low=n_ineq_upp=0.;
  double* idl_vec=idl->local_data(); double* idu_vec=idu->local_data();
  double* dl_vec = dl->local_data(); double* du_vec = du->local_data();
  for(int i=0; i<n_cons_ineq; i++) {
    if(dl_vec[i]>-1e20) { idl_vec[i]=1.; n_ineq_low++; }
    else                  idl_vec[i]=0.;
    if(du_vec[i]< 1e20) { idu_vec[i]=1.; n_ineq_upp++; }
    else                  idu_vec[i]=0.;
    //idl_vec[i] = dl_vec[i]<=-1e20?0.:1.;
    //idu_vec[i] = du_vec[i]>= 1e20?0.:1.;
  }
  //compute the overall n_low and n_upp
#ifdef WITH_MPI
  long long aux[2]={n_bnds_low_local, n_bnds_upp_local}, aux_g[2];
  int err=MPI_Allreduce(aux, aux_g, 2, MPI_LONG_LONG, MPI_SUM, comm); assert(MPI_SUCCESS==ierr);
  n_bnds_low=aux_g[0]; n_bnds_upp=aux_g[1];
#else
  n_bnds_low=n_bnds_low_local; n_bnds_upp=n_bnds_upp_local;
#endif

}

hiopNlpDenseConstraints::~hiopNlpDenseConstraints()
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
}


bool hiopNlpDenseConstraints::eval_f(const double* x, bool new_x, double& f)
{
  return interface.eval_f(n_vars,x,new_x,f);
}
bool hiopNlpDenseConstraints::eval_grad_f(const double* x, bool new_x, double* gradf)
{
  return interface.eval_grad_f(n_vars,x,new_x,gradf);
}
bool hiopNlpDenseConstraints::eval_Jac_c(const double* x, bool new_x, double** Jac_c)
{
  return interface.eval_Jac_cons(n_vars,n_cons,n_cons_eq,cons_eq_mapping,x,new_x,Jac_c);
}
bool hiopNlpDenseConstraints::eval_Jac_d(const double* x, bool new_x, double** Jac_d)
{
  return interface.eval_Jac_cons(n_vars,n_cons,n_cons_ineq,cons_ineq_mapping,x,new_x,Jac_d);
}
bool hiopNlpDenseConstraints::eval_c(const double*x, bool new_x, double* c)
{
  return interface.eval_cons(n_vars,n_cons,n_cons_eq,cons_eq_mapping,x,new_x,c);
}
bool hiopNlpDenseConstraints::eval_d(const double*x, bool new_x, double* d)
{
  return interface.eval_cons(n_vars,n_cons,n_cons_ineq,cons_ineq_mapping,x,new_x,d);
}
bool  hiopNlpDenseConstraints::eval_d(const hiopVector& x_, bool new_x, hiopVector& d_)
{
  const hiopVectorPar &x = dynamic_cast<const hiopVectorPar&>(x_);
  hiopVectorPar &d = dynamic_cast<hiopVectorPar&>(d_);
  return interface.eval_cons(n_vars,n_cons,n_cons_ineq,cons_ineq_mapping,x.local_data_const(),new_x,d.local_data());
}
hiopVector* hiopNlpDenseConstraints::alloc_primal_vec() const
{
  return xl->alloc_clone();
}

hiopVector* hiopNlpDenseConstraints::alloc_dual_eq_vec() const
{
  return c_rhs->alloc_clone();
}
hiopVector* hiopNlpDenseConstraints::alloc_dual_ineq_vec() const
{
  return dl->alloc_clone();
}
hiopVector* hiopNlpDenseConstraints::alloc_dual_vec() const
{
  hiopVectorPar* ret=new hiopVectorPar(n_cons);
#ifdef DEEP_CHECKING
  assert(ret!=NULL);
#endif
  return ret;

}
hiopMatrixDense* hiopNlpDenseConstraints::alloc_Jac_c() const
{
  return alloc_multivector_primal(n_cons_eq);
}

hiopMatrixDense* hiopNlpDenseConstraints::alloc_Jac_d() const
{
  /*  hiopMatrixDense* M;
#ifdef WITH_MPI
  int numRanks; 
  int ierr=MPI_Comm_size(comm, &numRanks); assert(MPI_SUCCESS==ierr);
  long long* columns_partitioning=new long long[numRanks+1];
  if(true==interface.get_vecdistrib_info(n_vars,columns_partitioning)) {
    M = new hiopMatrixDense(n_cons_ineq, n_vars, columns_partitioning, comm);
  } else {
    M = new hiopMatrixDense(n_cons_ineq, n_vars);   
  }
  delete[] columns_partitioning;
#else
  M = new hiopMatrixDense(n_cons_ineq, n_vars);   
#endif
  return M;
  */
  return alloc_multivector_primal(n_cons_ineq);
}
hiopMatrixDense* hiopNlpDenseConstraints::alloc_multivector_primal(int nrows, int maxrows/*=-1*/) const
{
  hiopMatrixDense* M;
#ifdef WITH_MPI
  int numRanks; 
  int ierr=MPI_Comm_size(comm, &numRanks); assert(MPI_SUCCESS==ierr);
  long long* columns_partitioning=new long long[numRanks+1];
  if(true==interface.get_vecdistrib_info(n_vars,columns_partitioning)) {
    M = new hiopMatrixDense(nrows, n_vars, columns_partitioning, comm, maxrows);
  } else {
    //the if is not really needed, but let's keep it clear, costs only a comparison
    if(-1==maxrows)
      M = new hiopMatrixDense(nrows, n_vars);   
    else
      M = new hiopMatrixDense(nrows, n_vars, NULL, MPI_COMM_SELF, maxrows);
  }
  delete[] columns_partitioning;
#else
  //the if is not really needed, but let's keep it clear, costs only a comparison
  if(-1==maxrows)
    M = new hiopMatrixDense(nrows, n_vars);   
  else
    M = new hiopMatrixDense(nrows, n_vars, NULL, MPI_COMM_SELF, maxrows);
#endif
  return M;
}

bool hiopNlpDenseConstraints::get_starting_point(hiopVector& x0_)
{
  hiopVectorPar &x0 = dynamic_cast<hiopVectorPar&>(x0_);
  return interface.get_starting_point(n_vars,x0.local_data());
}
