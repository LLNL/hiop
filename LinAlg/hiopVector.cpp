#include "hiopVector.hpp"

#include <cmath>
#include <cstring> //for memcpy
#include <cassert>

#include "blasdefs.hpp"

hiopVectorPar::hiopVectorPar(const long long& glob_n, long long* col_part/*=NULL*/, MPI_Comm comm_/*=MPI_COMM_NULL*/)
  : comm(comm_)
{
  n = glob_n;

#ifdef WITH_MPI
  // if this is a serial vector, make sure it has a valid comm in the mpi case
  if(comm==MPI_COMM_NULL) comm=MPI_COMM_SELF;
#endif

  int P=0; 
  if(col_part) {
#ifdef WITH_MPI
    int ierr=MPI_Comm_rank(comm, &P);  assert(ierr==MPI_SUCCESS);
#endif
    glob_il=col_part[P]; glob_iu=col_part[P+1];
  } else { 
    glob_il=0; glob_iu=n;
  }   
  n_local=glob_iu-glob_il;

  data = new double[n_local];
}
hiopVectorPar::hiopVectorPar(const hiopVectorPar& v)
{
  n_local=v.n_local; n = v.n;
  glob_il=v.glob_il; glob_iu=v.glob_iu;
  comm=v.comm;
  data=new double[n_local];  
}
hiopVectorPar::~hiopVectorPar()
{
  delete[] data; data=NULL;
}

hiopVectorPar* hiopVectorPar::alloc_clone() const
{
  hiopVectorPar* v = new hiopVectorPar(*this); assert(v);
  return v;
}
hiopVectorPar* hiopVectorPar::new_copy () const
{
  hiopVectorPar* v = new hiopVectorPar(*this); assert(v);
  v->copyFrom(*this);
  return v;
}

//hiopVector* hiopVectorPar::new_alloc() const
//{ }
//hiopVector* hiopVectorPar::new_copy() const
//{ }


void hiopVectorPar::setToZero()
{
  for(int i=0; i<n_local; i++) data[i]=0.0;
}
void hiopVectorPar::setToConstant(double c)
{
  for(int i=0; i<n_local; i++) data[i]=c;
}
void hiopVectorPar::setToConstant_w_patternSelect(double c, const hiopVector& select)
{
  const hiopVectorPar& s = dynamic_cast<const hiopVectorPar&>(select);
  const double* svec = s.data;
  for(int i=0; i<n_local; i++) if(svec[i]==1.) data[i]=c; else data[i]=0.;
}
void hiopVectorPar::copyFrom(const hiopVector& v_ )
{
  const hiopVectorPar& v = dynamic_cast<const hiopVectorPar&>(v_);
  assert(n_local==v.n_local);
  assert(glob_il==v.glob_il); assert(glob_iu==v.glob_iu);
  memcpy(this->data, v.data, n_local*sizeof(double));
}

void hiopVectorPar::copyFromStarting(const hiopVector& v_, int start_index)
{
  const hiopVectorPar& v = dynamic_cast<const hiopVectorPar&>(v_);
#ifdef DEEP_CHECKING
  assert(n_local==n && "are you sure you want to call this?");
#endif
  assert(start_index+v.n_local <= n_local);
  memcpy(data+start_index, v.data, v.n_local*sizeof(double));
}

void hiopVectorPar::copyToStarting(hiopVector& v_, int start_index)
{
  const hiopVectorPar& v = dynamic_cast<const hiopVectorPar&>(v_);
#ifdef DEEP_CHECKING
  assert(n_local==n && "are you sure you want to call this?");
#endif
  assert(start_index+v.n_local <= n_local);
  memcpy(v.data, data+start_index, v.n_local*sizeof(double));
}

double hiopVectorPar::twonorm() const 
{
  int one=1; int n=n_local;
  double nrm = dnrm2_(&n,data,&one); 

#ifdef WITH_MPI
  nrm *= nrm;
  double nrmG;
  int ierr = MPI_Allreduce(&nrm, &nrmG, 1, MPI_DOUBLE, MPI_SUM, comm); assert(MPI_SUCCESS==ierr);
  nrm=sqrt(nrmG);
#endif  
  return nrm;
}

double hiopVectorPar::dotProductWith( const hiopVector& v_ ) const
{
  const hiopVectorPar& v = dynamic_cast<const hiopVectorPar&>(v_);
  int one=1; int n=n_local;
  assert(this->n_local==v.n_local);

  double dotprod=ddot_(&n, this->data, &one, v.data, &one);

#ifdef WITH_MPI
  double dotprodG;
  int ierr = MPI_Allreduce(&dotprod, &dotprodG, 1, MPI_DOUBLE, MPI_SUM, comm); assert(MPI_SUCCESS==ierr);
  dotprod=dotprodG;
#endif

  return dotprod;
}

double hiopVectorPar::infnorm() const
{
  if(n_local<=0) return 0.;
  double nrm=fabs(data[0]), aux;
  
  for(int i=1; i<n_local; i++) {
    aux=fabs(data[i]);
    if(aux>nrm) nrm=aux;
  }
#ifdef WITH_MPI
  double nrm_glob;
  int ierr = MPI_Allreduce(&nrm, &nrm_glob, 1, MPI_DOUBLE, MPI_MAX, comm); assert(MPI_SUCCESS==ierr);
  return nrm_glob;
#endif

  return nrm;
}

double hiopVectorPar::infnorm_local() const
{
  if(n_local<=0) return 0.;
  double nrm=fabs(data[0]), aux;
  
  for(int i=1; i<n_local; i++) {
    aux=fabs(data[i]);
    if(aux>nrm) nrm=aux;
  }
  return nrm;
}


double hiopVectorPar::onenorm() const
{
  double nrm1=0.; for(int i=0; i<n_local; i++) nrm1 += fabs(data[i]);
#ifdef WITH_MPI
  double nrm1_global;
  int ierr = MPI_Allreduce(&nrm1, &nrm1_global, 1, MPI_DOUBLE, MPI_SUM, comm); assert(MPI_SUCCESS==ierr);
  return nrm1_global;
#endif
  return nrm1;
}

double hiopVectorPar::onenorm_local() const
{
  double nrm1=0.; for(int i=0; i<n_local; i++) nrm1 += fabs(data[i]);
  return nrm1;
}

void hiopVectorPar::componentMult( const hiopVector& v_ )
{
  const hiopVectorPar& v = dynamic_cast<const hiopVectorPar&>(v_);
  assert(n_local==v.n_local);
  //for(int i=0; i<n_local; i++) data[i] *= v.data[i];
  double*s = data,*x=v.data; 
  int n2=(n_local/8)*8; double *send2=s+n2, *send=s+n_local;
  while(s<send2) {
    *s *= *x; s++; x++; //s[i] *= x[i]; i++;      
    *s *= *x; s++; x++;
    *s *= *x; s++; x++;
    *s *= *x; s++; x++;
    *s *= *x; s++; x++;
    *s *= *x; s++; x++;
    *s *= *x; s++; x++;
    *s *= *x; s++; x++;
  }
  while(s<send) { *s *= *x; s++; x++; }
}

void hiopVectorPar::componentDiv ( const hiopVector& v_ )
{
  const hiopVectorPar& v = dynamic_cast<const hiopVectorPar&>(v_);
  assert(n_local==v.n_local);
  for(int i=0; i<n_local; i++) data[i] /= v.data[i];
}

void hiopVectorPar::componentDiv_p_selectPattern( const hiopVector& v_, const hiopVector& ix_)
{
  const hiopVectorPar& v = dynamic_cast<const hiopVectorPar&>(v_);
  const hiopVectorPar& ix= dynamic_cast<const hiopVectorPar&>(ix_);
#ifdef DEEP_CHECKING
  assert(v.n_local==n_local);
  assert(n_local==ix.n_local);
#endif
  double *s=this->data, *x=v.data, *pattern=ix.data; 
  for(int i=0; i<n_local; i++)
    if(pattern[i]==0.0) s[i]=0.0;
    else                s[i]/=x[i];
}

void hiopVectorPar::scale(double num)
{
  if(1.0==num) return;
  int one=1; int n=n_local;
  dscal_(&n, &num, data, &one);
}

void hiopVectorPar::axpy(double alpha, const hiopVector& x_)
{
  const hiopVectorPar& x = dynamic_cast<const hiopVectorPar&>(x_);
  int one = 1; int n=n_local;
  daxpy_( &n, &alpha, x.data, &one, data, &one );
}

void hiopVectorPar::axzpy(double alpha, const hiopVector& x_, const hiopVector& z_)
{
  const hiopVectorPar& vx = dynamic_cast<const hiopVectorPar&>(x_);
  const hiopVectorPar& vz = dynamic_cast<const hiopVectorPar&>(z_);
#ifdef DEEP_CHECKING
  assert(vx.n_local==vz.n_local);
  assert(   n_local==vz.n_local);
#endif  
  // this += alpha * x * z   (s+=alpha*x*z)
  double*s = data;
  const double *x = vx.local_data_const(), *z=vz.local_data_const();

  //unroll loops to save on comparison; hopefully the compiler will take it from here
  int nn=(n_local/8)*8; double *send1=s+nn, *send2=s+n_local;

  if(alpha== 1.0) { 
    while(s<send1) {
      *s += *x * *z; s++; x++; z++; //s[i] += x[i]*z[i]; i++;      
      *s += *x * *z; s++; x++; z++; 
      *s += *x * *z; s++; x++; z++;
      *s += *x * *z; s++; x++; z++;
      *s += *x * *z; s++; x++; z++; 
      *s += *x * *z; s++; x++; z++;
      *s += *x * *z; s++; x++; z++;
      *s += *x * *z; s++; x++; z++;
    }
    while(s<send2) { *s += *x * *z; s++; x++; z++; }

  } else if(alpha==-1.0) { 
    while(s<send1) {
      *s -= *x * *z; s++; x++; z++; //s[i] += x[i]*z[i]; i++;      
      *s -= *x * *z; s++; x++; z++; 
      *s -= *x * *z; s++; x++; z++;
      *s -= *x * *z; s++; x++; z++;
      *s -= *x * *z; s++; x++; z++; 
      *s -= *x * *z; s++; x++; z++;
      *s -= *x * *z; s++; x++; z++;
      *s -= *x * *z; s++; x++; z++;
    }
    while(s<send2) { 
      *s -= *x * *z; s++; x++; z++; 
    }    

  } else { // alpha is neither 1.0 nor -1.0
    while(s<send1) {
      *s += *x * *z * alpha; s++; x++; z++; //s[i] += x[i]*z[i]; i++;      
      *s += *x * *z * alpha; s++; x++; z++; 
      *s += *x * *z * alpha; s++; x++; z++; 
      *s += *x * *z * alpha; s++; x++; z++; 
      *s += *x * *z * alpha; s++; x++; z++; 
      *s += *x * *z * alpha; s++; x++; z++; 
      *s += *x * *z * alpha; s++; x++; z++; 
    }
    while(s<send2) { *s += *x * *z * alpha; s++; x++; z++; }
  }
}

void hiopVectorPar::axdzpy( double alpha, const hiopVector& x_, const hiopVector& z_)
{
  const hiopVectorPar& vx = dynamic_cast<const hiopVectorPar&>(x_);
  const hiopVectorPar& vz = dynamic_cast<const hiopVectorPar&>(z_);
#ifdef DEEP_CHECKING
  assert(vx.n_local==vz.n_local);
  assert(   n_local==vz.n_local);
#endif  
  // this += alpha * x / z   (s+=alpha*x/z)
  double*s = data;
  const double *x = vx.local_data_const(), *z=vz.local_data_const();

  //unroll loops to save on comparison; hopefully the compiler will take it from here
  int nn=(n_local/3)*3; double *send1=s+nn, *send2=s+n_local;

  if(alpha== 1.0) { 
    while(s<send1) {
      *s += *x / *z; s++; x++; z++; //s[i] += x[i]*z[i]; i++;      
      *s += *x / *z; s++; x++; z++; 
      *s += *x / *z; s++; x++; z++;
      //*s += *x / *z; s++; x++; z++;
      //*s += *x / *z; s++; x++; z++; 
      //*s += *x / *z; s++; x++; z++;
      //*s += *x / *z; s++; x++; z++;
      //*s += *x / *z; s++; x++; z++;
    }
    while(s<send2) { *s += *x * *z; s++; x++; z++; }

  } else if(alpha==-1.0) { 
    while(s<send1) {
      *s -= *x / *z; s++; x++; z++; //s[i] += x[i]*z[i]; i++;      
      *s -= *x / *z; s++; x++; z++; 
      *s -= *x / *z; s++; x++; z++;
      //*s -= *x / *z; s++; x++; z++;
      //*s -= *x / *z; s++; x++; z++; 
      //*s -= *x / *z; s++; x++; z++;
      //*s -= *x / *z; s++; x++; z++;
      //*s -= *x / *z; s++; x++; z++;
    }
    while(s<send2) { 
      *s -= *x / *z; s++; x++; z++; 
    }    

  } else { // alpha is neither 1.0 nor -1.0
    while(s<send1) {
      *s += *x / *z * alpha; s++; x++; z++; //s[i] += x[i]*z[i]; i++;      
      *s += *x / *z * alpha; s++; x++; z++; 
      *s += *x / *z * alpha; s++; x++; z++; 
      //!opt *s += *x / *z * alpha; s++; x++; z++; 
      //*s += *x / *z * alpha; s++; x++; z++; 
      //*s += *x / *z * alpha; s++; x++; z++; 
      //*s += *x / *z * alpha; s++; x++; z++; 
      //*s += *x / *z * alpha; s++; x++; z++; 
    }
    while(s<send2) { *s += *x / *z * alpha; s++; x++; z++; }
  }
}

void hiopVectorPar::axdzpy_w_pattern( double alpha, const hiopVector& x_, const hiopVector& z_, const hiopVector& select)
{
  const hiopVectorPar& vx = dynamic_cast<const hiopVectorPar&>(x_);
  const hiopVectorPar& vz = dynamic_cast<const hiopVectorPar&>(z_);
  const hiopVectorPar& sel= dynamic_cast<const hiopVectorPar&>(select);
#ifdef DEEP_CHECKING
  assert(vx.n_local==vz.n_local);
  assert(   n_local==vz.n_local);
#endif  
  // this += alpha * x / z   (y+=alpha*x/z)
  double*y = data;
  const double *x = vx.local_data_const(), *z=vz.local_data_const(), *s=sel.local_data_const();
  int it;
  if(alpha==1.0) {
    for(it=0;it<n_local;it++)
      if(s[it]==1.0) y[it] += x[it]/z[it];
  } else 
    if(alpha==-1.0) {
      for(it=0; it<n_local;it++)
	if(s[it]==1.0) y[it] -= x[it]/z[it];
    } else
      for(it=0; it<n_local; it++)
	if(s[it]==1.0) y[it] += alpha*x[it]/z[it];
}


void hiopVectorPar::addConstant( double c )
{
  assert(false && "not implemented");
}

void  hiopVectorPar::addConstant_w_patternSelect(double c, const hiopVector& ix_)
{
  const hiopVectorPar& ix = dynamic_cast<const hiopVectorPar&>(ix_);
  assert(this->n_local == ix.n_local);
  const double* ix_vec = ix.data;
  for(int i=0; i<n_local; i++) if(ix_vec[i]==1.) data[i]+=c;
}

void hiopVectorPar::min( double& m, int& index ) const
{
  assert(false && "not implemented");
}

void hiopVectorPar::negate()
{
  double minusOne=-1.0; int one=1, n=n_local;
  dscal_(&n, &minusOne, data, &one);
}

void hiopVectorPar::invert()
{
  for(int i=0; i<n_local; i++) {
#ifdef DEEP_CHECKING
    if(fabs(data[i])<1e-35) assert(false);
#endif
    data[i]=1/data[i];
  }
}

double hiopVectorPar::logBarrier(const hiopVector& select) const 
{
  double res=0.0;
  const hiopVectorPar& ix = dynamic_cast<const hiopVectorPar&>(select);
  assert(this->n_local == ix.n_local);
  const double* ix_vec = ix.data;
  for(int i=0; i<n_local; i++) 
    if(ix_vec[i]==1.) 
      res += log(data[i]);
  return res;
}

int hiopVectorPar::allPositive()
{
  int allPos=true, i=0;
  while(i<n_local && allPos) if(data[i++]<=0) allPos=false;

#ifdef WITH_MPI
  int allPosG;
  int ierr=MPI_Allreduce(&allPos, &allPosG, 1, MPI_INT, MPI_MIN, comm); assert(MPI_SUCCESS==ierr);
  return allPosG;
#endif
  return allPos;
}

void hiopVectorPar::projectIntoBounds(const hiopVector& xl_, const hiopVector& ixl_, 
				      const hiopVector& xu_, const hiopVector& ixu_,
				      double kappa1, double kappa2)
{

#ifdef DEEP_CHECKING
  assert((dynamic_cast<const hiopVectorPar&>(xl_) ).n_local==n_local);
  assert((dynamic_cast<const hiopVectorPar&>(ixl_)).n_local==n_local);
  assert((dynamic_cast<const hiopVectorPar&>(xu_) ).n_local==n_local);
  assert((dynamic_cast<const hiopVectorPar&>(ixu_)).n_local==n_local);
#endif
  const double* xl = (dynamic_cast<const hiopVectorPar&>(xl_) ).local_data_const();
  const double* ixl= (dynamic_cast<const hiopVectorPar&>(ixl_)).local_data_const();
  const double* xu = (dynamic_cast<const hiopVectorPar&>(xu_) ).local_data_const();
  const double* ixu= (dynamic_cast<const hiopVectorPar&>(ixu_)).local_data_const();
  double* x0=data; 

  double aux, aux2;
  for(long long i=0; i<n_local; i++) {
    if(ixl[i]!=0 && ixu[i]!=0) {
      aux=kappa2*(xu[i]-xl[i]);
      aux2=xl[i]+fmin(kappa1*fmax(1, fabs(xl[i])),aux);
      if(x0[i]<aux2) {
	x0[i]=aux2;
      } else {
	aux2=xu[i]-fmin(kappa1*fmax(1, fabs(xu[i])),aux);
	if(x0[i]>aux2) {
	  x0[i]=aux2;
	}
      }
#ifdef DEEP_CHECKING
      assert(x0[i]>xl[i] && x0[i]<xu[i]);
#endif
    } else {
      if(ixl[i]!=0.)
	x0[i] = fmax(x0[i], xl[i]+kappa1*fmax(1, fabs(xl[i])));
      else 
	if(ixu[i]!=0)
	  x0[i] = fmin(x0[i], xu[i]-kappa1*fmax(1, fabs(xu[i])));
	else { /*nothing for free vars  */ }
    }
  }
}

/* max{a\in(0,1]| x+ad >=(1-tau)x} */
double hiopVectorPar::fractionToTheBdry(const hiopVector& dx, const double& tau) const 
{
#ifdef DEEP_CHECKING
  assert((dynamic_cast<const hiopVectorPar&>(dx) ).n_local==n_local);
  assert(tau>0);
  assert(tau<1);
#endif
  double alpha=1.0, aux;
  const double* d = (dynamic_cast<const hiopVectorPar&>(dx) ).local_data_const();
  const double* x = data;
  for(int i=0; i<n_local; i++) {
#ifdef DEEP_CHECKING
    assert(x[i]>0);
#endif
    if(d[i]>=0) continue;
    aux = -tau*x[i]/d[i];
    if(aux<alpha) alpha=aux;
  }
  return alpha;
}
/* max{a\in(0,1]| x+ad >=(1-tau)x} */
double hiopVectorPar::fractionToTheBdry_w_pattern(const hiopVector& dx, const double& tau, const hiopVector& ix) const 
{
#ifdef DEEP_CHECKING
  assert((dynamic_cast<const hiopVectorPar&>(dx) ).n_local==n_local);
  assert((dynamic_cast<const hiopVectorPar&>(ix) ).n_local==n_local);
  assert(tau>0);
  assert(tau<1);
#endif
  double alpha=1.0, aux;
  const double* d = (dynamic_cast<const hiopVectorPar&>(dx) ).local_data_const();
  const double* x = data;
  const double* pat = (dynamic_cast<const hiopVectorPar&>(ix) ).local_data_const();
  for(int i=0; i<n_local; i++) {
    if(d[i]>=0) continue;
    if(pat[i]==0) continue;
#ifdef DEEP_CHECKING
    assert(x[i]>0);
#endif
    aux = -tau*x[i]/d[i];
    if(aux<alpha) alpha=aux;
  }
  return alpha;
}

void hiopVectorPar::selectPattern(const hiopVector& ix_)
{
#ifdef DEEP_CHECKING
  assert((dynamic_cast<const hiopVectorPar&>(ix_) ).n_local==n_local);
#endif
  const double* ix = (dynamic_cast<const hiopVectorPar&>(ix_) ).local_data_const();
  double* x=data;
  for(int i=0; i<n_local; i++) if(ix[i]==0.0) x[i]=0.0;
}

bool hiopVectorPar::matchesPattern(const hiopVector& ix_)
{
#ifdef DEEP_CHECKING
  assert((dynamic_cast<const hiopVectorPar&>(ix_) ).n_local==n_local);
#endif
  const double* ix = (dynamic_cast<const hiopVectorPar&>(ix_) ).local_data_const();
  bool bmatches=true;
  double* x=data;
  for(int i=0; (i<n_local) && bmatches; i++) 
    if(ix[i]==0.0 && x[i]!=0.0) bmatches=false; 

#ifdef WITH_MPI
  int bmatches_glob;
  int ierr=MPI_Allreduce(&bmatches, &bmatches_glob, 1, MPI_C_BOOL, MPI_LAND, comm); assert(MPI_SUCCESS==ierr);
  return bmatches_glob;
#endif

  return bmatches;
}

int hiopVectorPar::allPositive_w_patternSelect(const hiopVector& w_)
{
#ifdef DEEP_CHECKING
  assert((dynamic_cast<const hiopVectorPar&>(w_) ).n_local==n_local);
#endif 
  const double* w = (dynamic_cast<const hiopVectorPar&>(w_) ).local_data_const();
  const double* x=data;
  int allPos=1; 
  for(int i=0; i<n_local && allPos; i++) 
    if(w[i]!=0.0 && x[i]<=0.) allPos=0;
  
#ifdef WITH_MPI
  int allPosG;
  int ierr = MPI_Allreduce(&allPos, &allPosG, 1, MPI_INT, MPI_MIN, comm); assert(MPI_SUCCESS==ierr);
  return allPosG;
#endif  
  return allPos;
}

void hiopVectorPar::print(const char* msg/*=NULL*/, int max_elems/*=-1*/, int rank/*=-1*/) const 
{
  int myrank=0, numranks=1; 
#ifdef WITH_MPI
  if(rank>=0) {
    int err = MPI_Comm_rank(comm, &myrank); assert(err==MPI_SUCCESS);
    err = MPI_Comm_size(comm, &numranks); assert(err==MPI_SUCCESS);
  }
#endif
  if(myrank==rank || rank==-1) {
    if(max_elems>n_local) max_elems=n_local;

    if(NULL==msg) {
      if(numranks>1)
	printf("vector of size %d, printing %d elems (on rank=%d)\n", n, max_elems, myrank);
      else
	printf("vector of size %d, printing %d elems (serial)\n", n, max_elems);
    } else {
      printf("%s: ", msg);
    }    
    max_elems = max_elems>=0?max_elems:n_local;
    for(int it=0; it<max_elems; it++)  printf("%12.8e ", data[it]);
    printf("\n");
  }
}
