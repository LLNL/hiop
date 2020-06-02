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

#include "hiopVector.hpp"

#include <cmath>
#include <cstring> //for memcpy
#include <algorithm>
#include <cassert>

#include "hiop_blasdefs.hpp"

#include <limits>
#include <cstddef>

namespace hiop
{


hiopVectorPar::hiopVectorPar(const long long& glob_n, long long* col_part/*=NULL*/, MPI_Comm comm_/*=MPI_COMM_NULL*/)
  : comm(comm_)
{
  n = glob_n;
  assert(n>=0);
#ifdef HIOP_USE_MPI
  // if this is a serial vector, make sure it has a valid comm in the mpi case
  if(comm==MPI_COMM_NULL) comm=MPI_COMM_SELF;
#endif

  int P=0; 
  if(col_part) {
#ifdef HIOP_USE_MPI
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

void hiopVectorPar::copyFrom(const double* v_local_data )
{
  if(v_local_data)
    memcpy(this->data, v_local_data, n_local*sizeof(double));
}

void hiopVectorPar::copyFromStarting(int start_index_in_this, const double* v, int nv)
{
  assert(start_index_in_this+nv <= n_local);
  memcpy(data+start_index_in_this, v, nv*sizeof(double));
}

void hiopVectorPar::copyFromStarting(int start_index/*_in_src*/,const hiopVector& v_)
{
#ifdef HIOP_DEEPCHECKS
  assert(n_local==n && "only for local/non-distributed vectors");
#endif
  const hiopVectorPar& v = dynamic_cast<const hiopVectorPar&>(v_);
  assert(start_index+v.n_local <= n_local);
  memcpy(data+start_index, v.data, v.n_local*sizeof(double));
}

void hiopVectorPar::startingAtCopyFromStartingAt(int start_idx_src, 
						 const hiopVector& v_, 
						 int start_idx_dest)
{
#ifdef HIOP_DEEPCHECKS
  assert(n_local==n && "only for local/non-distributed vectors");
#endif
  assert((start_idx_src>=0 && start_idx_src<this->n_local) || this->n_local==0);
  const hiopVectorPar& v = dynamic_cast<const hiopVectorPar&>(v_);
  assert((start_idx_dest>=0 && start_idx_dest<v.n_local) || v.n_local==0);

  int howManyToCopy = this->n_local - start_idx_src;
  const int howManyToCopyDest = v.n_local-start_idx_dest;
  assert(howManyToCopy <= howManyToCopyDest);
  //howManyToCopy = howManyToCopy <= v.n_local-start_idx_dest ? howManyToCopy : v.n_local-start_idx_dest;
  if(howManyToCopy > howManyToCopyDest) howManyToCopy = howManyToCopyDest;

  assert(howManyToCopy>=0);
  memcpy(data+start_idx_src, v.data+start_idx_dest, howManyToCopy*sizeof(double));
}

void hiopVectorPar::copyToStarting(int start_index, hiopVector& v_)
{
  const hiopVectorPar& v = dynamic_cast<const hiopVectorPar&>(v_);
#ifdef HIOP_DEEPCHECKS
  assert(n_local==n && "are you sure you want to call this?");
#endif
  assert(start_index+v.n_local <= n_local);
  memcpy(v.data, data+start_index, v.n_local*sizeof(double));
}
/* Copy 'this' to v starting at start_index in 'v'. */
void hiopVectorPar::copyToStarting(hiopVector& v_, int start_index/*_in_dest*/)
{
#ifdef HIOP_DEEPCHECKS
  assert(n_local==n && "only for local/non-distributed vectors");
#endif
  const hiopVectorPar& v = dynamic_cast<const hiopVectorPar&>(v_);
  assert(start_index+n_local <= v.n_local);
  memcpy(v.data+start_index, data, n_local*sizeof(double)); 
}

/* copy 'this' (source) starting at 'start_idx_in_src' to 'dest' starting at index 'int start_idx_dest' 
 * If num_elems>=0, 'num_elems' will be copied; if num_elems<0, elements will be copied till the end of
 * either source ('this') or destination ('dest') is reached */
void hiopVectorPar::
startingAtCopyToStartingAt(int start_idx_in_src, hiopVector& dest_, int start_idx_dest, int num_elems/*=-1*/) const
{
#ifdef HIOP_DEEPCHECKS
  assert(n_local==n && "only for local/non-distributed vectors");
#endif  
  assert(start_idx_in_src>=0 && start_idx_in_src<=this->n_local);
#ifdef DEBUG  
  if(start_idx_in_src==this->n_local) assert((num_elems==-1 || num_elems==0));
#endif
  const hiopVectorPar& dest = dynamic_cast<hiopVectorPar&>(dest_);
  assert(start_idx_dest>=0 && start_idx_dest<=dest.n_local);
#ifdef DEBUG  
  if(start_idx_dest==dest.n_local) assert((num_elems==-1 || num_elems==0));
#endif
  if(num_elems<0) {
    num_elems = std::min(this->n_local-start_idx_in_src, dest.n_local-start_idx_dest);
  } else {
    assert(num_elems+start_idx_in_src <= this->n_local);
    assert(num_elems+start_idx_dest   <= dest.n_local);
    //make sure everything stays within bounds (in release)
    num_elems = std::min(num_elems, (int)this->n_local-start_idx_in_src);
    num_elems = std::min(num_elems, (int)dest.n_local-start_idx_dest);
  }

  memcpy(dest.data+start_idx_dest, this->data+start_idx_in_src, num_elems*sizeof(double));
}

void hiopVectorPar::copyTo(double* dest) const
{
  memcpy(dest, this->data, n_local*sizeof(double));
}

double hiopVectorPar::twonorm() const 
{
  int one=1; int n=n_local;
  double nrm = DNRM2(&n,data,&one); 

#ifdef HIOP_USE_MPI
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

  double dotprod=DDOT(&n, this->data, &one, v.data, &one);

#ifdef HIOP_USE_MPI
  double dotprodG;
  int ierr = MPI_Allreduce(&dotprod, &dotprodG, 1, MPI_DOUBLE, MPI_SUM, comm); assert(MPI_SUCCESS==ierr);
  dotprod=dotprodG;
#endif

  return dotprod;
}

double hiopVectorPar::infnorm() const
{
  assert(n_local>=0);
  double nrm=0.;
  if(n_local!=0) {
    nrm=fabs(data[0]);
    double aux;
  
    for(int i=1; i<n_local; i++) {
      aux=fabs(data[i]);
      if(aux>nrm) nrm=aux;
    }
  }
#ifdef HIOP_USE_MPI
  double nrm_glob;
  int ierr = MPI_Allreduce(&nrm, &nrm_glob, 1, MPI_DOUBLE, MPI_MAX, comm); assert(MPI_SUCCESS==ierr);
  return nrm_glob;
#endif

  return nrm;
}

double hiopVectorPar::infnorm_local() const
{
  assert(n_local>=0);
  double nrm=0.;
  if(n_local>0) {
    nrm = fabs(data[0]); 
    double aux;
    
    for(int i=1; i<n_local; i++) {
      aux=fabs(data[i]);
      if(aux>nrm) nrm=aux;
    }
  }
  return nrm;
}


double hiopVectorPar::onenorm() const
{
  double nrm1=0.; for(int i=0; i<n_local; i++) nrm1 += fabs(data[i]);
#ifdef HIOP_USE_MPI
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
#ifdef HIOP_DEEPCHECKS
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
  DSCAL(&n, &num, data, &one);
}

void hiopVectorPar::axpy(double alpha, const hiopVector& x_)
{
  const hiopVectorPar& x = dynamic_cast<const hiopVectorPar&>(x_);
  int one = 1; int n=n_local;
  DAXPY( &n, &alpha, x.data, &one, data, &one );
}

void hiopVectorPar::axzpy(double alpha, const hiopVector& x_, const hiopVector& z_)
{
  const hiopVectorPar& vx = dynamic_cast<const hiopVectorPar&>(x_);
  const hiopVectorPar& vz = dynamic_cast<const hiopVectorPar&>(z_);
#ifdef HIOP_DEEPCHECKS
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
#ifdef HIOP_DEEPCHECKS
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
#ifdef HIOP_DEEPCHECKS
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
  for(long long i=0; i<n_local; i++) data[i]+=c;
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
  DSCAL(&n, &minusOne, data, &one);
}

void hiopVectorPar::invert()
{
  for(int i=0; i<n_local; i++) {
#ifdef HIOP_DEEPCHECKS
    if(fabs(data[i])<1e-35) assert(false);
#endif
    data[i]=1./data[i];
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

/* adds the gradient of the log barrier, namely this=this+alpha*1/select(x) */
void  hiopVectorPar::addLogBarrierGrad(double alpha, const hiopVector& x, const hiopVector& ix)
{
#ifdef HIOP_DEEPCHECKS
  assert(this->n_local == dynamic_cast<const hiopVectorPar&>(ix).n_local);
  assert(this->n_local == dynamic_cast<const hiopVectorPar&>( x).n_local);
#endif
  const double* ix_vec = dynamic_cast<const hiopVectorPar&>(ix).data;
  const double*  x_vec = dynamic_cast<const hiopVectorPar&>( x).data;

  for(int i=0; i<n_local; i++) 
    if(ix_vec[i]==1.) 
      data[i] += alpha/x_vec[i];
}


double hiopVectorPar::linearDampingTerm(const hiopVector& ixleft, const hiopVector& ixright, 
				   const double& mu, const double& kappa_d) const
{
  const double* ixl= (dynamic_cast<const hiopVectorPar&>(ixleft)).local_data_const();
  const double* ixr= (dynamic_cast<const hiopVectorPar&>(ixright)).local_data_const();
#ifdef HIOP_DEEPCHECKS
  assert(n_local==(dynamic_cast<const hiopVectorPar&>(ixleft) ).n_local);
  assert(n_local==(dynamic_cast<const hiopVectorPar&>(ixright) ).n_local);
#endif
  double term=0.0;
  for(long long i=0; i<n_local; i++) {
    if(ixl[i]==1. && ixr[i]==0.) term += data[i];
  }
  term *= mu; 
  term *= kappa_d;
  return term;
}

int hiopVectorPar::allPositive()
{
  int allPos=true, i=0;
  while(i<n_local && allPos) if(data[i++]<=0) allPos=false;

#ifdef HIOP_USE_MPI
  int allPosG;
  int ierr=MPI_Allreduce(&allPos, &allPosG, 1, MPI_INT, MPI_MIN, comm); assert(MPI_SUCCESS==ierr);
  return allPosG;
#endif
  return allPos;
}

bool hiopVectorPar::projectIntoBounds(const hiopVector& xl_, const hiopVector& ixl_, 
				      const hiopVector& xu_, const hiopVector& ixu_,
				      double kappa1, double kappa2)
{
#ifdef HIOP_DEEPCHECKS
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

  const double small_double = std::numeric_limits<double>::min() * 100;

  double aux, aux2;
  for(long long i=0; i<n_local; i++) {
    if(ixl[i]!=0 && ixu[i]!=0) {
      if(xl[i]>xu[i]) return false;
      aux=kappa2*(xu[i]-xl[i])-small_double;
      aux2=xl[i]+fmin(kappa1*fmax(1., fabs(xl[i])),aux);
      if(x0[i]<aux2) {
	x0[i]=aux2;
      } else {
	aux2=xu[i]-fmin(kappa1*fmax(1., fabs(xu[i])),aux);
	if(x0[i]>aux2) {
	  x0[i]=aux2;
	}
      }
#ifdef HIOP_DEEPCHECKS
      //if(x0[i]>xl[i] && x0[i]<xu[i]) {
      //} else {
      //printf("i=%d  x0=%g xl=%g xu=%g\n", i, x0[i], xl[i], xu[i]);
      //}
      assert(x0[i]>xl[i] && x0[i]<xu[i] && "this should not happen -> HiOp bug");
      
#endif
    } else {
      if(ixl[i]!=0.)
	x0[i] = fmax(x0[i], xl[i]+kappa1*fmax(1, fabs(xl[i]))-small_double);
      else 
	if(ixu[i]!=0)
	  x0[i] = fmin(x0[i], xu[i]-kappa1*fmax(1, fabs(xu[i]))-small_double);
	else { /*nothing for free vars  */ }
    }
  }
  return true;
}

/* max{a\in(0,1]| x+ad >=(1-tau)x} */
double hiopVectorPar::fractionToTheBdry(const hiopVector& dx, const double& tau) const 
{
#ifdef HIOP_DEEPCHECKS
  assert((dynamic_cast<const hiopVectorPar&>(dx) ).n_local==n_local);
  assert(tau>0);
  assert(tau<1);
#endif
  double alpha=1.0, aux;
  const double* d = (dynamic_cast<const hiopVectorPar&>(dx) ).local_data_const();
  const double* x = data;
  for(int i=0; i<n_local; i++) {
#ifdef HIOP_DEEPCHECKS
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
#ifdef HIOP_DEEPCHECKS
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
#ifdef HIOP_DEEPCHECKS
    assert(x[i]>0);
#endif
    aux = -tau*x[i]/d[i];
    if(aux<alpha) alpha=aux;
  }
  return alpha;
}

void hiopVectorPar::selectPattern(const hiopVector& ix_)
{
#ifdef HIOP_DEEPCHECKS
  assert((dynamic_cast<const hiopVectorPar&>(ix_) ).n_local==n_local);
#endif
  const double* ix = (dynamic_cast<const hiopVectorPar&>(ix_) ).local_data_const();
  double* x=data;
  for(int i=0; i<n_local; i++) if(ix[i]==0.0) x[i]=0.0;
}

bool hiopVectorPar::matchesPattern(const hiopVector& ix_)
{
#ifdef HIOP_DEEPCHECKS
  assert((dynamic_cast<const hiopVectorPar&>(ix_) ).n_local==n_local);
#endif
  const double* ix = (dynamic_cast<const hiopVectorPar&>(ix_) ).local_data_const();
  int bmatches=true;
  double* x=data;
  for(int i=0; (i<n_local) && bmatches; i++) 
    if(ix[i]==0.0 && x[i]!=0.0) bmatches=false; 

#ifdef HIOP_USE_MPI
  int bmatches_glob=bmatches;
  int ierr=MPI_Allreduce(&bmatches, &bmatches_glob, 1, MPI_INT, MPI_LAND, comm); assert(MPI_SUCCESS==ierr);
  return bmatches_glob;
#endif
  return bmatches;
}

int hiopVectorPar::allPositive_w_patternSelect(const hiopVector& w_)
{
#ifdef HIOP_DEEPCHECKS
  assert((dynamic_cast<const hiopVectorPar&>(w_) ).n_local==n_local);
#endif 
  const double* w = (dynamic_cast<const hiopVectorPar&>(w_) ).local_data_const();
  const double* x=data;
  int allPos=1; 
  for(int i=0; i<n_local && allPos; i++) 
    if(w[i]!=0.0 && x[i]<=0.) allPos=0;
  
#ifdef HIOP_USE_MPI
  int allPosG=allPos;
  int ierr = MPI_Allreduce(&allPos, &allPosG, 1, MPI_INT, MPI_MIN, comm); assert(MPI_SUCCESS==ierr);
  return allPosG;
#endif  
  return allPos;
}

void hiopVectorPar::adjustDuals_plh(const hiopVector& x_, const hiopVector& ix_, const double& mu, const double& kappa)
{
#ifdef HIOP_DEEPCHECKS
  assert((dynamic_cast<const hiopVectorPar&>(x_) ).n_local==n_local);
  assert((dynamic_cast<const hiopVectorPar&>(ix_)).n_local==n_local);
#endif
  const double* x  = (dynamic_cast<const hiopVectorPar&>(x_ )).local_data_const();
  const double* ix = (dynamic_cast<const hiopVectorPar&>(ix_)).local_data_const();
  double* z=data; //the dual
  double a,b;
  for(long long i=0; i<n_local; i++) {
    if(ix[i]==1.) {
      a=mu/x[i]; b=a/kappa; a=a*kappa;
      if(*z<b) 
	*z=b;
      else //z[i]>=b
	if(a<=b) 
	  *z=b;
	else //a>b
	  if(a<*z) *z=a;
          //else a>=z[i] then *z=*z (z[i] does not need adjustment)
    }
    z++;
  }
}

bool hiopVectorPar::isnan() const
{
  for(long long i=0; i<n_local; i++) if(std::isnan(data[i])) return true;
  return false;
}

bool hiopVectorPar::isinf() const
{
  for(long long i=0; i<n_local; i++) if(std::isinf(data[i])) return true;
  return false;
}

bool hiopVectorPar::isfinite() const
{
  for(long long i=0; i<n_local; i++) if(0==std::isfinite(data[i])) return false;
  return true;
}

void hiopVectorPar::print(FILE* file, const char* msg/*=NULL*/, int max_elems/*=-1*/, int rank/*=-1*/) const 
{
  int myrank=0, numranks=1; 
#ifdef HIOP_USE_MPI
  if(rank>=0) {
    int err = MPI_Comm_rank(comm, &myrank); assert(err==MPI_SUCCESS);
    err = MPI_Comm_size(comm, &numranks); assert(err==MPI_SUCCESS);
  }
#endif
  if(myrank==rank || rank==-1) {
    if(max_elems>n_local) max_elems=n_local;

    if(NULL==msg) {
      if(numranks>1)
	fprintf(file, "vector of size %lld, printing %d elems (on rank=%d)\n", n, max_elems, myrank);
      else
	fprintf(file, "vector of size %lld, printing %d elems (serial)\n", n, max_elems);
    } else {
      fprintf(file, "%s ", msg);
    }    
    fprintf(file, "=[");
    max_elems = max_elems>=0?max_elems:n_local;
    for(int it=0; it<max_elems; it++)  fprintf(file, "%24.18e ; ", data[it]);
    fprintf(file, "];\n");
  }
}

};
