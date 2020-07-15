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

#include "hiopNlpTransforms.hpp"
#include "hiopFactory.hpp"

#include <cmath>
namespace hiop
{

hiopFixedVarsRemover::  
hiopFixedVarsRemover(const hiopVector& xl, 
		     const hiopVector& xu, 
		     const double& fixedVarTol_,
		     const long long& numFixedVars,
		     const long long& numFixedVars_local)
  : n_fixed_vars_local(numFixedVars_local), fixedVarTol(fixedVarTol_),
    Jacc_fs(NULL), Jacd_fs(NULL),
    fs2rs_idx_map(xl.get_local_size()),
    x_rs_ref(NULL), Jacc_rs_ref(NULL), Jacd_rs_ref(NULL)
{
  xl_fs = xl.new_copy();
  xu_fs = xu.new_copy();
  x_fs  = xl.alloc_clone();
  grad_fs = xl.alloc_clone();

  n_fs = xl.get_size();
  n_rs = n_fs-numFixedVars;
};

hiopFixedVarsRemover::~hiopFixedVarsRemover()
{
  delete xl_fs; 
  delete xu_fs;
  delete x_fs;
  delete grad_fs;
  if(Jacc_fs) delete Jacc_fs;
  if(Jacd_fs) delete Jacd_fs;
};

#ifdef HIOP_USE_MPI
/* saves the inter-process distribution of (primal) vectors distribution */
void hiopFixedVarsRemover::setFSVectorDistrib(long long* vec_distrib_in, int num_ranks)
{
  assert(vec_distrib_in!=NULL);
  fs_vec_distrib.resize(num_ranks+1);
  std::copy(vec_distrib_in, vec_distrib_in+num_ranks+1, fs_vec_distrib.begin());
};
/* allocates and returns the reduced-space column partitioning to be used internally by HiOp */
long long* hiopFixedVarsRemover::allocRSVectorDistrib()
{
  int nlen = fs_vec_distrib.size(); //nlen==nranks+1
  assert(nlen>=1);
  long long* rsVecDistrib = new long long[nlen];
  rsVecDistrib[0]=0;
#ifdef HIOP_DEEPCHECKS
  assert(fs_vec_distrib[0]==0);
  assert(nlen>=1);
#endif

#ifdef HIOP_USE_MPI
  int ierr;
#ifdef HIOP_DEEPCHECKS
  int nRanks=-1; 
  ierr = MPI_Comm_size(comm, &nRanks);
  assert(nRanks==nlen-1);
#endif
  //first gather on all ranks the number of variables fixed on each rank
  ierr = MPI_Allgather(&n_fixed_vars_local, 1, MPI_LONG_LONG_INT, rsVecDistrib+1, 1, MPI_LONG_LONG_INT, comm);
  assert(ierr==MPI_SUCCESS);
#else
  assert(nlen==1);
#endif
  assert(rsVecDistrib[0]==0);
  //then accumulate these 
  for(int r=1; r<nlen; r++)
    rsVecDistrib[r] += rsVecDistrib[r-1];
  
  //finally substract these from the full-space index vector distribution 
  for(int r=0; r<nlen; r++)
    rsVecDistrib[r] = fs_vec_distrib[r]-rsVecDistrib[r];

  assert(rsVecDistrib[0]==0);
#ifdef HIOP_DEEPCHECKS
  assert(rsVecDistrib[nlen-1]==n_rs);
#endif  
  return rsVecDistrib;
};
#endif

bool hiopFixedVarsRemover::setupDecisionVectorPart()
{
  int n_fs_local=xl_fs->get_local_size();
  double  *xl_vec= xl_fs->local_data(), *xu_vec= xu_fs->local_data();

  /* build the map from full-space to reduced-space */
  int it_rs=0; 
  for(int i=0;i<n_fs_local; i++) {
    //if(xl_vec[i]==xu_vec[i]) {
    if(fabs(xl_vec[i]-xu_vec[i])<= fixedVarTol*fmax(1.,fabs(xu_vec[i]))) {
      fs2rs_idx_map[i]=-1;
    } else {
      fs2rs_idx_map[i]=it_rs;
      it_rs++;
    }
  }
  assert(it_rs+n_fixed_vars_local==n_fs_local);

  return true;
};

bool hiopFixedVarsRemover::setupConstraintsPart(const int& neq, const int& nineq)
{
  assert(Jacc_fs==NULL && "should not be allocated at this point");
  assert(Jacd_fs==NULL && "should not be allocated at this point");

#ifdef HIOP_USE_MPI
  if(fs_vec_distrib.size())
  {
    Jacc_fs = getMatrixDenseInstance(neq,   n_fs, fs_vec_distrib.data(), comm);
    Jacd_fs = getMatrixDenseInstance(nineq, n_fs, fs_vec_distrib.data(), comm);
  } else {
    Jacc_fs = getMatrixDenseInstance(neq,   n_fs, NULL, comm);
    Jacd_fs = getMatrixDenseInstance(nineq, n_fs, NULL, comm);
  }
#else
  Jacc_fs = getMatrixDenseInstance(neq,   n_fs);
  Jacd_fs = getMatrixDenseInstance(nineq, n_fs);
#endif
  return true;
}

/* "copies" a full space vector/array to a reduced space vector/array */
void hiopFixedVarsRemover::copyFsToRs(const hiopVector& fsVec,  hiopVector& rsVec)
{
  assert(fsVec.get_local_size()==fs2rs_idx_map.size());
  applyInvToArray(fsVec.local_data_const(), rsVec.local_data());
}

void hiopFixedVarsRemover::
copyFsToRs(const hiopInterfaceBase::NonlinearityType* fs, hiopInterfaceBase::NonlinearityType* rs)
{
  int rs_idx;
  for(int i=0; i<fs2rs_idx_map.size(); i++) {
    rs_idx = fs2rs_idx_map[i];
    if(rs_idx>=0) {
      rs[rs_idx] = fs[i];
    } 
  }
}


/* from rs to fs */
void hiopFixedVarsRemover::applyToArray(const double* vec_rs, double* vec_fs)
{
  double* xl_fs_arr = xl_fs->local_data();
  int rs_idx;
  for(int i=0; i<fs2rs_idx_map.size(); i++) {
    rs_idx = fs2rs_idx_map[i];
    if(rs_idx<0) {
      vec_fs[i] = xl_fs_arr[i];
    } else {
      vec_fs[i] = vec_rs[rs_idx];
    }
  }
}

/* from fs to rs */
void hiopFixedVarsRemover::applyInvToArray(const double* x_fs, double* x_rs)
{
  int rs_idx;
  for(int i=0; i<fs2rs_idx_map.size(); i++)  {
    rs_idx = fs2rs_idx_map[i];
    if(rs_idx>=0) {
      x_rs[rs_idx]=x_fs[i];
    }
  }
}

/* from rs to fs */
void hiopFixedVarsRemover::applyToMatrix(const double*const* M_rs, const int& m_in, double** M_fs)
{
  int rs_idx;
  for(int i=0; i<m_in; i++) {
    for(int j=0; j<fs2rs_idx_map.size(); j++) {
      rs_idx = fs2rs_idx_map[j];
      if(rs_idx<0) {
  	M_fs[i][j] = 0.; //really no need to initialize this, these entries will be later ignored
      } else {
  	M_fs[i][j] = M_rs[i][rs_idx];
      }
    }
  }
}

/* from fs to rs */
void hiopFixedVarsRemover::applyInvToMatrix(const double*const* M_fs, const int& m_in, double** M_rs)
{
  int rs_idx;
  for(int i=0; i<m_in; i++) {
    for(int j=0; j<fs2rs_idx_map.size(); j++) {
      rs_idx = fs2rs_idx_map[j];
      if(rs_idx>=0) {
  	M_rs[i][rs_idx] = M_fs[i][j];
      }
    }
  }
}

hiopFixedVarsRelaxer::
hiopFixedVarsRelaxer(const hiopVector& xl, 
		     const hiopVector& xu, 
		     const long long& numFixedVars,
		     const long long& numFixedVars_local)
  : xl_copy(NULL), xu_copy(NULL), n_vars(xl.get_size()), n_vars_local(xl.get_local_size())
{
  //xl_copy = xl.new_copy(); // no need to copy at this point
  //xu_copy = xu.new_copy(); // no need to copy at this point
}
hiopFixedVarsRelaxer::~hiopFixedVarsRelaxer()
{
  if(xl_copy) delete xl_copy;
  if(xu_copy) delete xu_copy;
}

void hiopFixedVarsRelaxer::
relax(const double& fixed_var_tol, const double& fixed_var_perturb, hiopVector& xl, hiopVector& xu)
{
  double *xla=xl.local_data(), *xua=xu.local_data(), *v;
  long long n=xl.get_local_size();
  double xuabs;
  for(long long i=0; i<n; i++) {
    xuabs = fabs(xua[i]);
    if(fabs(xua[i]-xla[i])<= fixed_var_tol*fmax(1.,xuabs)) {

      xua[i] += fixed_var_perturb*fmax(1.,xuabs);
      xla[i] -= fixed_var_perturb*fmax(1.,xuabs);
      //if(xla[i]==xua[i]) {
      // this does not apply anymore
      //if fixed a zero or less,  increase upper bound
      //if fixed at positive val, decrease lower bound
      //if(xua[i]<=0.)      xua[i] += fixed_var_perturb*fmax(1.,fabs(xua[i]));
      //else                xla[i] -= fixed_var_perturb*fmax(1.,fabs(xla[i]));
    }
  }
}


} //end of namespace
