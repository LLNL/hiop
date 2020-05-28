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

#include "hiopMatrix.hpp"

#include <cstdio>
#include <cstring> //for memcpy
#include <cmath>
#include <algorithm>
#include <cassert>

#include "hiop_blasdefs.hpp"

#include "hiopVector.hpp"

namespace hiop
{

hiopMatrixDense::hiopMatrixDense(const long long& m, 
				 const long long& glob_n, 
				 long long* col_part/*=NULL*/, 
				 MPI_Comm comm_/*=MPI_COMM_SELF*/, 
				 const long long& m_max_alloc/*=-1*/)
{
  m_local=m; n_global=glob_n;
  comm=comm_;
  int P=0;
  if(col_part) {
#ifdef HIOP_USE_MPI
    int ierr=MPI_Comm_rank(comm, &P); assert(ierr==MPI_SUCCESS);
#endif
    glob_jl=col_part[P]; glob_ju=col_part[P+1];
  } else {
    glob_jl=0; glob_ju=n_global;
  }
  n_local=glob_ju-glob_jl;

  myrank = P;

  max_rows=m_max_alloc;
  if(max_rows==-1) max_rows=m_local;
  assert(max_rows>=m_local && "the requested extra allocation is smaller than the allocation needed by the matrix");

  //M=new double*[m_local==0?1:m_local];
  M=new double*[max_rows==0?1:max_rows];
  M[0] = max_rows==0?NULL:new double[max_rows*n_local];
  for(int i=1; i<max_rows; i++)
    M[i]=M[0]+i*n_local;

  //! valgrind reports a shit load of errors without this; check this
  for(int i=0; i<max_rows*n_local; i++) M[0][i]=0.0;

  //internal buffers 
  _buff_mxnlocal = NULL;//new double[max_rows*n_local];
}
hiopMatrixDense::~hiopMatrixDense()
{
  if(_buff_mxnlocal) delete[] _buff_mxnlocal;
  if(M) {
    if(M[0]) delete[] M[0];
    delete[] M;
  }
}

hiopMatrixDense::hiopMatrixDense(const hiopMatrixDense& dm)
{
  n_local=dm.n_local; m_local=dm.m_local; n_global=dm.n_global;
  glob_jl=dm.glob_jl; glob_ju=dm.glob_ju;
  comm=dm.comm; myrank=dm.myrank;

  //M=new double*[m_local==0?1:m_local];
  max_rows = dm.max_rows;
  M=new double*[max_rows==0?1:max_rows];
  //M[0] = m_local==0?NULL:new double[m_local*n_local];
  M[0] = max_rows==0?NULL:new double[max_rows*n_local];
  //for(int i=1; i<m_local; i++)
  for(int i=1; i<max_rows; i++)
    M[i]=M[0]+i*n_local;

  _buff_mxnlocal = NULL;
}

void hiopMatrixDense::appendRow(const hiopVectorPar& row)
{
#ifdef HIOP_DEEPCHECKS  
  assert(row.get_local_size()==n_local);
  assert(m_local<max_rows && "no more space to append rows ... should have preallocated more rows.");
#endif
  memcpy(M[m_local], row.local_data_const(), n_local*sizeof(double));
  m_local++;
}

void hiopMatrixDense::copyFrom(const hiopMatrixDense& dm)
{
  assert(n_local==dm.n_local); assert(m_local==dm.m_local); assert(n_global==dm.n_global);
  assert(glob_jl==dm.glob_jl); assert(glob_ju==dm.glob_ju);
  if(NULL==dm.M[0]) {
    M[0] = NULL;
  } else {
    memcpy(M[0], dm.M[0], m_local*n_local*sizeof(double));
  }
}

void hiopMatrixDense::copyFrom(const double* buffer)
{
  if(NULL==buffer) {
    M[0] = NULL;
  } else {
    memcpy(M[0], buffer, m_local*n_local*sizeof(double));
  }
}

void hiopMatrixDense::copyRowsFrom(const hiopMatrixDense& src, int num_rows, int row_dest)
{
#ifdef HIOP_DEEPCHECKS
  assert(row_dest>=0);
  assert(n_global==src.n_global);
  assert(n_local==src.n_local);
  assert(row_dest+num_rows<=m_local);
  assert(num_rows<=src.m_local);
#endif
  if(num_rows>0)
    memcpy(M[row_dest], src.M[0], n_local*num_rows*sizeof(double));
}

void hiopMatrixDense::copyRowsFrom(const hiopMatrix& src_gen, const long long* rows_idxs, long long n_rows)
{
  const hiopMatrixDense& src = dynamic_cast<const hiopMatrixDense&>(src_gen);
  assert(n_global==src.n_global);
  assert(n_local==src.n_local);
  assert(n_rows<=src.m_local);
  assert(n_rows == m_local);

  // todo //! opt -> copy multiple (consecutive rows at the time -> maybe keep blocks of eq and ineq,
  //instead of indexes)

  //int i should suffice for dense matrices
  for(int i=0; i<n_rows; ++i) {
    memcpy(M[i], src.M[rows_idxs[i]], n_local*sizeof(double));
  }
}

  
void hiopMatrixDense::copyBlockFromMatrix(const long i_start, const long j_start,
					  const hiopMatrixDense& src)
{
  assert(n_local==n_global && "this method should be used only in 'serial' mode");
  assert(src.n_local==src.n_global && "this method should be used only in 'serial' mode");
  assert(m_local>=i_start+src.m_local && "the matrix does not fit as a sublock in 'this' at specified coordinates");
  assert(n_local>=j_start+src.n_local && "the matrix does not fit as a sublock in 'this' at specified coordinates");

  //quick returns for empty source matrices
  if(src.n()==0) return;
  if(src.m()==0) return;
#ifdef HIOP_DEEPCHECKS
  assert(i_start<m_local || !m_local);
  assert(j_start<n_local || !n_local);
  assert(i_start>=0); assert(j_start>=0);
#endif
  const size_t buffsize=src.n_local*sizeof(double);
  for(long ii=0; ii<src.m_local; ii++)
    memcpy(M[ii+i_start]+j_start, src.M[ii], buffsize);
}

void hiopMatrixDense::copyFromMatrixBlock(const hiopMatrixDense& src, const int i_block, const int j_block)
{
  assert(n_local==n_global && "this method should be used only in 'serial' mode");
  assert(src.n_local==src.n_global && "this method should be used only in 'serial' mode");
  assert(m_local+i_block<=src.m_local && "the source does not enough rows to fill 'this'");
  assert(n_local+j_block<=src.n_local && "the source does not enough cols to fill 'this'");

  if(n_local==src.n_local) //and j_block=0
    memcpy(M[0], src.M[i_block], n_local*m_local*sizeof(double));
  else {
    for(int i=0; i<m_local; i++)
      memcpy(M[i], src.M[i+i_block]+j_block, n_local*sizeof(double));
  }
}

void hiopMatrixDense::shiftRows(long long shift)
{
  if(shift==0) return;
  if(fabs(shift)==m_local) return; //nothing to shift
  if(m_local<=1) return; //nothing to shift
  
  assert(fabs(shift)<m_local); 

  //at this point m_local should be >=2
  assert(m_local>=2);
  //and
  assert(m_local-fabs(shift)>=1);
#ifdef HIOP_DEEPCHECKS
  double test1=8.3, test2=-98.3;
  if(n_local>0) {
    //not sure if memcpy is copying sequentially on all systems. we check this.
    //let's at least check it
    test1=shift<0 ? M[-shift][0] : M[m_local-shift-1][0];
    test2=shift<0 ? M[-shift][n_local-1] : M[m_local-shift-1][n_local-1];
  }
#endif

  //shift < 0 -> up; shift > 0 -> down
  //if(shift<0) memcpy(M[0], M[-shift], n_local*(m_local+shift)*sizeof(double));
  //else        memcpy(M[shift], M[0],  n_local*(m_local-shift)*sizeof(double));
  if(shift<0) {
    for(int row=0; row<m_local+shift; row++)
      memcpy(M[row], M[row-shift], n_local*sizeof(double));
  } else {
    for(int row=m_local-1; row>=shift; row--) {
      memcpy(M[row], M[row-shift], n_local*sizeof(double));
    }
  }
 
#ifdef HIOP_DEEPCHECKS
  if(n_local>0) {
    assert(test1==M[shift<0?0:m_local-1][0] && "a different copy technique than memcpy is needed on this system");
    assert(test2==M[shift<0?0:m_local-1][n_local-1] && "a different copy technique than memcpy is needed on this system");
  }
#endif
}
void hiopMatrixDense::replaceRow(long long row, const hiopVectorPar& vec)
{
  assert(row>=0); assert(row<m_local);
  long long vec_size=vec.get_local_size();
  memcpy(M[row], vec.local_data_const(), (vec_size>=n_local?n_local:vec_size)*sizeof(double));
}

void hiopMatrixDense::getRow(long long irow, hiopVector& row_vec)
{
  assert(irow>=0); assert(irow<m_local);
  hiopVectorPar& vec=dynamic_cast<hiopVectorPar&>(row_vec);
  assert(n_local==vec.get_local_size());
  memcpy(vec.local_data(), M[irow], n_local*sizeof(double));
}

#ifdef HIOP_DEEPCHECKS
void hiopMatrixDense::overwriteUpperTriangleWithLower()
{
  assert(n_local==n_global && "Use only with local, non-distributed matrices");
  for(int i=0; i<m_local; i++)
    for(int j=i+1; j<n_local; j++)
      M[i][j] = M[j][i];
}
void hiopMatrixDense::overwriteLowerTriangleWithUpper()
{
  assert(n_local==n_global && "Use only with local, non-distributed matrices");
  for(int i=1; i<m_local; i++)
    for(int j=0; j<i; j++)
      M[i][j] = M[j][i];
}
#endif

hiopMatrixDense* hiopMatrixDense::alloc_clone() const
{
  hiopMatrixDense* c = new hiopMatrixDense(*this);
  return c;
}

hiopMatrixDense* hiopMatrixDense::new_copy() const
{
  hiopMatrixDense* c = new hiopMatrixDense(*this);
  c->copyFrom(*this);
  return c;
}

void hiopMatrixDense::setToZero()
{
  setToConstant(0.0);
}
void hiopMatrixDense::setToConstant(double c)
{
  if(!M[0]) {
    assert(m_local==0);
    return;
  }
  double* buf=M[0]; 
  for(int j=0; j<n_local; j++) *(buf++)=c;
  
  buf=M[0]; int inc=1;
  for(int i=1; i<m_local; i++)
   DCOPY(&n_local, buf, &inc, M[i], &inc);
  
  //memcpy(M[i], buf, sizeof(double)*n_local); 
  //memcpy has similar performance as dcopy_; both faster than a loop
}

bool hiopMatrixDense::isfinite() const
{
  for(int i=0; i<m_local; i++)
    for(int j=0; j<n_local; j++)
      if(false==std::isfinite(M[i][j])) return false;
  return true;
}

void hiopMatrixDense::print(FILE* f, 
			    const char* msg/*=NULL*/, 
			    int maxRows/*=-1*/, 
			    int maxCols/*=-1*/, 
			    int rank/*=-1*/) const
{
  if(myrank==rank || rank==-1) {
    if(NULL==f) f=stdout;
    if(maxRows>m_local) maxRows=m_local;
    if(maxCols>n_local) maxCols=n_local;

    if(msg) {
      fprintf(f, "%s (local_dims=[%d,%d])\n", msg, m_local,n_local);
    } else { 
      fprintf(f, "hiopMatrixDense::printing max=[%d,%d] (local_dims=[%d,%d], on rank=%d)\n", 
	      maxRows, maxCols, m_local,n_local,myrank);
    }
    maxRows = maxRows>=0?maxRows:m_local;
    maxCols = maxCols>=0?maxCols:n_local;
    fprintf(f, "[");
    for(int i=0; i<maxRows; i++) {
      if(i>0) fprintf(f, " ");
      for(int j=0; j<maxCols; j++) 
	fprintf(f, "%20.12e ", M[i][j]);
      if(i<maxRows-1)
	fprintf(f, "; ...\n");
      else
	fprintf(f, "];\n");
    }
  }
}

#include <unistd.h>

/*  y = beta * y + alpha * this * x  
 *
 * Sizes: y is m_local, x is n_local, the matrix is m_local x n_global, and the
 * local chunk is m_local x n_local 
*/
void hiopMatrixDense::timesVec(double beta, hiopVector& y_,
			       double alpha, const hiopVector& x_) const
{
  hiopVectorPar& y = dynamic_cast<hiopVectorPar&>(y_);
  const hiopVectorPar& x = dynamic_cast<const hiopVectorPar&>(x_);
#ifdef HIOP_DEEPCHECKS
  assert(y.get_local_size() == m_local);
  assert(y.get_size() == m_local); //y should not be distributed
  assert(x.get_local_size() == n_local);
  assert(x.get_size() == n_global);

  if(beta!=0) assert(y.isfinite()); 
  assert(x.isfinite());
#endif
  
  timesVec(beta, y.local_data(), alpha, x.local_data_const());

#ifdef HIOP_DEEPCHECKS  
  assert(y.isfinite());
#endif
}

void hiopMatrixDense::timesVec(double beta,  double* ya,
			       double alpha, const double* xa) const
{
  char fortranTrans='T';
  int MM=m_local, NN=n_local, incx_y=1;

#ifdef HIOP_USE_MPI
  //only add beta*y on one processor (rank 0)
  if(myrank!=0) beta=0.0; 
#endif

  if( MM != 0 && NN != 0 ) {
    // the arguments seem reversed but so is trans='T' 
    // required since we keep the matrix row-wise, while the Fortran/BLAS expects them column-wise
    DGEMV( &fortranTrans, &NN, &MM, &alpha, &M[0][0], &NN,
	    xa, &incx_y, &beta, ya, &incx_y );
  } else {
    if( MM != 0 ) {
      //y.scale( beta );
      if(beta != 1.) {
	int one=1; 
	DSCAL(&NN, &beta, ya, &one);
      }
    } else {
      assert(MM==0);
      return;
    }
  }
#ifdef HIOP_USE_MPI
  //here m_local is > 0
  double yglob[m_local]; 
  int ierr=MPI_Allreduce(ya, yglob, m_local, MPI_DOUBLE, MPI_SUM, comm); assert(MPI_SUCCESS==ierr);
  memcpy(ya, yglob, m_local*sizeof(double));
#endif

}

/* y = beta * y + alpha * transpose(this) * x */
void hiopMatrixDense::transTimesVec(double beta, hiopVector& y_,
				    double alpha, const hiopVector& x_) const
{
  hiopVectorPar& y = dynamic_cast<hiopVectorPar&>(y_);
  const hiopVectorPar& x = dynamic_cast<const hiopVectorPar&>(x_);
#ifdef HIOP_DEEPCHECKS
  assert(x.get_local_size() == m_local);
  assert(x.get_size() == m_local); //x should not be distributed
  assert(y.get_local_size() == n_local);
  assert(y.get_size() == n_global);
  assert(y.isfinite());
  assert(x.isfinite());
#endif
  transTimesVec(beta, y.local_data(), alpha, x.local_data_const());
}

void hiopMatrixDense::transTimesVec(double beta, double* ya,
				    double alpha, const double* xa) const
{
  char fortranTrans='N';
  int MM=m_local, NN=n_local, incx_y=1;

  if( MM!=0 && NN!=0 ) {
    // the arguments seem reversed but so is trans='T' 
    // required since we keep the matrix row-wise, while the Fortran/BLAS expects them column-wise
    DGEMV( &fortranTrans, &NN, &MM, &alpha, &M[0][0], &NN,
	    xa, &incx_y, &beta, ya, &incx_y );
  } else {
    if( NN != 0 ) {
      //y.scale( beta );
      int one=1; 
      DSCAL(&NN, &beta, ya, &one);
    }
  }
}

/* W = beta*W + alpha*this*X 
 * -- this is 'M' mxn, X is nxk, W is mxk
 *
 * Precondition:
 * - W, this, and X need to be local matrices (not distributed). All multiplications of distributed 
 * matrices needed by HiOp can be done efficiently in parallel using 'transTimesMat'
 */
void hiopMatrixDense::timesMat(double beta, hiopMatrix& W_, double alpha, const hiopMatrix& X_) const
{
#ifndef HIOP_USE_MPI
  timesMat_local(beta,W_,alpha,X_);
#else
  hiopMatrixDense& W = dynamic_cast<hiopMatrixDense&>(W_); double** WM=W.local_data();
  const hiopMatrixDense& X =  dynamic_cast<const hiopMatrixDense&>(X_);
  
  assert(W.m()==this->m());
  assert(X.m()==this->n());
  assert(W.n()==X.n());

  if(W.m()==0 || X.m()==0 || W.n()==0) return;
#ifdef HIOP_DEEPCHECKS  
  assert(W.isfinite());
  assert(X.isfinite());
#endif

  if(X.n_local!=X.n_global || this->n_local!=this->n_global) {
    assert(false && "'timesMat' involving distributed matrices is not needed/supported" &&
	   "also, it cannot be performed efficiently with the data distribution used by this class");
    W.setToConstant(beta);
    return;
  }
  timesMat_local(beta,W_,alpha,X_);
  // if(0==myrank) timesMat_local(beta,W_,alpha,X_);
  // else          timesMat_local(0.,  W_,alpha,X_);

  // int n2Red=W.m()*W.n(); 
  // double* Wglob = new_mxnlocal_buff(); //[n2Red];
  // int ierr = MPI_Allreduce(WM[0], Wglob, n2Red, MPI_DOUBLE, MPI_SUM,comm); assert(ierr==MPI_SUCCESS);
  // memcpy(WM[0], Wglob, n2Red*sizeof(double));
 
#endif

}

/* W = beta*W + alpha*this*X 
 * -- this is 'M' mxn, X is nxk, W is mxk
 */
void hiopMatrixDense::timesMat_local(double beta, hiopMatrix& W_, double alpha, const hiopMatrix& X_) const
{
  const hiopMatrixDense& X = dynamic_cast<const hiopMatrixDense&>(X_);
  hiopMatrixDense& W = dynamic_cast<hiopMatrixDense&>(W_);
#ifdef HIOP_DEEPCHECKS  
  assert(W.m()==this->m());
  assert(X.m()==this->n());
  assert(W.n()==X.n());
  assert(W.isfinite());
  assert(X.isfinite());
#endif
  assert(W.n_local==W.n_global && "requested multiplication is not supported, see timesMat");
  
  /* C = alpha*op(A)*op(B) + beta*C in our case is
     Wt= alpha* Xt  *Mt    + beta*Wt */
  char trans='N'; 
  int M=X.n(), N=m_local, K=X.m();
  int ldx=X.n(), ldm=n_local, ldw=X.n();

  double** XM=X.local_data(); double** WM=W.local_data();
  DGEMM(&trans,&trans, &M,&N,&K, &alpha,XM[0],&ldx, this->M[0],&ldm, &beta,WM[0],&ldw);

  /* C = alpha*op(A)*op(B) + beta*C in our case is
     Wt= alpha* Xt  *Mt    + beta*Wt */

  //char trans='T';
  //int lda=X.m(), ldb=n_local, ldc=W.n();
  //int M=X.n(), N=this->m(), K=this->n_local;

  //DGEMM(&trans,&trans, &M,&N,&K, &alpha,XM[0],&lda, this->M[0],&ldb, &beta,WM[0],&ldc);
}

/* W = beta*W + alpha*this^T*X 
 * -- this is mxn, X is mxk, W is nxk
 */
void hiopMatrixDense::transTimesMat(double beta, hiopMatrix& W_, double alpha, const hiopMatrix& X_) const
{
  const hiopMatrixDense& X = dynamic_cast<const hiopMatrixDense&>(X_);
  hiopMatrixDense& W = dynamic_cast<hiopMatrixDense&>(W_);

  assert(W.m()==n_local);
  assert(X.m()==m_local);
  assert(W.n()==X.n());
#ifdef HIOP_DEEPCHECKS
  assert(W.isfinite());
  assert(X.isfinite());
#endif
  if(W.m()==0) return;

  assert(this->n_global==this->n_local && "requested parallel multiplication is not supported");
  
  /* C = alpha*op(A)*op(B) + beta*C in our case is Wt= alpha* Xt  *M    + beta*Wt */
  char transX='N', transM='T';
  int ldx=X.n_local, ldm=n_local, ldw=W.n_local;
  int M=X.n_local, N=n_local, K=X.m();
  double** XM=X.local_data(); double** WM=W.local_data();
  
  DGEMM(&transX, &transM, &M,&N,&K, &alpha,XM[0],&ldx, this->M[0],&ldm, &beta,WM[0],&ldw);
}

/* W = beta*W + alpha*this*X^T
 * -- this is mxn, X is kxn, W is mxk
 */
void hiopMatrixDense::timesMatTrans_local(double beta, hiopMatrix& W_, double alpha, const hiopMatrix& X_) const
{
  const hiopMatrixDense& X = dynamic_cast<const hiopMatrixDense&>(X_);
  hiopMatrixDense& W = dynamic_cast<hiopMatrixDense&>(W_);
#ifdef HIOP_DEEPCHECKS
  assert(W.m()==m_local);
  //assert(X.n()==n_local);
  assert(W.n()==X.m());
#endif
  assert(W.n_local==W.n_global && "not intended for the case when the result matrix is distributed.");
  if(W.m()==0) return;
  if(W.n()==0) return;
  if(n_local==0) {
    if(beta!=1.0) {
      int one=1; int mn=W.m()*W.n();
      DSCAL(&mn, &beta, W.M[0], &one);
    }
    return;
  }

  /* C = alpha*op(A)*op(B) + beta*C in our case is Wt= alpha* X  *Mt    + beta*Wt */
  char transX='T', transM='N';
  int ldx=n_local;//=X.n(); (modified to support the parallel case)
  int ldm=n_local, ldw=W.n();
  int M=X.m(), N=m_local, K=n_local;
  double** XM=X.local_data(); double** WM=W.local_data();

  DGEMM(&transX, &transM, &M,&N,&K, &alpha,XM[0],&ldx, this->M[0],&ldm, &beta,WM[0],&ldw);
}
/* W = beta*W + alpha*this*X^T */
void hiopMatrixDense::timesMatTrans(double beta, hiopMatrix& W_, double alpha, const hiopMatrix& X_) const
{
  hiopMatrixDense& W = dynamic_cast<hiopMatrixDense&>(W_); 
  assert(W.n_local==W.n_global && "not intended for the case when the result matrix is distributed.");
#ifdef HIOP_DEEPCHECKS
  const hiopMatrixDense& X = dynamic_cast<const hiopMatrixDense&>(X_);
  assert(W.isfinite());
  assert(X.isfinite());
  assert(this->n()==X.n());
  assert(this->m()==W.m());
  assert(X.m()==W.n());
#endif

  if(W.m()==0) return;
  if(W.n()==0) return;

  if(0==myrank) timesMatTrans_local(beta,W_,alpha,X_);
  else          timesMatTrans_local(0.,  W_,alpha,X_);

#ifdef HIOP_USE_MPI
  int n2Red=W.m()*W.n(); 
  double** WM=W.local_data();
  double* Wglob= new_mxnlocal_buff(); //[n2Red];
  int ierr = MPI_Allreduce(WM[0], Wglob, n2Red, MPI_DOUBLE, MPI_SUM, comm); assert(ierr==MPI_SUCCESS);
  memcpy(WM[0], Wglob, n2Red*sizeof(double));
#endif
}
void hiopMatrixDense::addDiagonal(const double& alpha, const hiopVector& d_)
{
  const hiopVectorPar& d = dynamic_cast<const hiopVectorPar&>(d_);
#ifdef HIOP_DEEPCHECKS
  assert(d.get_size()==n());
  assert(d.get_size()==m());
  assert(d.get_local_size()==m_local);
  assert(d.get_local_size()==n_local);
#endif
  const double* dd=d.local_data_const();
  for(int i=0; i<n_local; i++) M[i][i] += alpha*dd[i];
}
void hiopMatrixDense::addDiagonal(const double& value)
{
  for(int i=0; i<n_local; i++) M[i][i] += value;
}
void hiopMatrixDense::addSubDiagonal(const double& alpha, long long start, const hiopVector& d_)
{
  const hiopVectorPar& d = dynamic_cast<const hiopVectorPar&>(d_);
  long long dlen=d.get_size();
#ifdef HIOP_DEEPCHECKS
  assert(start>=0);
  assert(start+dlen<=n_local);
#endif

  const double* dd=d.local_data_const();
  for(int i=start; i<start+dlen; i++) M[i][i] += alpha*dd[i-start];
}

/* add to the diagonal of 'this' (destination) starting at 'start_on_dest_diag' elements of
 * 'd_' (source) starting at index 'start_on_src_vec'. The number of elements added is 'num_elems' 
 * when num_elems>=0, or the remaining elems on 'd_' starting at 'start_on_src_vec'. */
void hiopMatrixDense::addSubDiagonal(int start_on_dest_diag, const double& alpha, 
				     const hiopVector& d_, int start_on_src_vec, int num_elems/*=-1*/)
{
  const hiopVectorPar& d = dynamic_cast<const hiopVectorPar&>(d_);
  if(num_elems<0) num_elems = d.get_size()-start_on_src_vec;
  assert(num_elems <= d.get_size());
  assert(n_local == n_global && "method supported only for non-distributed matrices");
  assert(n_local == m_local  && "method supported only for symmetric matrices");

  assert(start_on_dest_diag>=0 && start_on_dest_diag<m_local);
  num_elems = std::min(num_elems, m_local-start_on_dest_diag);

  const double* dd=d.local_data_const();
  const int nend = start_on_dest_diag+num_elems;
  for(int i=0; i<num_elems; i++)
    M[i+start_on_dest_diag][i+start_on_dest_diag] += alpha*dd[start_on_src_vec+i];
}

void hiopMatrixDense::addSubDiagonal(int start_on_dest_diag, int num_elems, const double& c)
{
  assert(num_elems>=0);
  assert(start_on_dest_diag>=0 && start_on_dest_diag+num_elems<=n_local);
  assert(n_local == n_global && "method supported only for non-distributed matrices");
  assert(n_local == m_local  && "method supported only for symmetric matrices");

  for(int i=0; i<num_elems; i++)
    M[i+start_on_dest_diag][i+start_on_dest_diag] += c;  
}

void hiopMatrixDense::addMatrix(double alpha, const hiopMatrix& X_)
{
  const hiopMatrixDense& X = dynamic_cast<const hiopMatrixDense&>(X_); 
#ifdef HIOP_DEEPCHECKS
  assert(m_local==X.m_local);
  assert(n_local==X.n_local);
#endif

  int N=m_local*n_local, inc=1;
  DAXPY(&N, &alpha, X.M[0], &inc, M[0], &inc);
}

/* block of W += alpha*this 
 * starts are in destination */
void hiopMatrixDense::addToSymDenseMatrixUpperTriangle(int row_start, int col_start, 
						       double alpha, hiopMatrixDense& W) const
{
  assert(row_start>=0 && m()+row_start<=W.m());
  assert(col_start>=0 && n()+col_start<=W.n());
  assert(W.n()==W.m());

  double** WM = W.get_M();
  for(int i=0; i<m_local; i++) {
    const int iW = i+row_start;
    for(int j=0; j<n_local; j++) {
      const int jW = j+col_start;
      assert(iW<=jW && "source entries need to map inside the upper triangular part of destination");
      WM[iW][jW] += alpha*this->M[i][j];
    }
  }
}

/* block of W += alpha*this' */
void hiopMatrixDense::transAddToSymDenseMatrixUpperTriangle(int row_start, int col_start, 
							    double alpha, hiopMatrixDense& W) const
{
  assert(row_start>=0 && n()+row_start<=W.m());
  assert(col_start>=0 && m()+col_start<=W.n());
  assert(W.n()==W.m());

  double** WM = W.get_M();
  for(int ir=0; ir<m_local; ir++) {
    const int jW = ir+col_start;
    for(int jc=0; jc<n_local; jc++) {
      const int iW = jc+row_start;
      assert(iW<=jW && "source entries need to map inside the upper triangular part of destination");
      WM[iW][jW] += alpha*this->M[ir][jc];
    }
  }
}

  /* diagonal block of W += alpha*this with 'diag_start' indicating the diagonal entry of W where
   * 'this' should start to contribute.
   * 
   * For efficiency, only upper triangle of W is updated since this will be eventually sent to LAPACK
   * and only the upper triangle of 'this' is accessed
   * 
   * Preconditions: 
   *  1. this->n()==this->m()
   *  2. W.n() == W.m()
   */
void hiopMatrixDense::
addUpperTriangleToSymDenseMatrixUpperTriangle(int diag_start, 
					      double alpha, hiopMatrixDense& W) const
{
  assert(W.n()==W.m());
  assert(this->n()==this->m());
  assert(diag_start+this->n() <= W.n());
  double** WM = W.get_M();
  for(int i=0; i<n_local; i++) {
    const int iW = i+diag_start;
    for(int j=i; j<m_local; j++) {
      const int jW = j+diag_start;
      assert(iW<=jW && "source entries need to map inside the upper triangular part of destination");
      assert(iW<W.n() && jW<W.m());
      WM[iW][jW] += alpha*this->M[i][j];
    }
  }
}


double hiopMatrixDense::max_abs_value()
{
  char norm='M';
  double maxv = DLANGE(&norm, &n_local, &m_local, M[0], &n_local, NULL);
#ifdef HIOP_USE_MPI
  double maxvg;
  int ierr=MPI_Allreduce(&maxv,&maxvg,1,MPI_DOUBLE,MPI_MAX,comm); assert(ierr==MPI_SUCCESS);
  return maxvg;
#endif
  return maxv;
}

#ifdef HIOP_DEEPCHECKS
bool hiopMatrixDense::assertSymmetry(double tol) const
{
  //must be square
  if(m_local!=n_global) assert(false);

  //symmetry
  for(int i=0; i<n_local; i++)
    for(int j=0; j<n_local; j++) {
      double ij=M[i][j], ji=M[j][i];
      double relerr= fabs(ij-ji)/(1+fabs(ij));
      assert(relerr<tol);
    }
  return true;
}
#endif
};

