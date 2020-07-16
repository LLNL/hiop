#include "hiopMatrixComplexDense.hpp"

#include "hiop_blasdefs.hpp"

#include <cstring>

namespace hiop
{
  hiopMatrixComplexDense::hiopMatrixComplexDense(const long long& m, 
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
    assert(max_rows>=m_local &&
	   "the requested extra allocation is smaller than the allocation needed by the matrix");
    
    M=new std::complex<double>*[max_rows==0?1:max_rows];
    M[0] = max_rows==0?NULL:new std::complex<double>[max_rows*n_local];
    for(int i=1; i<max_rows; i++)
      M[i]=M[0]+i*n_local;
    
    //! valgrind reports a shit load of errors without this; check this
    for(int i=0; i<max_rows*n_local; i++) M[0][i]=0.0;
    
    //internal buffers 
    _buff_mxnlocal = NULL;
  }
  hiopMatrixComplexDense::~hiopMatrixComplexDense()
  {
    if(_buff_mxnlocal) delete[] _buff_mxnlocal;
    if(M) {
      if(M[0]) delete[] M[0];
      delete[] M;
    }
  }
  
  hiopMatrixComplexDense::hiopMatrixComplexDense(const hiopMatrixComplexDense& dm)
  {
    n_local=dm.n_local; m_local=dm.m_local; n_global=dm.n_global;
    glob_jl=dm.glob_jl; glob_ju=dm.glob_ju;
    comm=dm.comm; myrank=dm.myrank;
    
    //M=new double*[m_local==0?1:m_local];
    max_rows = dm.max_rows;
    M=new std::complex<double>*[max_rows==0?1:max_rows];

    M[0] = max_rows==0?NULL:new std::complex<double>[max_rows*n_local];

    for(int i=1; i<max_rows; i++)
      M[i]=M[0]+i*n_local;
    
    _buff_mxnlocal = NULL;
  }

  void hiopMatrixComplexDense::copyFrom(const hiopMatrixComplexDense& dm)
  {
    assert(n_local==dm.n_local); assert(m_local==dm.m_local); assert(n_global==dm.n_global);
    assert(glob_jl==dm.glob_jl); assert(glob_ju==dm.glob_ju);
    if(NULL==dm.M[0]) {
      M[0] = NULL;
    } else {
      memcpy(M[0], dm.M[0], m_local*n_local*sizeof(std::complex<double>));
    }
  }
  
  void hiopMatrixComplexDense::copyFrom(const std::complex<double>* buffer)
  {
    if(NULL==buffer) {
      M[0] = NULL;
    } else {
      memcpy(M[0], buffer, m_local*n_local*sizeof(std::complex<double>));
    }
  }

  void hiopMatrixComplexDense::copyRowsFrom(const hiopMatrix& src_gen,
					    const long long* rows_idxs,
					    long long n_rows)
  {
    const hiopMatrixComplexDense& src = dynamic_cast<const hiopMatrixComplexDense&>(src_gen);
    assert(n_global==src.n_global);
    assert(n_local==src.n_local);
    assert(n_rows<=src.m_local);
    assert(n_rows == m_local);
    
    // todo //! opt - copy multiple consecutive rows at once ?!?

    //int i should suffice for this container
    for(int i=0; i<n_rows; ++i) {
      memcpy(M[i], src.M[rows_idxs[i]], n_local*sizeof(std::complex<double>));
    }
  }

  void hiopMatrixComplexDense::setToZero()
  {
    setToConstant(0.0);
  }
  void hiopMatrixComplexDense::setToConstant(double c)
  {
    std::complex<double> cc=c;
    setToConstant(cc);
  }
  void hiopMatrixComplexDense::setToConstant(std::complex<double>& c)
  {
    auto buf=M[0];
    //! optimization needed -> use zcopy if exists
    for(int j=0; j<n_local*m_local; j++) *(buf++)=c;
  }

  void hiopMatrixComplexDense::timesVec(std::complex<double> beta_in,
					std::complex<double>* ya_,
					std::complex<double> alpha_in,
					const std::complex<double>* xa_in) const
  {
    char fortranTrans='T';
    int MM=m_local, NN=n_local, incx_y=1;  

    dcomplex beta;
    beta.re = beta_in.real();
    beta.im = beta_in.imag();

    dcomplex alpha;
    alpha.re = alpha_in.real();
    alpha.im = alpha_in.imag();

    dcomplex* ya = reinterpret_cast<dcomplex*>(ya_);
    const dcomplex* xa = reinterpret_cast<const dcomplex*>(xa_in);
    dcomplex* Ma = reinterpret_cast<dcomplex*>(&M[0][0]);
#ifdef HIOP_USE_MPI
    assert(n_local == n_global && "timesVec for distributed matrices not supported/not needed");
#endif
    
    if( MM != 0 && NN != 0 ) {
      // the arguments seem reversed but so is trans='T' 
      // required since we keep the matrix row-wise, while the Fortran/BLAS expects them column-wise
      ZGEMV( &fortranTrans, &NN, &MM, &alpha, Ma, &NN,
	     xa, &incx_y, &beta, ya, &incx_y );
    } else {
      if( MM != 0 ) {
	int one=1; 
	ZSCAL(&NN, &beta, ya, &one);
	
      } else {
	assert(MM==0);
	return;
      }
    }    
  }
  
  bool hiopMatrixComplexDense::isfinite() const
  {
    for(int i=0; i<m_local; i++)
      for(int j=0; j<n_local; j++)
	if(false==std::isfinite(M[i][j].real()) ||
	   false==std::isfinite(M[i][j].imag())) return false;
    return true;
  }
  
  void hiopMatrixComplexDense::print(FILE* f, 
				     const char* msg/*=NULL*/, 
				     int maxRows/*=-1*/, 
				     int maxCols/*=-1*/, 
				     int rank/*=-1*/) const
  {
    if(myrank==rank || rank==-1) {
      if(NULL==f) f=stdout;
      if(maxRows>m_local) maxRows=m_local;
      if(maxCols>n_local) maxCols=n_local;
      
      if(f==NULL) f=stdout;
      
      if(msg) {
	fprintf(f, "%s (local_dims=[%d,%d])\n", msg, m_local,n_local);
      } else { 
	fprintf(f, "hiopMatrixComplexDense::printing max=[%d,%d] (local_dims=[%d,%d], on rank=%d)\n", 
		maxRows, maxCols, m_local,n_local,myrank);
      }
      maxRows = maxRows>=0?maxRows:m_local;
      maxCols = maxCols>=0?maxCols:n_local;
      fprintf(f, "[");
      for(int i=0; i<maxRows; i++) {
	if(i>0) fprintf(f, " ");
	for(int j=0; j<maxCols; j++) 
	  fprintf(f, "%8.5e+%8.5ei; ", M[i][j].real(), M[i][j].imag());
	if(i<maxRows-1)
	  fprintf(f, "; ...\n");
	else
	  fprintf(f, "];\n");
      }
    }
  }

  hiopMatrixComplexDense* hiopMatrixComplexDense::alloc_clone() const
  {
    hiopMatrixComplexDense* c = new hiopMatrixComplexDense(*this);
    return c;
  }
  
  hiopMatrixComplexDense* hiopMatrixComplexDense::new_copy() const
  {
    hiopMatrixComplexDense* c = new hiopMatrixComplexDense(*this);
    c->copyFrom(*this);
    return c;
  }
  double hiopMatrixComplexDense::max_abs_value()
  {
    char norm='M'; int one=1; int N=get_local_size_n() * get_local_size_m();
    hiop::dcomplex* MM = reinterpret_cast<dcomplex*>(M[0]);
    
    double maxv = ZLANGE(&norm, &one, &N, MM, &one, NULL);
    return maxv;
  }

  void hiopMatrixComplexDense::addMatrix(double alpha, const hiopMatrix& X_)
  {
    const hiopMatrixComplexDense& X = dynamic_cast<const hiopMatrixComplexDense&>(X_); 
    addMatrix(std::complex<double>(alpha,0), X);    
  }
  void hiopMatrixComplexDense::addMatrix(const std::complex<double>& alpha, const hiopMatrixComplexDense& X)
  {
#ifdef HIOP_DEEPCHECKS
    assert(m_local==X.m_local);
    assert(n_local==X.n_local);
#endif
    hiop::dcomplex* Mdest= reinterpret_cast<dcomplex*>(M[0]);
    hiop::dcomplex* Msrc = reinterpret_cast<dcomplex*>(X.M[0]);
    hiop::dcomplex a; a.re=alpha.real(); a.im=alpha.imag();
    int N=m_local*n_local, inc=1;
    ZAXPY(&N, &a, Msrc, &inc, Mdest, &inc);
  }

  /* this = this + alpha*X 
   * X is a general sparse matrix in triplet format (rows and cols indexes are assumed to be ordered)
   */
  void hiopMatrixComplexDense::addSparseMatrix(const std::complex<double>& alpha,
					       const hiopMatrixComplexSparseTriplet& X)
  {
    assert(m()==n());
    assert(X.m()==X.n());
    assert(m()==X.m());
    
    if(alpha==0.) return;

    const int* X_irow = X.storage()->i_row();
    const int* X_jcol =  X.storage()->j_col();
    const std::complex<double>* X_M = X.storage()->M();
    
    int nnz = X.numberOfNonzeros();

    for(int it=0; it<nnz; it++) {
      assert(X_irow[it] < m());
      assert(X_jcol[it] < n());
      M[X_irow[it]][X_jcol[it]] += alpha*X_M[it];
    }
  }
  
  /* uppertriangle(this) += uppertriangle(X)
   * where X is a sparse matrix stored in triplet format holding only upper triangle elements*/
  void hiopMatrixComplexDense::
  addSparseSymUpperTriangleToSymDenseMatrixUpperTriangle(const std::complex<double>& alpha,
							 const hiopMatrixComplexSparseTriplet& X)
  {
    assert(m()==n());
    assert(X.m()==X.n());
    assert(m()==X.m());

    if(alpha==0.) return;

    const int* X_irow = X.storage()->i_row();
    const int* X_jcol =  X.storage()->j_col();
    const std::complex<double>* X_M = X.storage()->M();
    
    int nnz = X.numberOfNonzeros();

    for(int it=0; it<nnz; it++) {
      assert(X_irow[it] <= X_jcol[it]);
      M[X_irow[it]][X_jcol[it]] += alpha*X_M[it];
    }
  }

#ifdef HIOP_DEEPCHECKS    
  bool hiopMatrixComplexDense::assertSymmetry(double tol/*=1e-16*/) const
  {
    assert(n_global==n_local && "not yet implemented for distributed matrices");
    if(n_global!=n_local) return false;
    if(n_local!=m_local) return false;

    for(int i=0; i<m_local; i++)
      for(int j=i+1; j<n_local; j++)
	if(std::abs(M[i][j]-M[j][i])>tol)
	  return false;
    return true;
  }
#endif
} //end namespace
