#include "hiopLinSolverMA86Z.hpp"

#include "hiop_blasdefs.hpp"

namespace hiop
{
  hiopLinSolverMA86Z::hiopLinSolverMA86Z(hiopMatrixComplexSparseTriplet& sysmat, hiopNlpFormulation* nlp_/*=NULL*/)
    : hiopLinSolver(), keep(NULL), ptr(NULL), row(NULL), order(NULL), vals(NULL), sys_mat(sysmat)
  {
    nlp = nlp_;

    n = sys_mat.n();
    nnz = sys_mat.numberOfNonzeros();

    ma86_default_control_z(&control);

    ptr = new int[n+1];
    row = new int[nnz];
    vals = new double _Complex[nnz];

    order = new int[n];
    for(int i=0; i<n; i++) order[i] = i;
  }
  
  hiopLinSolverMA86Z::~hiopLinSolverMA86Z()
  {
    ma86_finalise(&keep, &control);
    delete[] ptr;
    delete[] row;
    delete[] order;
    delete[] vals;
  }

  int hiopLinSolverMA86Z::matrixChanged()
  {
    assert(n==sys_mat.n());
    assert(nnz == sys_mat.numberOfNonzeros());
    //
    // update ptr, row, and vals from sys_mat
    //
    const int* irow = sys_mat.storage()->i_row();
    const int* jcol = sys_mat.storage()->j_col();
    const std::complex<double>* M = sys_mat.storage()->M();

    //since 
    // 1. sys_mat is upper triangle 
    // 2. sys_mat is ordered on (i,j) (first on i and then on j)
    // 3. ma86 expects lower triangular in column oriented
    //we can 
    // i. do the update in linear time
    //ii. copy sys_mat.j_col to this->row
    //iii.copy sys_mat.M to this->vals  

    //i.
    ptr[0] = 0;
    int next_col=1, it=0;
    for(it=0; it<nnz; it++) {
      if(irow[it]==next_col) {
	ptr[next_col]=it;
	next_col++;
      }
      assert(next_col<=n);
      assert(next_col>=0);
    }
    ptr[n] = nnz;

    //ii.
    memcpy(row, jcol, sizeof(int)*nnz);

    //iii.
    for(int it=0; it<nnz; it++) {
      vals[it] = M[it].real() + M[it].imag()*I;
    }

    //
    //analyze
    //
    ma86_analyse(n, ptr, row, order, &keep, &control, &info);
    if(info.flag < 0) {
      printf("hiopLinSolverMA86Z: Failure during analyse with info.flag = %i\n", info.flag);
      return -1;
    }

    //
    // factorize
    //
    hsl_matrix_type mat_type = HSL_MATRIX_CPLX_SYM;
    /* Factor */
    ma86_factor(mat_type, n, ptr, row, vals, order, &keep, &control, &info, 
		NULL);
    if(info.flag < 0) {
      printf("hiopLinSolverMA86Z: Failure during factor with info.flag = %i\n", info.flag);
      return -1;
    }

    return 0; 
  }

  void hiopLinSolverMA86Z::solve(hiopVector& x)
  {
    assert(false && "not yet implemented"); //not needed; also there is no complex vector at this point
  }

  void hiopLinSolverMA86Z::solve(hiopMatrix& X)
  {
        assert(false && "not yet implemented"); //not needed; 
  }
  
  void hiopLinSolverMA86Z::solve(const hiopMatrixComplexSparseTriplet& B, hiopMatrixComplexDense& X)
  {
    assert(X.n()==B.n());
    assert(n==B.m()); 
    assert(n==X.m()); 

    int ldx = n;
    int nrhs = X.n();

    X.setToZero();

    //copy from B to X. !!! 
    const int* B_irow = B.storage()->i_row();
    const int* B_jcol = B.storage()->j_col();
    const auto*B_M    = B.storage()->M();
    const int B_nnz = B.numberOfNonzeros();


    //this is messy - MA86 expects X column oriented 
    //  "x is a rank-2 array with size x[nrhs][ldx]. On entry, x[j][i] must hold the ith component 
    //   of the jth right-hand side; on exit, it holds the corresponding solution" (MA86 user manual)
    // but our hiopMatrixComplexDense X is row oriented
    //
    // We use an auxiliary buffer nrhs x ldx of _Complex double
    _Complex double* X_buf = new _Complex double[n*nrhs];

    // TODO: solve only for a smaller number of rhs (64, 128) at once (requires calling ma86_solve in a loop)
    // this would also reduce the buffer storage

    for(int itnz=0; itnz<B_nnz; itnz++) {

      assert(B_jcol[itnz]>=0 && B_jcol[itnz]<X.n());
      assert(B_irow[itnz]>=0 && B_irow[itnz]<X.m());

      X_buf[ B_jcol[itnz]*n + B_irow[itnz] ] = B_M[itnz].real() + I*B_M[itnz].imag();
    }
    
    ma86_solve(0, nrhs, ldx, X_buf, order, &keep, &control, &info, NULL);
    if(info.flag < 0) {
      printf("Failure during solve with info.flag = %i\n", info.flag);
    }
    //copy from X_buf to X
    auto** X_M = X.get_M();
    for(int i=0; i<n; i++)
      for(int j=0; i<nrhs; j++)
	X_M[i][j] = X_buf[j*n+i];

    delete[] X_buf;
  }

} //end namespace hiop
