#include "hiopLinSolverUMFPACKZ.hpp"

namespace hiop
{
  hiopLinSolverUMFPACKZ::hiopLinSolverUMFPACKZ(hiopMatrixComplexSparseTriplet& sysmat,
					       hiopNlpFormulation* nlp_/*=NULL*/)
    : m_symbolic(NULL), m_numeric(NULL), m_null(NULL), sys_mat(sysmat), nlp(nlp_)
  {
    n = sys_mat.n();
    nnz = sys_mat.numberOfNonzeros();

    m_colptr = new int[n+1];
    m_rowidx = new int[nnz];
    m_vals   = new double[2*nnz];

    //
    // initialize UMFPACK control
    //
    //get the default control parameters 
    umfpack_zi_defaults(m_control);

    //change the default controls 
    //m_control [UMFPACK_PRL] = 6; //printing/verbosity

    // print the control parameters 
    umfpack_zi_report_control(m_control);

    //others: [UMFPACK_STRATEGY], [UMFPACK_ORDERING]

    //
    // m_info needs no initialization
    //
  }
  hiopLinSolverUMFPACKZ::~hiopLinSolverUMFPACKZ()
  {
    if(m_symbolic) {
      umfpack_di_free_symbolic(&m_symbolic);
      m_symbolic = NULL;
    }

    if(m_numeric) {
      umfpack_di_free_numeric(&m_numeric) ;
      m_numeric = NULL;
    }
    
    delete[] m_colptr;
    delete[] m_rowidx;
    delete[] m_vals;
    //delete[] m_valsim;
  }
  
  int hiopLinSolverUMFPACKZ::matrixChanged()
  {
    assert(n==sys_mat.n());
    assert(nnz == sys_mat.numberOfNonzeros());
    //UMFPACK does not handle zero-dimensioned arrays
    if(n==0) return 0;
    int status;
    
    //
    // copy from sys_mat triplets to UMFPACK's column form sparse format
    //
    const int* irow = sys_mat.storage()->i_row();
    const int* jcol = sys_mat.storage()->j_col();
    const std::complex<double>* M = sys_mat.storage()->M();

    //Note: sys_mat is ordered on (i,j) (first on i and then on j)
    //but we'll just use the umfpack's conversion routine

    // oh boy
    {
      //double Aval[nnz], Avalz[nnz];
      //for(int i=0; i<nnz; i++) {
      //Aval [i] = M[i].real();
      //Avalz[i] = M[i].imag();
      //}
      
      const double* Aval  = reinterpret_cast<const double*>(M);
      // activate the so-called "packed" complex form by passing Avalz=NULL and
      // Avals with real and imaginary interleaved
      //Note that complex<double> interleaves real with imag (as per C++ standard)
      double* Avalz = NULL; 
      status = umfpack_zi_triplet_to_col(n, n, nnz,
					 irow, jcol, Aval, Avalz,
					 m_colptr, m_rowidx, m_vals, (double*) NULL, (int*) NULL);
      if(status<0) {
	umfpack_zi_report_status (m_control, status);
	printf("umfpack_zi_triplet_to_col failed\n");
	return -1;
      }
      // print the column-form of A 
      // printf ("\nA: ");
      // (void) umfpack_zi_report_matrix (n, n, Ap, Ai, Ax, Az, 1, Control) ;
    }
    
    status = umfpack_zi_symbolic(n, n, m_colptr, m_rowidx, m_vals, (double*) NULL,
				 &m_symbolic, m_control, m_info);
    if(status < 0) {
	umfpack_zi_report_info (m_control, m_info) ;
	umfpack_zi_report_status (m_control, status) ;
	return -1;
    }
    // print the symbolic factorization */
    //printf ("\nSymbolic factorization of A: ") ;
    //(void) umfpack_zi_report_symbolic (Symbolic, Control) ;

    
    status = umfpack_zi_numeric(m_colptr, m_rowidx, m_vals, (double*) NULL,
				&m_symbolic, &m_numeric, m_control, m_info);
    if(status < 0) {
	umfpack_zi_report_info (m_control, m_info) ;
	umfpack_zi_report_status (m_control, status) ;
	return -1;
    }
    // print the numeric factorization 
    //printf ("\nNumeric factorization of A: ") ;
    //(void) umfpack_zi_report_numeric (Numeric, Control) ;

    
    return 0;
  }

  void hiopLinSolverUMFPACKZ::solve(hiopVector& x)
  {
    assert(false && "not yet implemented"); //not needed; also there is no complex vector at this point
  }

  void hiopLinSolverUMFPACKZ::solve(hiopMatrix& X)
  {
    assert(false && "not yet implemented"); //not needed; 
  }
  
  void hiopLinSolverUMFPACKZ::solve(const hiopMatrixComplexSparseTriplet& B, hiopMatrixComplexDense& X)
  {
    if(n==0) return;
    int status;
    
    //    status = umfpack_zi_solve(UMFPACK_A, m_colptr, m_rowidx, m_valsre, m_valsim, XXa, xz, b, bz,
  // 	Numeric, Control, Info) ;
  //   umfpack_zi_report_info (Control, Info) ;
  //   umfpack_zi_report_status (Control, status) ;
  //   if (status < 0)
  //   {
  // 	error ("umfpack_zi_solve failed") ;
  //   }
  //   printf ("\nx (solution of Ax=b): ") ;
  //   (void) umfpack_zi_report_vector (n, x, xz, Control) ;
  //   rnorm = resid (FALSE, Ap, Ai, Ax, Az) ;
  //   printf ("maxnorm of residual: %g\n\n", rnorm) ;
  }
  
} //end namespace hiop
