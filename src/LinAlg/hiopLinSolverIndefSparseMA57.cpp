#include "hiopLinSolverIndefSparseMA57.hpp"

#include "hiop_blasdefs.hpp"

namespace hiop
{
  hiopLinSolverIndefSparseMA57::hiopLinSolverIndefSparseMA57(const int& n, const int& nnz, hiopNlpFormulation* nlp)
    : hiopLinSolverIndefSparse(n, nnz, nlp),
    m_irowM{nullptr}, m_jcolM{nullptr},
//    m_M{nullptr},
    m_lifact{0}, m_ifact{nullptr}, m_lfact{0}, m_fact{nullptr},
    m_lkeep{0}, m_keep{nullptr},
    m_iwork{nullptr}, m_dwork{nullptr},
    m_ipessimism{1.05}, m_rpessimism{1.05},
    m_n{n}, m_nnz{nnz}
  {
//    m_ipiv = new int[n];
//    m_dwork = LinearAlgebraFactory::createVector(0);

    FNAME(ma57id)( m_cntl, m_icntl );

    /*
    * initialize MA57 parameters
    */
    m_icntl[1-1] = 0;       // don't print warning messages
    m_icntl[2-1] = 0;       // no Warning messages
    m_icntl[4-1] = 1;       // no statistics messages
    m_icntl[5-1] = 0;       // no Print messages.
    m_icntl[6-1] = 2;       // 2 use MC47;
                            // 3 min degree ordering as in MA27;
                            // 4 use Metis;
                            // 5 automatic choice(MA47 or Metis);
    m_icntl[7-1] = 1;       // Pivoting strategy.
    m_icntl[9-1] = 10;      // up to 10 steps of iterative refinement
    m_icntl[11-1] = 16;
    m_icntl[12-1] = 16;
    m_icntl[15-1] = 0;
    m_icntl[16-1] = 0;

    m_cntl[1-1] = 1e-8;     // pivot tolerance

  }
  hiopLinSolverIndefSparseMA57::~hiopLinSolverIndefSparseMA57()
  {
    if(m_irowM)
      delete [] m_irowM;
    if(m_jcolM)
      delete [] m_jcolM;

    if(m_ifact)
      delete [] m_ifact;
    if(m_fact)
      delete [] m_fact;
    if(m_keep)
      delete [] m_keep;
    if(m_iwork)
      delete[] m_iwork;
    if(m_dwork)
      delete[] m_dwork;
  }


  void hiopLinSolverIndefSparseMA57::firstCall()
  {
    assert(m_n==M.n() && M.n()==M.m());
    assert(m_nnz==M.numberOfNonzeros());
    assert(m_n>0);

    m_irowM = new int[m_nnz];
    m_jcolM = new int[m_nnz];
    for(int k=0;k<m_nnz;k++){
      m_irowM[k] = M.i_row()[k]+1;
      m_jcolM[k] = M.j_col()[k]+1;
    }

    m_lkeep = ( m_nnz > m_n ) ? (5 * m_n + 2 *m_nnz + 42) : (6 * m_n + m_nnz + 42);
    m_keep = new int[m_lkeep]{0}; // Initialize to 0 as otherwise MA57ED can sometimes fail

    m_iwork = new int[5 * m_n];
    m_dwork = new double[m_n];
    

    FNAME(ma57ad)( &m_n, &m_nnz, m_irowM, m_jcolM, &m_lkeep, m_keep, m_iwork, m_icntl, m_info, m_rinfo );
        
    m_lfact = (int) (m_rpessimism * m_info[8]);
    m_fact  = new double[m_lfact];

    m_lifact = (int) (m_ipessimism * m_info[9]);
    m_ifact  = new int[m_lifact];

  }


  int hiopLinSolverIndefSparseMA57::matrixChanged()
  {
    assert(m_n==M.n() && M.n()==M.m());
    assert(m_nnz==M.numberOfNonzeros());
    assert(m_n>0);

    nlp_->runStats.linsolv.tmFactTime.start();

    if( !m_keep ) this->firstCall();

    bool done{false}, is_singular{false};
    int num_tries{0};

    do {
      FNAME(ma57bd)( &m_n, &m_nnz, M.M(), m_fact, &m_lfact, m_ifact,
	     &m_lifact, &m_lkeep, m_keep, m_iwork, m_icntl, m_cntl, m_info, m_rinfo );

      switch( m_info[0] ) {
        case 0:
          done = true;
          break;
        case -3: {
          //Failure due to insufficient REAL space on a call to MA57B/BD
          int ic{0}, intTemp{0};
          int lnfact = (int) (m_info[16] * m_rpessimism);
          double * newfact = new double[lnfact];

          FNAME(ma57ed)( &m_n, &ic, m_keep,
                m_fact, &m_info[1], newfact, &lnfact,
                m_ifact, &m_info[1], &intTemp, &m_lifact,
                m_info );

          delete [] m_fact;
          m_fact = newfact;
          m_lfact = lnfact;
          m_rpessimism *= 1.1;
          };
          break;
        case -4: {
          // Failure due to insufficient INTEGER space on a call to MA57B/BD
          int ic = 1;
          int lnifact = (int) (m_info[17] * m_ipessimism);
          int * nifact = new int[ lnifact ];
          FNAME(ma57ed)( &m_n, &ic, m_keep, m_fact, &m_lfact, m_fact, &m_lfact,
               m_ifact, &m_lifact, nifact, &lnifact, m_info );
          delete [] m_ifact;
          m_ifact = nifact;
          m_lifact = lnifact;
          m_ipessimism *= 1.1;
          };
          break;
        case 4: {
          //Matrix is rank deficient on exit from MA57B/BD.
          is_singular=true;
          done=true;
          };
          break;
        default:
          if( m_info[0] >= 0 ) done = true;
          assert( "unknown error!" && 0 );
      } // end switch
      num_tries++;
    } while( !done );

    nlp_->runStats.linsolv.tmInertiaComp.start();
    
    int negEigVal{0};
    if(is_singular)
  	  negEigVal = -1;
    else
	  negEigVal = m_info[24-1];

    nlp_->runStats.linsolv.tmInertiaComp.stop();

    return negEigVal;
  }

  bool hiopLinSolverIndefSparseMA57::solve ( hiopVector& x_ )
  {
    assert(m_n==M.n() && M.n()==M.m());
    assert(m_nnz==M.numberOfNonzeros());
    assert(m_n>0);
    assert(x_.get_size()==M.n());

    nlp_->runStats.linsolv.tmTriuSolves.start();

    int job = 1; // full solve
    int one = 1;
    m_icntl[9-1] = 1; // do one step of iterative refinement

    hiopVectorPar* x = dynamic_cast<hiopVectorPar*>(&x_);
    assert(x != NULL);
    hiopVectorPar* rhs = dynamic_cast<hiopVectorPar*>(x->new_copy());
    hiopVectorPar* resid = dynamic_cast<hiopVectorPar*>(x->new_copy());
    double* dx = x->local_data();
    double* drhs = rhs->local_data();
    double* dresid = resid->local_data();

//    FNAME(ma57cd)( &job, &m_n, m_fact, &m_lfact, m_ifact, &m_lifact,
//    	   &one, drhs, &m_n, m_dwork, &m_n, m_iwork, m_icntl, m_info );

    FNAME(ma57dd)( &job, &m_n, &m_nnz, M.M(), m_irowM, m_jcolM,
        m_fact, &m_lfact, m_ifact, &m_lifact, drhs, dx,
        dresid, m_dwork, m_iwork, m_icntl, m_cntl, m_info, m_rinfo );

    if (m_info[0]<0){
      nlp_->log->printf(hovError, "hiopLinSolverIndefSparseMA57: MA57 returned error %d\n", m_info[0]);
    } else if(m_info[0]>0) {
      nlp_->log->printf(hovError, "hiopLinSolverIndefSparseMA57: MA57 returned warning %d\n", m_info[0]);
    }

    nlp_->runStats.linsolv.tmTriuSolves.stop();

    delete rhs; rhs=nullptr;
    delete resid; resid=nullptr;
    return m_info[0]==0;
  }

} //end namespace hiop
